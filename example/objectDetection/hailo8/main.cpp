//Runtime Version v4.23
#include "hailo/hailort.hpp"
#include "ai_bmt_interface.h"
#include "ai_bmt_gui_caller.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <memory>
#include <string>
#include <algorithm>
#include <thread>
#include <variant>
#include <stdexcept>
#include <mutex>
#include <future>
#include <regex>
#include <cmath>
#include "utils/async_inference.hpp"
#include "utils/utils.hpp"
using namespace hailort;
using namespace std;
#if defined(__unix__)
#include <sys/mman.h>
#endif

constexpr int WIDTH = 640;
constexpr int HEIGHT = 640;

using BMTDataType = vector<float>;

using namespace std;

// ─────────────────────────────────────────────────────────────────────────────
// Model type detection
// ─────────────────────────────────────────────────────────────────────────────
enum class YoloModelType {
    YoloV5,       // anchor-based: output (25200 * 85)
    YoloV7,       // anchor-based: output (25200 * 85), different anchors from v5
    YoloV6,       // anchor-free: output (8400 * 85)
    YoloDFL,      // DFL (v5u/v8/v9/v10/11/12): output (84 * 8400)
    YoloV10,      // v10 one2one: output (300 * 6)
    Unknown
};

YoloModelType detect_model_type(const std::string &model_path) {
    std::string name = model_path;
    // lowercase
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);

    if (std::regex_search(name, std::regex("yolov10"))) return YoloModelType::YoloV10;
    if (std::regex_search(name, std::regex("yolov5[mns]u|yolov8|yolov9|yolo11|yolo12"))) return YoloModelType::YoloDFL;
    if (std::regex_search(name, std::regex("yolov6"))) return YoloModelType::YoloV6;
    if (std::regex_search(name, std::regex("yolov7"))) return YoloModelType::YoloV7;
    if (std::regex_search(name, std::regex("yolov5"))) return YoloModelType::YoloV5;
    return YoloModelType::Unknown;
}

// ─────────────────────────────────────────────────────────────────────────────
// DFL decode helper: 4*reg_max → cx,cy,w,h (in pixel)
// Input: 4 * reg_max floats (dist for l,t,r,b)
// ─────────────────────────────────────────────────────────────────────────────
static float dfl_softmax_sum(const float *dist, int reg_max) {
    // softmax over reg_max values, then weighted sum with index
    float max_val = *std::max_element(dist, dist + reg_max);
    float sum = 0.f, weighted = 0.f;
    for (int i = 0; i < reg_max; ++i) {
        float e = expf(dist[i] - max_val);
        sum += e;
        weighted += e * i;
    }
    return weighted / sum;
}

// Decode DFL box output for one anchor:
// box_raw: 4 * reg_max floats (ltrb distances)
// returns [cx, cy, w, h] in image pixels
static void decode_dfl_box(const float *box_raw, int reg_max, float stride, int grid_x, int grid_y,
                            float &cx, float &cy, float &bw, float &bh) {
    float l = dfl_softmax_sum(box_raw + 0 * reg_max, reg_max);
    float t = dfl_softmax_sum(box_raw + 1 * reg_max, reg_max);
    float r = dfl_softmax_sum(box_raw + 2 * reg_max, reg_max);
    float b = dfl_softmax_sum(box_raw + 3 * reg_max, reg_max);

    float x1 = (grid_x + 0.5f - l) * stride;
    float y1 = (grid_y + 0.5f - t) * stride;
    float x2 = (grid_x + 0.5f + r) * stride;
    float y2 = (grid_y + 0.5f + b) * stride;

    cx = (x1 + x2) * 0.5f;
    cy = (y1 + y2) * 0.5f;
    bw = x2 - x1;
    bh = y2 - y1;
}

// ─────────────────────────────────────────────────────────────────────────────
// Post-process functions per model type
// ─────────────────────────────────────────────────────────────────────────────

// YOLOv5 / YOLOv7 anchor-based
// HEF outputs: 3 tensors, each [H, W, 3, 85], already sigmoid-activated from Hailo
// Output format: (25200 * 85)
static vector<float> postprocess_yolov5_v7(
    const InferenceOutputItem &output_item,
    const vector<vector<pair<float,float>>> &anchors,
    const vector<int> &strides)
{
    vector<float> output;
    output.reserve(25200 * 85);

    for (size_t tensor_index = 0; tensor_index < output_item.output_data_and_infos.size(); ++tensor_index) {
        float *data = reinterpret_cast<float *>(output_item.output_data_and_infos[tensor_index].first);
        const auto &anchorSet = anchors[tensor_index];
        int stride = strides[tensor_index];

        int H = 80 >> tensor_index; // 80, 40, 20
        int W = H;
        int C = 85 * 3;

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                for (int a = 0; a < 3; ++a) {
                    float raw[85];
                    int offset = ((y * W + x) * C) + (a * 85);
                    memcpy(raw, data + offset, sizeof(float) * 85);

                    float pw = anchorSet[a].first;
                    float ph = anchorSet[a].second;

                    raw[0] = (raw[0] * 2.0f - 0.5f + x) * stride;
                    raw[1] = (raw[1] * 2.0f - 0.5f + y) * stride;
                    raw[2] = (raw[2] * 2.0f) * (raw[2] * 2.0f) * pw;
                    raw[3] = (raw[3] * 2.0f) * (raw[3] * 2.0f) * ph;

                    output.insert(output.end(), raw, raw + 85);
                }
            }
        }
    }
    return output;
}

// YOLOv6 anchor-free
// HEF outputs: 6 tensors ordered as [Sigmoid_cls_P3, reg_P3, Sigmoid_cls_P4, reg_P4, Sigmoid_cls_P5, reg_P5]
// Each scale: cls tensor [H,W,80], reg tensor [H,W,4]
// Output format: (8400 * 85)  — [cx,cy,w,h, obj_score=1, cls_0..cls_79]
// Note: YOLOv6 has no objectness score; we set it to max class score for compatibility.
static vector<float> postprocess_yolov6(const InferenceOutputItem &output_item) {
    // Scales: P3(80x80), P4(40x40), P5(20x20) → stride 8,16,32
    const int scales = 3;
    const int grids[3] = {80, 40, 20};
    const int strides[3] = {8, 16, 32};
    const int num_classes = 80;

    vector<float> output;
    output.reserve(8400 * 85);

    // Tensors come in pairs per scale: [cls0, reg0, cls1, reg1, cls2, reg2]
    for (int s = 0; s < scales; ++s) {
        int G = grids[s];
        int stride = strides[s];

        float *cls_data = reinterpret_cast<float *>(output_item.output_data_and_infos[s * 2 + 0].first);
        float *reg_data = reinterpret_cast<float *>(output_item.output_data_and_infos[s * 2 + 1].first);

        for (int y = 0; y < G; ++y) {
            for (int x = 0; x < G; ++x) {
                int idx = y * G + x;

                // reg: [l, t, r, b] relative to stride grid
                float l = reg_data[idx * 4 + 0] * stride;
                float t = reg_data[idx * 4 + 1] * stride;
                float r = reg_data[idx * 4 + 2] * stride;
                float b = reg_data[idx * 4 + 3] * stride;

                float cx = (x + 0.5f) * stride - l + r * 0.5f + l * 0.5f;  // = (x*stride + (r-l)/2)... simplified:
                // standard YOLOv6 decode: x1 = (cx - l), x2 = (cx + r)
                float x1 = (x + 0.5f) * stride - l;
                float y1 = (y + 0.5f) * stride - t;
                float x2 = (x + 0.5f) * stride + r;
                float y2 = (y + 0.5f) * stride + b;
                cx = (x1 + x2) * 0.5f;
                float cy2 = (y1 + y2) * 0.5f;
                float bw = x2 - x1;
                float bh = y2 - y1;

                // cls scores (already sigmoid from Hailo)
                const float *cls_ptr = cls_data + idx * num_classes;
                float max_cls = *std::max_element(cls_ptr, cls_ptr + num_classes);

                // Row: [cx, cy, w, h, obj, cls_0..cls_79]
                output.push_back(cx);
                output.push_back(cy2);
                output.push_back(bw);
                output.push_back(bh);
                output.push_back(max_cls); // use max cls as objectness

                for (int c = 0; c < num_classes; ++c)
                    output.push_back(cls_ptr[c]);
            }
        }
    }
    return output;
}

// YOLOv5u / YOLOv8 / YOLOv9 / YOLO11 DFL
// HEF outputs: 6 tensors [cv2_P3, cv3_P3, cv2_P4, cv3_P4, cv2_P5, cv3_P5]
//   cv2 (box): [H, W, 4*reg_max]   where reg_max=16 → 64 channels
//   cv3 (cls): [H, W, 80]          already sigmoid from Hailo
// Output format: (84 * 8400) — rows are [cx,cy,w,h, cls_0..cls_79], transposed
static vector<float> postprocess_yolo_dfl(const InferenceOutputItem &output_item) {
    const int scales = 3;
    const int grids[3] = {80, 40, 20};
    const int strides[3] = {8, 16, 32};
    const int reg_max = 16;
    const int num_classes = 80;
    const int total_anchors = 8400; // 80*80 + 40*40 + 20*20

    // output shape: 84 * 8400, stored as [84][8400] (column-major anchor index)
    // i.e., output[feat_idx * 8400 + anchor_idx]
    vector<float> output(84 * total_anchors, 0.f);

    int anchor_offset = 0;
    for (int s = 0; s < scales; ++s) {
        int G = grids[s];
        int stride = strides[s];

        float *box_data = reinterpret_cast<float *>(output_item.output_data_and_infos[s * 2 + 0].first);
        float *cls_data = reinterpret_cast<float *>(output_item.output_data_and_infos[s * 2 + 1].first);

        int box_ch = 4 * reg_max; // 64
        // Determine if box tensor has more channels than cls → box is cv2
        // (cv2 has 64 ch, cv3 has 80 ch for COCO)
        // If the order was swapped, auto-detect by checking vstream info or just trust parse order.

        for (int y = 0; y < G; ++y) {
            for (int x = 0; x < G; ++x) {
                int local_idx = y * G + x;
                int anchor_idx = anchor_offset + local_idx;

                // Decode DFL box
                const float *box_raw = box_data + local_idx * box_ch;
                float cx, cy, bw, bh;
                decode_dfl_box(box_raw, reg_max, (float)stride, x, y, cx, cy, bw, bh);

                output[0 * total_anchors + anchor_idx] = cx;
                output[1 * total_anchors + anchor_idx] = cy;
                output[2 * total_anchors + anchor_idx] = bw;
                output[3 * total_anchors + anchor_idx] = bh;

                // Class scores (already sigmoid)
                const float *cls_ptr = cls_data + local_idx * num_classes;
                for (int c = 0; c < num_classes; ++c)
                    output[(4 + c) * total_anchors + anchor_idx] = cls_ptr[c];
            }
        }
        anchor_offset += G * G;
    }
    return output;
}

// YOLOv10 one2one DFL
// HEF outputs: 6 tensors [cv2_P3, cv3_P3, cv2_P4, cv3_P4, cv2_P5, cv3_P5] (one2one head)
// After NMS (top-300 from model), output format: (300 * 6) — [x1,y1,x2,y2, score, class_id]
// Since Hailo doesn't do NMS internally here (we get raw conv outputs), we collect all
static vector<float> postprocess_yolov10(const InferenceOutputItem &output_item) {
    const int scales = 3;
    const int grids[3] = {80, 40, 20};
    const int strides[3] = {8, 16, 32};
    const int reg_max = 16;
    const int num_classes = 80;

    struct Box { float x1, y1, x2, y2, score; int cls_id; };
    vector<Box> boxes;
    boxes.reserve(8400);

    for (int s = 0; s < scales; ++s) {
        int G = grids[s];
        int stride = strides[s];

        float *box_data = reinterpret_cast<float *>(output_item.output_data_and_infos[s * 2 + 0].first);
        float *cls_data = reinterpret_cast<float *>(output_item.output_data_and_infos[s * 2 + 1].first);

        int box_ch = 4 * reg_max;

        for (int y = 0; y < G; ++y) {
            for (int x = 0; x < G; ++x) {
                int local_idx = y * G + x;

                const float *box_raw = box_data + local_idx * box_ch;
                float cx, cy, bw, bh;
                decode_dfl_box(box_raw, reg_max, (float)stride, x, y, cx, cy, bw, bh);

                const float *cls_ptr = cls_data + local_idx * num_classes;
                int best_cls = (int)(std::max_element(cls_ptr, cls_ptr + num_classes) - cls_ptr);
                float best_score = cls_ptr[best_cls];

                Box b;
                b.x1 = cx - bw * 0.5f;
                b.y1 = cy - bh * 0.5f;
                b.x2 = cx + bw * 0.5f;
                b.y2 = cy + bh * 0.5f;
                b.score = best_score;
                b.cls_id = best_cls;
                boxes.push_back(b);
            }
        }
    }

    // Sort by score descending, keep top-300
    std::sort(boxes.begin(), boxes.end(), [](const Box &a, const Box &b){ return a.score > b.score; });
    if (boxes.size() > 300) boxes.resize(300);

    vector<float> output(300 * 6, 0.f);
    for (size_t i = 0; i < boxes.size(); ++i) {
        output[i * 6 + 0] = boxes[i].x1;
        output[i * 6 + 1] = boxes[i].y1;
        output[i * 6 + 2] = boxes[i].x2;
        output[i * 6 + 3] = boxes[i].y2;
        output[i * 6 + 4] = boxes[i].score;
        output[i * 6 + 5] = (float)boxes[i].cls_id;
    }
    return output;
}

hailo_status run_preprocess(std::shared_ptr<BoundedTSQueue<PreprocessedFrameItem>> preprocessed_queue, const vector<VariantType> &data, size_t start, size_t end)
{
    for (int i = start; i < end; i++)
    {
        vector<uint8_t> inputBuf = get<vector<uint8_t>>(data[i]);
        auto preprocessed_frame_item = create_preprocessed_frame_item(inputBuf, WIDTH, HEIGHT, i);
        preprocessed_queue->push(preprocessed_frame_item);
    }
    preprocessed_queue->stop();
    return HAILO_SUCCESS;
}

hailo_status run_inference_async(std::shared_ptr<BoundedTSQueue<PreprocessedFrameItem>> preprocessed_queue, shared_ptr<AsyncModelInfer> model)
{
    while (true)
    {
        PreprocessedFrameItem item;
        if (!preprocessed_queue->pop(item))
            break;
        model->infer(std::make_shared<vector<uint8_t>>(item.resized_for_infer), item.frame_idx);
    }
    return HAILO_SUCCESS;
}

//(25200 * 85) : Yolov5, Yolov7
//(8400 * 85) : Yolov6
//(84 * 8400) : Yolov5u, Yolov8, Yolov9, Yolo11, Yolo12
//(300 * 6) : Yolov10
hailo_status run_post_process_worker(std::shared_ptr<BoundedTSQueue<InferenceOutputItem>> results_queue,
                                      vector<BMTVisionResult> &batchResult,
                                      std::atomic<size_t> &processed_count,
                                      size_t bs,
                                      YoloModelType model_type)
{
    // YOLOv5 anchors (standard COCO-trained)
    static const vector<vector<pair<float, float>>> v5_anchors = {
        {{10, 13}, {16, 30}, {33, 23}},
        {{30, 61}, {62, 45}, {59, 119}},
        {{116, 90}, {156, 198}, {373, 326}}
    };
    // YOLOv7 anchors (different from v5)
    static const vector<vector<pair<float, float>>> v7_anchors = {
        {{12, 16}, {19, 36}, {40, 28}},
        {{36, 75}, {76, 55}, {72, 146}},
        {{142, 110}, {192, 243}, {459, 401}}
    };
    static const vector<int> strides = {8, 16, 32};

    while (true)
    {
        InferenceOutputItem output_item;
        if (!results_queue->pop(output_item))
            break;

        auto frame_idx = output_item.frame_idx;
        vector<float> output;

        switch (model_type)
        {
        case YoloModelType::YoloV5:
            output = postprocess_yolov5_v7(output_item, v5_anchors, strides);
            break;
        case YoloModelType::YoloV7:
            output = postprocess_yolov5_v7(output_item, v7_anchors, strides);
            break;
        case YoloModelType::YoloV6:
            output = postprocess_yolov6(output_item);
            break;
        case YoloModelType::YoloDFL:
            output = postprocess_yolo_dfl(output_item);
            break;
        case YoloModelType::YoloV10:
            output = postprocess_yolov10(output_item);
            break;
        default:
            output = postprocess_yolov5_v7(output_item, v5_anchors, strides);
            break;
        }

        batchResult[frame_idx].objectDetectionResult = std::move(output);
        if (processed_count.fetch_add(1, std::memory_order_relaxed) + 1 == bs)
            results_queue->stop();
    }
    return HAILO_SUCCESS;
}

class Virtual_Submitter_Implementation : public AI_BMT_Interface
{
    const size_t MAX_QUEUE_SIZE = 80; // must bigger than or equal to residual set(80)
    std::shared_ptr<BoundedTSQueue<PreprocessedFrameItem>> preprocessed_queue;
    std::shared_ptr<BoundedTSQueue<InferenceOutputItem>> results_queue;
    shared_ptr<AsyncModelInfer> model;
    YoloModelType model_type = YoloModelType::Unknown;

public:
    Virtual_Submitter_Implementation()
    {
    }

    virtual InterfaceType getInterfaceType() override
    {
        return InterfaceType::ObjectDetection;
    }

    virtual Optional_Data getOptionalData() override
    {
        Optional_Data data;
        data.cpu_type = "Broadcom BCM2712 quad-core Arm Cortex A76 processor @ 2.4GHz"; // e.g., Intel i7-9750HF
        data.accelerator_type = "Hailo-8";                                              // e.g., DeepX M1(NPU)
        data.submitter = "Hailo";                                                       // e.g., DeepX
        data.cpu_core_count = "4";                                                      // e.g., 16
        data.cpu_ram_capacity = "8GB";                                                  // e.g., 32GB
        data.cooling = "Air";                                                           // e.g., Air, Liquid, Passive
        data.cooling_option = "Active";                                                 // e.g., Active, Passive (Active = with fan/pump, Passive = without fan)
        data.cpu_accelerator_interconnect_interface = "PCIe 3.0 4-lane";                // e.g., PCIe Gen5 x16
        data.benchmark_model = "YoloV5";                                                // e.g., ResNet-50
        data.operating_system = "Ubuntu 24.04.2 LTS";                                   // e.g., Ubuntu 20.04.5 LTS
        return data;
    }

    virtual void initialize(string modelPath) override
    {
        model_type = detect_model_type(modelPath);
        model = make_shared<AsyncModelInfer>();
        model->crt();
        model->PathAndResult(modelPath);
        preprocessed_queue = std::make_shared<BoundedTSQueue<PreprocessedFrameItem>>(MAX_QUEUE_SIZE);
        results_queue = std::make_shared<BoundedTSQueue<InferenceOutputItem>>(MAX_QUEUE_SIZE);
        model->configure(results_queue);
    }

    virtual VariantType preprocessVisionData(const string &imagePath) override
    {
        cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
        // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        if (img.rows != HEIGHT || img.cols != WIDTH) {
            cv::resize(img, img, cv::Size(WIDTH, HEIGHT));
        }
        vector<uint8_t> inputBuf(HEIGHT * WIDTH * 3);
        std::memcpy(inputBuf.data(), img.data, HEIGHT * WIDTH * 3);
        return inputBuf;
    }

    virtual vector<BMTVisionResult> inferVision(const vector<VariantType> &data) override
    {
        size_t frame_count = data.size();
        vector<BMTVisionResult> batchResult(frame_count);

        // Use half the available cores for postprocess workers (others: preprocess + inference dispatch)
        const int num_pp_threads = std::max(1, (int)std::thread::hardware_concurrency() / 2);

        for (size_t i = 0; i < frame_count; i += MAX_QUEUE_SIZE)
        {
            size_t currentBatchSize = min(MAX_QUEUE_SIZE, frame_count - i);
            size_t start = i;
            size_t end = i + currentBatchSize;

            std::atomic<size_t> processed_count{0};

            // Launch postprocess workers first so they are ready when results arrive
            vector<std::thread> pp_threads;
            pp_threads.reserve(num_pp_threads);
            for (int t = 0; t < num_pp_threads; ++t)
                pp_threads.emplace_back(run_post_process_worker,
                                        results_queue,
                                        std::ref(batchResult),
                                        std::ref(processed_count),
                                        currentBatchSize,
                                        model_type);

            auto preprocess_future = std::async(std::launch::async, run_preprocess,
                                                preprocessed_queue, std::ref(data), start, end);
            auto inference_future  = std::async(std::launch::async, run_inference_async,
                                                preprocessed_queue, model);

            preprocess_future.get();
            inference_future.get();
            for (auto &t : pp_threads)
                t.join();
            preprocessed_queue->reset();
            results_queue->reset();
            model->clear();
        }
        return batchResult;
    }
};

int main(int argc, char *argv[])
{
    try
    {
        shared_ptr<AI_BMT_Interface> interface = make_shared<Virtual_Submitter_Implementation>();
        return AI_BMT_GUI_CALLER::call_BMT_GUI_For_Single_Task(argc, argv, interface);
    }
    catch (const exception &ex)
    {
        cout << ex.what() << endl;
    }
}

