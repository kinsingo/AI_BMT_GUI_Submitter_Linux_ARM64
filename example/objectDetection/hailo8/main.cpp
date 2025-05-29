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

hailo_status run_post_process(std::shared_ptr<BoundedTSQueue<InferenceOutputItem>> results_queue, vector<BMTResult> &batchResult, size_t bs)
{
    // YOLOv5n Anchor definitions (standard)
    vector<vector<pair<float, float>>> anchors = {
        {{10, 13}, {16, 30}, {33, 23}},     // P3: 80x80
        {{30, 61}, {62, 45}, {59, 119}},    // P4: 40x40
        {{116, 90}, {156, 198}, {373, 326}} // P5: 20x20
    };

    vector<int> strides = {8, 16, 32};

    size_t i = 0;
    while (true)
    {
        InferenceOutputItem output_item;
        if (!results_queue->pop(output_item))
            break;

        auto frame_idx = output_item.frame_idx;
        vector<float> output;
        output.reserve(25200 * 85); // 2142000

        for (size_t tensor_index = 0; tensor_index < output_item.output_data_and_infos.size(); ++tensor_index)
        {
            float *data = reinterpret_cast<float *>(output_item.output_data_and_infos[tensor_index].first);
            auto &anchorSet = anchors[tensor_index];
            int stride = strides[tensor_index];

            int H = 80 >> tensor_index; // 80, 40, 20
            int W = 80 >> tensor_index;
            int C = 85 * 3;

            for (int y = 0; y < H; ++y)
            {
                for (int x = 0; x < W; ++x)
                {
                    for (int a = 0; a < 3; ++a)
                    {
                        float raw[85];
                        int offset = ((y * W + x) * C) + (a * 85);
                        memcpy(raw, data + offset, sizeof(float) * 85);

                        // anchor
                        float pw = anchorSet[a].first;
                        float ph = anchorSet[a].second;

                        // center
                        raw[0] = (raw[0] * 2.0f - 0.5f + x) * stride;
                        raw[1] = (raw[1] * 2.0f - 0.5f + y) * stride;

                        // size
                        raw[2] = pow(raw[2] * 2.0f, 2.0f) * pw;
                        raw[3] = pow(raw[3] * 2.0f, 2.0f) * ph;

                        output.insert(output.end(), raw, raw + 85);
                    }
                }
            }
        }

        BMTResult result;
        result.objectDetectionResult = output;
        batchResult[frame_idx] = result;
        if ((++i) == bs)
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

public:
    Virtual_Submitter_Implementation()
    {
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

    virtual void Initialize(string modelPath) override
    {
        model = make_shared<AsyncModelInfer>();
        model->crt();
        model->PathAndResult(modelPath);
        preprocessed_queue = std::make_shared<BoundedTSQueue<PreprocessedFrameItem>>(MAX_QUEUE_SIZE);
        results_queue = std::make_shared<BoundedTSQueue<InferenceOutputItem>>(MAX_QUEUE_SIZE);
        model->configure(results_queue);
    }

    virtual VariantType convertToPreprocessedDataForInference(const string &imagePath) override
    {
        cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        vector<uint8_t> inputBuf(HEIGHT * WIDTH * 3);
        std::memcpy(inputBuf.data(), img.data, HEIGHT * WIDTH * 3);
        return inputBuf;
    }

    virtual vector<BMTResult> runInference(const vector<VariantType> &data) override
    {
        size_t frame_count = data.size();
        vector<BMTResult> batchResult(frame_count);
        for (size_t i = 0; i < frame_count; i += MAX_QUEUE_SIZE)
        {
            size_t currentBatchSize = min(MAX_QUEUE_SIZE, frame_count - i);
            size_t start = i;
            size_t end = i + currentBatchSize;
            auto preprocess_thread = std::async(run_preprocess,
                                                preprocessed_queue,
                                                std::ref(data),
                                                start,
                                                end);
            auto inference_thread = std::async(run_inference_async,
                                               preprocessed_queue,
                                               model);
            auto output_parser_thread = std::async(run_post_process,
                                                   results_queue,
                                                   std::ref(batchResult),
                                                   currentBatchSize);
            hailo_status status = wait_and_check_threads(
                preprocess_thread, "Preprocess",
                inference_thread, "Inference",
                output_parser_thread, "Postprocess ");

            if (status != HAILO_SUCCESS)
            {
                throw std::runtime_error("Inference failed");
            }
        }
        preprocessed_queue->reset();
        results_queue->reset();
        model->clear();
        return batchResult;
    }
};

int main(int argc, char *argv[])
{
    filesystem::path exePath = filesystem::absolute(argv[0]).parent_path(); // Get the current executable file path
    filesystem::path model_path = exePath / "Model" / "ObjectDetection" / "yolov5n_opset12_normalized_quantized.hef";
    string modelPath = model_path.string();
    try
    {
        // sample_latency_average: 46.3471 ms (without post processing)
        // sample_latency_average: 64.8433 ms (with post processing)

        shared_ptr<AI_BMT_Interface> interface = make_shared<Virtual_Submitter_Implementation>();
        AI_BMT_GUI_CALLER caller(interface, modelPath);
        return caller.call_BMT_GUI(argc, argv);
    }
    catch (const exception &ex)
    {
        cout << ex.what() << endl;
    }
}
