#include "ai_bmt_gui_caller.h"
#include "ai_bmt_interface.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "dxrt/dxrt_api.h"
using namespace std;
using namespace cv;

class ObjectDetection_Implementation_SingleCore : public AI_BMT_Interface
{
    shared_ptr<dxrt::InferenceEngine> ie;
    int align_factor;
    int input_w = 640, input_h = 640, input_c = 3;

public:
    virtual Optional_Data getOptionalData() override
    {
        Optional_Data data;
        data.cpu_type = "Rockchip RK3588";
        data.accelerator_type = "M1(NPU) Sync";
        data.submitter = "DeepX";
        return data;
    }

    virtual void Initialize(string modelPath) override
    {
        cout << "Initialze() is called" << endl;
        ie = make_shared<dxrt::InferenceEngine>(modelPath);
        align_factor = ((int)(input_w * input_c)) & (-64);
        align_factor = (input_w * input_c) - align_factor;
    }

    virtual VariantType convertToPreprocessedDataForInference(const string &imagePath) override
    {
        cv::Mat input;
        input = cv::imread(imagePath, cv::IMREAD_COLOR);
        cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
        return std::vector<uint8_t>(input.data, input.data + input.total() * input.elemSize());
    }

    inline float sigmoid(float x)
    {
        return 1.0f / (1.0f + exp(-x));
    }

    // Example Code for (YoloV5n/s/m)
    virtual vector<BMTResult> runInference(const vector<VariantType> &data) override
    {
        vector<BMTResult> queryResult;
        const int querySize = data.size();

        // YOLOv5n Anchor definitions (standard)
        vector<vector<pair<float, float>>> anchors = {
            {{10, 13}, {16, 30}, {33, 23}},     // P3: 80x80
            {{30, 61}, {62, 45}, {59, 119}},    // P4: 40x40
            {{116, 90}, {156, 198}, {373, 326}} // P5: 20x20
        };

        vector<int> strides = {8, 16, 32};

        for (int i = 0; i < querySize; i++)
        {
            vector<uint8_t> inputBuf = get<vector<uint8_t>>(data[i]);
            vector<shared_ptr<dxrt::Tensor>> outputs = ie->Run(inputBuf.data());

            BMTResult result;
            vector<float *> feature_maps;
            for (int i = 0; i < outputs.size(); i++)
                feature_maps.push_back((float *)outputs[i]->data());

            vector<float> output;
            output.reserve(25200 * 85); // 2142000
            for (int i = 0; i < outputs.size(); ++i)
            {
                auto out = outputs[i];
                auto shape = out->shape(); // [1, H, W, 256]
                float *data = (float *)out->data();

                int H = shape[1];
                int W = shape[2];
                int C = shape[3]; // 256

                const auto &anchorSet = anchors[i];
                int stride = strides[i];

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
                            raw[0] = (sigmoid(raw[0]) * 2.0f - 0.5f + x) * stride;
                            raw[1] = (sigmoid(raw[1]) * 2.0f - 0.5f + y) * stride;

                            // size
                            raw[2] = pow(sigmoid(raw[2]) * 2.0f, 2.0f) * pw;
                            raw[3] = pow(sigmoid(raw[3]) * 2.0f, 2.0f) * ph;

                            // confidence
                            raw[4] = sigmoid(raw[4]);

                            // class probs
                            for (int c = 5; c < 85; ++c)
                                raw[c] = sigmoid(raw[c]);

                            // confidence threshold나 NMS는 나중에 사용
                            output.insert(output.end(), raw, raw + 85);
                        }
                    }
                }
            }
            // cout << "output size: " << output.size() << " (expect 25200 × 85 = 2142000)" << endl;
            result.objectDetectionResult = output;
            queryResult.push_back(result);
        }

        return queryResult;
    }
};
