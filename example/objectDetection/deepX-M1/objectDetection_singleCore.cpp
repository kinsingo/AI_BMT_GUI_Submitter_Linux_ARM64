//Runtime Version v3.2.0

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
    virtual InterfaceType getInterfaceType() override
    {
        return InterfaceType::ObjectDetection;
    }

    virtual Optional_Data getOptionalData() override
    {
        Optional_Data data;
        data.cpu_type = "Rockchip RK3588";
        data.accelerator_type = "M1(NPU) Sync";
        data.submitter = "DeepX";
        return data;
    }

    virtual void initialize(string modelPath) override
    {
        cout << "Initialze() is called" << endl;
        ie = make_shared<dxrt::InferenceEngine>(modelPath);
    }

    virtual VariantType preprocessVisionData(const string &imagePath) override
    {
        cv::Mat input;
        input = cv::imread(imagePath, cv::IMREAD_COLOR);
        cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
        vector<uint8_t> inputBuf(ie->GetInputSize(), 0);
        memcpy(&inputBuf[0], &input.data[0], ie->GetInputSize());
        return inputBuf;
    }

    // Example Code for (YoloV5n/s/m)
    virtual vector<BMTVisionResult> inferVision(const vector<VariantType> &data) override
    {
        vector<BMTVisionResult> queryResult;
        const int querySize = data.size();

        vector<int> strides = {8, 16, 32};

        for (int i = 0; i < querySize; i++)
        {
            vector<uint8_t> inputBuf = get<vector<uint8_t>>(data[i]);
            vector<shared_ptr<dxrt::Tensor>> outputs = ie->Run(inputBuf.data());
            BMTVisionResult result;
            float *output_data = (float *)outputs.front()->data();

            //(25200 * 85) : Yolov7
            //(8400 * 85) : Yolov6
            //(84 * 8400) : Yolov5u, Yolov8, Yolov9, Yolo11, Yolo12
            //(300 * 6) : Yolov10
            vector<float> output(output_data, output_data + (300 * 6));
            result.objectDetectionResult = output;

            queryResult.push_back(result);
        }

        return queryResult;
    }
};
