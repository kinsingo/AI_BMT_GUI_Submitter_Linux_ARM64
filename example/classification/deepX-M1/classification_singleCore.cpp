#include "ai_bmt_gui_caller.h"
#include "ai_bmt_interface.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "dxrt/dxrt_api.h"
using namespace std;
using namespace cv;
class Classification_Implementation_SingleCore : public AI_BMT_Interface
{
    shared_ptr<dxrt::InferenceEngine> ie;
    int align_factor;
    int input_w = 224, input_h = 224, input_c = 3;

public:
    virtual Optional_Data getOptionalData() override
    {
        Optional_Data data;
        data.cpu_type = "Rockchip RK3588";
        data.accelerator_type = "M1(NPU) Sync";
        data.submitter = "DeepX";
        data.benchmark_model = "regnet_y_800mf_opset10.dxnn";
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
        vector<uint8_t> inputBuf(input_h * (input_w * input_c + align_factor));
        for (int y = 0; y < input_h; y++)
        {
            memcpy(&inputBuf[y * (input_w * input_c + align_factor)], &input.data[y * input_w * input_c], input_w * input_c);
        }
        return inputBuf;
    }

    virtual vector<BMTResult> runInference(const vector<VariantType> &data) override
    {
        vector<BMTResult> queryResult;
        const int querySize = data.size();
        for (int i = 0; i < querySize; i++)
        {
            vector<uint8_t> inputBuf = get<vector<uint8_t>>(data[i]);
            BMTResult result;
            auto outputs = ie->Run(inputBuf.data());
            float *output_data = (float *)outputs.front()->data();
            vector<float> output(output_data, output_data + 1000);
            result.classProbabilities = output;
            queryResult.push_back(result);
        }
        return queryResult;
    }
};
