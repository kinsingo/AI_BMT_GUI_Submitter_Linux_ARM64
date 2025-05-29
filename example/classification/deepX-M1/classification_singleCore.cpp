#include "ai_bmt_gui_caller.h"
#include "ai_bmt_interface.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "dxrt/dxrt_api.h"
using namespace std;

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
        //  input = cv::imread(imagePath, cv::IMREAD_COLOR);
        //  cv::resize(input, input, cv::Size(input_w, input_h));
        cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
        vector<uint8_t> inputBuf(input_h * (input_w * input_c + align_factor));
        for (int y = 0; y < input_h; y++)
        {
            memcpy(&inputBuf[y * (input_w * input_c + align_factor)], &input.data[y * input_w * input_c], input_w * input_c);
        }
        return inputBuf;
    }

    int getArgMax(float *output_data, int number_of_classes)
    {
        int max_idx = 0;
        for (int i = 0; i < number_of_classes; i++)
        {
            if (output_data[max_idx] < output_data[i])
            {
                max_idx = i;
            }
        }
        return max_idx;
    }

    virtual vector<BMTResult> runInference(const vector<VariantType> &data) override
    {
        vector<BMTResult> batchResult;
        const int batchSize = data.size();
        for (int i = 0; i < batchSize; i++)
        {
            vector<uint8_t> inputBuf = get<vector<uint8_t>>(data[i]);
            auto outputs = ie->Run(inputBuf.data());
            BMTResult result;
            int index = (ie->outputs().front().type() == dxrt::DataType::FLOAT) ? getArgMax((float *)outputs.front()->data(), 1000) : *(uint16_t *)outputs.front()->data();
            result.Classification_ImageNet_PredictedIndex_0_to_999 = index; // temporary value(0~999) is assigned here. It should be replaced with the actual predicted value.
            batchResult.push_back(result);
        }
        return batchResult;
    }
};
