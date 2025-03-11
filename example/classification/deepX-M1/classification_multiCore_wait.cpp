#include "snu_bmt_gui_caller.h"
#include "snu_bmt_interface.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "dxrt/dxrt_api.h"
#include <getopt.h>
#include <future>
#include <thread>
#include <iostream>
#include <queue>
#include <atomic>
#include <mutex>
#include <condition_variable>

using namespace std;

class Classification_Implementation_MultiCore_Wait : public SNU_BMT_Interface
{
    string modelPath;
    shared_ptr<dxrt::InferenceEngine> ie;
    int align_factor;
    int input_w = 224, input_h = 224, input_c = 3;

public:
    Classification_Implementation_MultiCore_Wait(string modelPath)
    {
        this->modelPath = modelPath;
    }

    virtual Optional_Data getOptionalData() override
    {
        Optional_Data data;
        data.cpu_type = "Rockchip RK3588";
        data.accelerator_type = "M1(NPU) Async(Wait)";
        data.submitter = "DeepX";
        return data;
    }

    virtual void Initialize() override
    {
        cout << "Initialze() is called" << endl;
        align_factor = ((int)(input_w * input_c)) & (-64);
        align_factor = (input_w * input_c) - align_factor;
        ie = make_shared<dxrt::InferenceEngine>(modelPath);
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
        int batchSize = data.size();

        vector<int> reqIds;
        vector<vector<uint8_t>> inputBufs(batchSize); // the inputBuf's memory must be maintained until the callback function or wait is called.
        const int maxConcurrentRequests = 3;

        for (int i = 0; i < batchSize; i += maxConcurrentRequests)
        {
            reqIds.clear();
            int currentBatchSize = min(maxConcurrentRequests, batchSize - i);

            // Start a batch of RunAsync calls
            for (int j = 0; j < currentBatchSize; j++)
            {
                int index = i + j;
                inputBufs[index] = get<vector<uint8_t>>(data[index]);
                reqIds.push_back(ie->RunAsync(inputBufs[index].data()));
            }

            // Wait for all requests in the current batch
            for (int j = 0; j < currentBatchSize; j++)
            {
                int index = i + j;
                auto outputs = ie->Wait(reqIds[j]);
                BMTResult result;
                int predictedIndex = (ie->outputs().front().type() == dxrt::DataType::FLOAT)
                                         ? getArgMax((float *)outputs.front()->data(), 1000)
                                         : *(uint16_t *)outputs.front()->data();
                result.Classification_ImageNet2012_PredictedIndex_0_to_999 = predictedIndex;
                batchResult.push_back(result);
            }
        }

        return batchResult;
    }
};
