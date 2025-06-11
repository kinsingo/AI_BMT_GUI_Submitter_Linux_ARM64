#include "ai_bmt_gui_caller.h"
#include "ai_bmt_interface.h"
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
using namespace cv;

class Classification_Implementation_MultiCore_Wait : public AI_BMT_Interface
{
    shared_ptr<dxrt::InferenceEngine> ie;
    int align_factor;
    int input_w = 224, input_h = 224, input_c = 3;

public:
    virtual Optional_Data getOptionalData() override
    {
        Optional_Data data;
        data.cpu_type = "Rockchip RK3588";
        data.accelerator_type = "M1(NPU) Async(Wait)";
        data.submitter = "DeepX";
        data.benchmark_model = "regnet_y_800mf_opset10.dxnn";
        return data;
    }

    virtual void Initialize(string modelPath) override
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
        int querySize = data.size();
        vector<BMTResult> queryResult(querySize);
        vector<int> reqIds(querySize);
        vector<vector<uint8_t>> inputBufs(querySize); // the inputBuf's memory must be maintained until the callback function or wait is called.
        const int maxConcurrentRequests = 3;

        for (int i = 0; i < querySize; i += maxConcurrentRequests)
        {
            int currentBatchSize = min(maxConcurrentRequests, querySize - i);

            // Start a batch of RunAsync calls
            for (int j = 0; j < currentBatchSize; j++)
            {
                int index = i + j;
                inputBufs[index] = get<vector<uint8_t>>(data[index]);
                reqIds[index] = (ie->RunAsync(inputBufs[index].data()));
            }

            // 병렬 처리로 Wait 실행
            vector<std::future<void>> futures;
            for (int j = 0; j < currentBatchSize; j++)
            {
                int index = i + j;
                futures.push_back(std::async(std::launch::async, [&, index]()
                                             {
                auto outputs = ie->Wait(reqIds[index]);
                inputBufs[index].clear(); // input buffer 해제
                BMTResult result;
                float *output_data = (float *)outputs.front()->data();
                vector<float> output(output_data, output_data + 1000);
                result.classProbabilities = output;
                queryResult[index] = result; }));
            }

            // 모든 요청 완료 대기
            for (auto &f : futures)
            {
                f.get();
            }
        }
        return queryResult;
    }
};
