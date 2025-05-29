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

class ConcurrentQueue
{
    queue<int> q;
    mutex m;
    condition_variable cv;

public:
    void push(int value)
    {
        lock_guard<std::mutex> lock(m);
        q.push(value);
        cv.notify_one();
    }
    int pop()
    {
        unique_lock<std::mutex> lock(m);
        cv.wait(lock, [this]
                { return !q.empty(); });
        int value = q.front();
        q.pop();
        return value;
    }
};

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

ConcurrentQueue gResultQueue;
vector<int> predictedIndexes;
mutex gCBMutex;
atomic<int> gCallbackCnt = 0;
int onInferenceCallbackFunc(vector<shared_ptr<dxrt::Tensor>> outputs, void *userArg)
{
    pair<int, int> *user_data = reinterpret_cast<pair<int, int> *>(userArg);
    int index = getArgMax((float *)outputs.front()->data(), 1000); // ResNet50
    // auto index = *(uint16_t *)outputs.front()->data();//EfficientNet
    predictedIndexes[user_data->first] = index;
    {
        lock_guard<mutex> lock(gCBMutex);
        gCallbackCnt++;
        if (user_data->second == gCallbackCnt)
        {
            gResultQueue.push(gCallbackCnt);
        }
    }
    delete user_data;
    return 0;
}

class Classification_Implementation_MultiCore_CallBack : public AI_BMT_Interface
{
    shared_ptr<dxrt::InferenceEngine> ie;
    int align_factor;
    int input_w = 224, input_h = 224, input_c = 3;
    int batchSize = -1;

public:
    virtual Optional_Data getOptionalData() override
    {
        Optional_Data data;
        data.cpu_type = "Rockchip RK3588";
        data.accelerator_type = "M1(NPU) Async(Callback)";
        data.submitter = "DeepX";
        return data;
    }

    virtual void Initialize(string modelPath) override
    {
        cout << "Initialze() is called" << endl;
        align_factor = ((int)(input_w * input_c)) & (-64);
        align_factor = (input_w * input_c) - align_factor;

        ie = make_shared<dxrt::InferenceEngine>(modelPath);
        ie->RegisterCallBack(onInferenceCallbackFunc); // async(콜백 함수 등록)
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

    virtual vector<BMTResult> runInference(const vector<VariantType> &data) override
    {
        gCallbackCnt = 0;
        vector<BMTResult> batchResult;
        batchSize = data.size();

        // the inputBuf's memory must be maintained until the callback function or wait is called.
        int maxConcurrentRequests = 3;
        vector<vector<uint8_t>> inputBufs(batchSize);

        for (int i = 0; i < batchSize; i += maxConcurrentRequests)
        {
            gCallbackCnt = 0; // Reset callback counter for each batch
            int currentBatchSize = min(maxConcurrentRequests, batchSize - i);
            predictedIndexes.resize(currentBatchSize);

            // Submit up to 3 async requests
            for (int j = 0; j < currentBatchSize; j++)
            {
                int index = i + j;
                inputBufs[index] = get<vector<uint8_t>>(data[index]);
                pair<int, int> *userData = new pair<int, int>(j, currentBatchSize);
                ie->RunAsync(inputBufs[index].data(), userData);
            }

            // Wait until all 3 callbacks are completed
            gResultQueue.pop();

            // Store results
            for (int j = 0; j < currentBatchSize; j++)
            {
                BMTResult result;
                result.Classification_ImageNet_PredictedIndex_0_to_999 = predictedIndexes[j];
                batchResult.push_back(result);
            }
        }
        return batchResult;
    }
};
