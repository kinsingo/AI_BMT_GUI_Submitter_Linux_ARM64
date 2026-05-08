#include "ai_bmt_gui_caller.h"
#include "ai_bmt_interface.h"
#include <thread>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <filesystem>

using namespace std;
using namespace cv;
using namespace Ort;

class ImageClassification_Interface_Implementation : public AI_BMT_Interface
{
private:
    Env env;
    RunOptions runOptions;
    shared_ptr<Session> session;
    array<const char*, 1> inputNames;
    array<const char*, 1> outputNames;
    MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    static constexpr int     inputResolution = 224;
    static constexpr int64_t imageSize       = 3LL * inputResolution * inputResolution;

    // Pre-allocated buffers for single-stream optimization
    vector<float> singleInputBuffer;
    vector<float> singleOutputBuffer;

    // Pre-allocated buffers for batch processing
    // Jetson Orin unified memory: batch inference amortizes CUDA kernel launch overhead
    static constexpr int64_t maxBatchSize = 32;
    vector<float> batchInputBuffer;
    vector<float> batchOutputBuffer;
    vector<int> validIndices;

public:

    virtual InterfaceType getInterfaceType() override
    {
        return InterfaceType::ImageClassification;
    }

    virtual PowerDeviceType getPowerDeviceType() override
    {
        return PowerDeviceType::JetsonSoC;
    }

    virtual void initialize(string modelPath) override
    {
        // Try CUDAExecutionProvider first; fall back to CPUExecutionProvider on failure.
        auto createSession = [&](bool withCuda) -> shared_ptr<Session> {
            SessionOptions opts;
            opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
            opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            if (withCuda) {
                OrtCUDAProviderOptions cudaOptions{};
                cudaOptions.device_id                 = 0;   // Jetson has single iGPU
                cudaOptions.gpu_mem_limit             = 0;   // 0 = no limit
                cudaOptions.arena_extend_strategy     = 0;   // kNextPowerOfTwo
                cudaOptions.cudnn_conv_algo_search    = OrtCudnnConvAlgoSearchHeuristic; // faster init than Exhaustive on Jetson
                cudaOptions.do_copy_in_default_stream = 1;
                opts.AppendExecutionProvider_CUDA(cudaOptions);
            }
            return make_shared<Session>(env, modelPath.c_str(), opts);
        };

        try {
            session = createSession(true);
            cout << "[INFO] CUDAExecutionProvider 활성화" << endl;
        } catch (const Ort::Exception& e) {
            cerr << "[WARN] CUDAExecutionProvider 사용 불가, CPU로 fallback: " << e.what() << endl;
            session = createSession(false);
        }

        // Get input and output names
        AllocatorWithDefaultOptions allocator;
        AllocatedStringPtr inputName  = session->GetInputNameAllocated(0, allocator);
        AllocatedStringPtr outputName = session->GetOutputNameAllocated(0, allocator);
        inputNames  = { inputName.get() };
        outputNames = { outputName.get() };
        inputName.release();
        outputName.release();

        // Pre-allocate buffers (avoids per-inference heap allocation)
        singleInputBuffer.resize(imageSize);
        singleOutputBuffer.resize(1000);
        batchInputBuffer.resize(maxBatchSize * imageSize);
        batchOutputBuffer.resize(maxBatchSize * 1000);
        validIndices.reserve(maxBatchSize);
    }

    virtual Optional_Data getOptionalData() override
    {
        Optional_Data data;
        data.cpu_type = "NVIDIA Jetson Orin"; 
        data.accelerator_type = ""; // e.g., Ampere iGPU
        data.submitter = "CUDA 12.6 + cuDNN 9.3.0 + ONNX Runtime 1.20.0 + maxBatchSize (32)";
        data.cpu_core_count = "";
        data.cpu_ram_capacity = "";
        data.cooling = "";
        data.cooling_option = "";
        data.cpu_accelerator_interconnect_interface = "";
        data.benchmark_model = "";
        data.operating_system = "Ubuntu 22.04 LTS (JetPack 6.x)";
        return data;
    }

    virtual VariantType preprocessVisionData(const string& imagePath) override
    {
        Mat image = imread(imagePath);
        if (image.empty()) {
            throw runtime_error("Failed to load image: " + imagePath);
        }

        cvtColor(image, image, cv::COLOR_BGR2RGB);
        image = image.reshape(1, 1);

        vector<float> vec;
        image.convertTo(vec, CV_32FC1, 1. / 255);

        const vector<float> means = { 0.485, 0.456, 0.406 };
        const vector<float> stds  = { 0.229, 0.224, 0.225 };

        // Transpose (H,W,C) -> (C,H,W)
        vector<float> output;
        output.reserve(imageSize);
        for (size_t ch = 0; ch < 3; ++ch) {
            for (size_t i = ch; i < vec.size(); i += 3) {
                output.emplace_back((vec[i] - means[ch]) / stds[ch]);
            }
        }
        return output;
    }

    virtual vector<BMTVisionResult> inferVision(const vector<VariantType>& data) override
    {
        const int querySize = data.size();
        vector<BMTVisionResult> results;
        results.reserve(querySize);

        // Fast path for single query (most common in single-stream BMT scenario)
        if (querySize == 1) {
            const vector<float>& imageVec = get<vector<float>>(data[0]);
            copy(imageVec.begin(), imageVec.end(), singleInputBuffer.begin());

            const array<int64_t, 4> inputShape  = { 1, 3, inputResolution, inputResolution };
            const array<int64_t, 2> outputShape = { 1, 1000 };
            auto inputTensor  = Ort::Value::CreateTensor<float>(memory_info, singleInputBuffer.data(),  imageSize, inputShape.data(),  inputShape.size());
            auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, singleOutputBuffer.data(), 1000,      outputShape.data(), outputShape.size());

            session->Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);

            BMTVisionResult result;
            result.classProbabilities = singleOutputBuffer;
            results.push_back(result);
            return results;
        }

        // Batch path: pack up to maxBatchSize samples per GPU inference call
        for (int startIdx = 0; startIdx < querySize; startIdx += maxBatchSize) {
            const int64_t currentBatchSize = min((int64_t)(querySize - startIdx), maxBatchSize);
            validIndices.clear();

            for (int i = 0; i < currentBatchSize; ++i) {
                const vector<float>& imageVec = get<vector<float>>(data[startIdx + i]);
                copy(imageVec.begin(), imageVec.end(),
                     batchInputBuffer.begin() + (int64_t)validIndices.size() * imageSize);
                validIndices.push_back(startIdx + i);
            }

            if (validIndices.empty()) continue;

            const int64_t validCount = (int64_t)validIndices.size();
            const array<int64_t, 4> inputShape  = { validCount, 3, inputResolution, inputResolution };
            const array<int64_t, 2> outputShape = { validCount, 1000 };
            auto inputTensor  = Ort::Value::CreateTensor<float>(memory_info, batchInputBuffer.data(),  validCount * imageSize, inputShape.data(),  inputShape.size());
            auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, batchOutputBuffer.data(), validCount * 1000,      outputShape.data(), outputShape.size());

            session->Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);

            for (int64_t i = 0; i < validCount; ++i) {
                BMTVisionResult result;
                result.classProbabilities.assign(
                    batchOutputBuffer.begin() + i * 1000,
                    batchOutputBuffer.begin() + (i + 1) * 1000
                );
                results.push_back(move(result));
            }
        }

        return results;
    }
};


int main(int argc, char *argv[])
{
    try
    {
        shared_ptr<AI_BMT_Interface> interface = make_shared<ImageClassification_Interface_Implementation>();
        return AI_BMT_GUI_CALLER::call_BMT_GUI_For_Single_Task(argc, argv, interface);
    }
    catch (const exception &ex)
    {
        cout << ex.what() << endl;
    }
}