#include "hailo/hailort.hpp"
#include "snu_bmt_interface.h"
#include "snu_bmt_gui_caller.h"
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

constexpr int WIDTH = 224;
constexpr int HEIGHT = 224;

using BMTDataType = vector<float>;
/////////// Constants ///////////
constexpr size_t MAX_QUEUE_SIZE = 9960;
/////////////////////////////////

int argmax(const std::vector<float>& vec) {       
    return static_cast<int>(std::distance(vec.begin(), std::max_element(vec.begin(), vec.end())));
}

hailo_status run_post_process(std::shared_ptr<BoundedTSQueue<InferenceOutputItem>> results_queue,vector<BMTResult>& batchResult, size_t bs) {

    size_t i =0;
    while (true) {
        InferenceOutputItem output_item;  
        if (!results_queue->pop(output_item)) {

            break;
        }
        
        auto frame_idx = output_item.frame_idx;
        //std::cout<<output_item.frame_idx<<std::endl;
        size_t num_elements = 1000;
        std::vector<float> float_data(num_elements);
        std::memcpy(float_data.data(), output_item.output_data_and_infos[0].first, float_data.size()*sizeof(float));
        BMTResult result;

        result.classProbabilities=float_data;
        batchResult[frame_idx] = result;
        i++;
        if (i==bs){
            results_queue->stop();
        }
    }
    
    return HAILO_SUCCESS;
}

hailo_status run_preprocess(std::shared_ptr<BoundedTSQueue<PreprocessedFrameItem>> preprocessed_queue, const vector<VariantType>& data, size_t start, size_t end) {
    for (int i = start; i < end; i ++){
        vector<uint8_t> inputBuf = get<vector<uint8_t>>(data[i]);
        auto preprocessed_frame_item = create_preprocessed_frame_item(inputBuf , WIDTH, HEIGHT, i);

        preprocessed_queue->push(preprocessed_frame_item);
    }
    preprocessed_queue->stop();
    return HAILO_SUCCESS;
}

hailo_status run_inference_async(std::shared_ptr<BoundedTSQueue<PreprocessedFrameItem>> preprocessed_queue,AsyncModelInfer& model) {
    while (true) {
        PreprocessedFrameItem item;
        if (!preprocessed_queue->pop(item)) {     
            break;
        }

        model.infer(std::make_shared<vector<uint8_t>>(item.resized_for_infer), item.frame_idx);
    }
  
   
    return HAILO_SUCCESS;
}

class Virtual_Submitter_Implementation : public SNU_BMT_Interface
{
    //string modelPath;
    std::shared_ptr<BoundedTSQueue<PreprocessedFrameItem>> preprocessed_queue =
    std::make_shared<BoundedTSQueue<PreprocessedFrameItem>>(MAX_QUEUE_SIZE);

    std::shared_ptr<BoundedTSQueue<InferenceOutputItem>>   results_queue =
    std::make_shared<BoundedTSQueue<InferenceOutputItem>>(MAX_QUEUE_SIZE);
    AsyncModelInfer model;
    
public:
    Virtual_Submitter_Implementation()
    {
        model.crt();
    }

    virtual Optional_Data getOptionalData() override
    {
        Optional_Data data;
        data.cpu_type = "Broadcom BCM2712 quad-core Arm Cortex A76 processor @ 2.4GHz"; // e.g., Intel i7-9750HF
        data.accelerator_type = "Hailo-8"; // e.g., DeepX M1(NPU)
        data.submitter = "Hailo"; // e.g., DeepX
        data.cpu_core_count = "4"; // e.g., 16
        data.cpu_ram_capacity = "8GB"; // e.g., 32GB
        data.cooling = "Air"; // e.g., Air, Liquid, Passive
        data.cooling_option = "Active"; // e.g., Active, Passive (Active = with fan/pump, Passive = without fan)
        data.cpu_accelerator_interconnect_interface = "PCIe 3.0 4-lane"; // e.g., PCIe Gen5 x16
        data.benchmark_model = "mobilenet_v2_opset10"; // e.g., ResNet-50
        data.operating_system = "Ubuntu 24.04.2 LTS"; // e.g., Ubuntu 20.04.5 LTS
        return data;
    }

    virtual void Initialize(string modelPath) override
    {
            
        model.PathAndResult(modelPath);
        model.configure(results_queue);
    }

    virtual VariantType convertToPreprocessedDataForInference(const string& imagePath) override
    {
        cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (img.empty()) {
            throw std::runtime_error("Image not found or invalid.");
        }

        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        vector<uint8_t> inputBuf(HEIGHT*WIDTH*3);
        std::memcpy(inputBuf.data(), img.data, HEIGHT * WIDTH * 3);

        return inputBuf;
    }   
    virtual vector<BMTResult> runInference(const vector<VariantType>& data) override
    {           
        size_t frame_count = data.size();
        vector<BMTResult> batchResult(frame_count);
        for (size_t i = 0; i < frame_count; i += MAX_QUEUE_SIZE){
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
                    std::ref(model)
                    );
            auto output_parser_thread = std::async(run_post_process,
                results_queue,
                std::ref(batchResult),
                currentBatchSize);          
            hailo_status status = wait_and_check_threads(
                    preprocess_thread,    "Preprocess",
                    inference_thread,     "Inference",
                    output_parser_thread, "Postprocess "
                );

            if (status != HAILO_SUCCESS) {
                throw std::runtime_error("Inference failed");
            }
            preprocessed_queue->reset();
            results_queue->reset();      
        }
        return batchResult;     
    }
};

int main(int argc, char *argv[])
{
    filesystem::path exePath = filesystem::absolute(argv[0]).parent_path(); // Get the current executable file path
    filesystem::path model_path = "/home/hailoubuntu/BMT250507/SNU_BMT_GUI_Submitter_Linux_ARM64/mobilenet_v2_opset10.hef";
    string modelPath = model_path.string();
    try
    {
        shared_ptr<SNU_BMT_Interface> interface = make_shared<Virtual_Submitter_Implementation>();
        SNU_BMT_GUI_CALLER caller(interface, modelPath);
        return caller.call_BMT_GUI(argc, argv);
    }
    catch (const exception &ex)
    {
        cout << ex.what() << endl;
    }
}
