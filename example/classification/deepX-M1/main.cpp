#include "snu_bmt_gui_caller.h"
#include "snu_bmt_interface.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "dxrt/dxrt_api.h"
using namespace std;

constexpr int input_w = 224, input_h = 224, input_c = 3;

class Virtual_Submitter_Implementation : public SNU_BMT_Interface
{
    string modelPath;
    shared_ptr<dxrt::InferenceEngine> ie;
    int align_factor;

public:
    Virtual_Submitter_Implementation(string modelPath)
    {
        this->modelPath = modelPath;
    }

    virtual Optional_Data getOptionalData() override
    {
        Optional_Data data;
        data.CPU_Type = "";
        data.Accelerator_Type = "DeepX M1";
        return data;
    }

    virtual void Initialize() override
    {
        cout << "Initialze() is called" << endl;
        ie = make_shared<dxrt::InferenceEngine>(modelPath);
        align_factor = ((int)(input_w * input_c)) & (-64);
        align_factor = (input_w * input_c) - align_factor;
    }

    virtual VariantType convertToPreprocessedDataForInference(const string &imagePath) override
    {
        cv::Mat image, resized, input;
        image = cv::imread(imagePath, cv::IMREAD_COLOR);
        cv::resize(image, resized, cv::Size(input_w, input_h));
        cv::cvtColor(resized, input, cv::COLOR_BGR2RGB);
        vector<uint8_t> inputBuf(input_h * (input_w * input_c + align_factor));
        for (int y = 0; y < input_h; y++)
        {
            memcpy(&inputBuf[y * (input_w * input_c + align_factor)], &input.data[y * input_w * input_c], input_w * input_c);
        }
        return inputBuf;
    }

    virtual vector<BMTResult> runInference(const vector<VariantType> &data) override
    {
        vector<BMTResult> batchResult;
        const int batchSize = data.size();
        for (int i = 0; i < batchSize; i++)
        {
            vector<uint8_t> inputBuf = get<vector<uint8_t>>(data[i]);
            auto outputs = ie->Run(inputBuf.data());
            auto index = *(uint16_t *)outputs.front()->data();
            BMTResult result;
            result.Classification_ImageNet2012_PredictedIndex_0_to_999 = index; // temporary value(0~999) is assigned here. It should be replaced with the actual predicted value.
            batchResult.push_back(result);
        }
        return batchResult;
    }
};

int main(int argc, char *argv[])
{
    filesystem::path exePath = filesystem::absolute(argv[0]).parent_path(); // Get the current executable file path
    filesystem::path model_path = exePath / "Model" / "Classification" / "EfficientNetB0_4.dxnn";
    string modelPath = model_path.string();
    try
    {
        shared_ptr<SNU_BMT_Interface> interface = make_shared<Virtual_Submitter_Implementation>(modelPath);
        SNU_BMT_GUI_CALLER caller(interface, modelPath);
        return caller.call_BMT_GUI(argc, argv);
    }
    catch (const exception &ex)
    {
        cout << ex.what() << endl;
    }
}
