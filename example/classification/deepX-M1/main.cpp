#include "snu_bmt_gui_caller.h"
#include "snu_bmt_interface.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "dxrt/dxrt_api.h"
#include "classification_multiCore_callBack.cpp"
#include "classification_multiCore_wait.cpp"
#include "classification_singleCore.cpp"
using namespace std;

int main(int argc, char *argv[])
{
    filesystem::path exePath = filesystem::absolute(argv[0]).parent_path(); // Get the current executable file path
    filesystem::path model_path = exePath / "Model" / "Classification" / "resnet50_v2_opset10_dynamicBatch.dxnn";
    string modelPath = model_path.string();
    try
    {
        shared_ptr<SNU_BMT_Interface> interface = make_shared<Classification_Implementation_SingleCore>();
        // shared_ptr<SNU_BMT_Interface> interface = make_shared<Classification_Implementation_MultiCore_Wait>();
        // shared_ptr<SNU_BMT_Interface> interface = make_shared<Classification_Implementation_MultiCore_CallBack>();
        SNU_BMT_GUI_CALLER caller(interface, modelPath);
        return caller.call_BMT_GUI(argc, argv);
    }
    catch (const exception &ex)
    {
        cout << ex.what() << endl;
    }
}
