#include "ai_bmt_gui_caller.h"
#include "ai_bmt_interface.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "dxrt/dxrt_api.h"
#include "objectDetection_singleCore.cpp"
using namespace std;

int main(int argc, char *argv[])
{
    filesystem::path exePath = filesystem::absolute(argv[0]).parent_path(); // Get the current executable file path
    filesystem::path model_path = exePath / "Model" / "ObjectDetection" / "yolov5s_opset12.dxnn";
    string modelPath = model_path.string();
    try
    {
        shared_ptr<AI_BMT_Interface> interface = make_shared<ObjectDetection_Implementation_SingleCore>();
        AI_BMT_GUI_CALLER caller(interface, modelPath);
        return caller.call_BMT_GUI(argc, argv);
    }
    catch (const exception &ex)
    {
        cout << ex.what() << endl;
    }
}
