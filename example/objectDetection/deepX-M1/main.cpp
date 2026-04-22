// Runtime Version v3.2.0

#include "ai_bmt_gui_caller.h"
#include "ai_bmt_interface.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "dxrt/dxrt_api.h"
#include "objectDetection_singleCore.cpp"
#include "objectDetection_multiCore_wait.cpp"
using namespace std;

int main(int argc, char *argv[])
{
    try
    {
        //shared_ptr<AI_BMT_Interface> interface = make_shared<ObjectDetection_Implementation_SingleCore>();
        shared_ptr<AI_BMT_Interface> interface = make_shared<ObjectDetection_Implementation_MultiThreads>();
        return AI_BMT_GUI_CALLER::call_BMT_GUI_For_Single_Task(argc, argv, interface);
    }
    catch (const exception &ex)
    {
        cout << ex.what() << endl;
    }
}
