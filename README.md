> **Last Updated:** 2025-03-21
## Environment
 1. ISA(Instruction Set Architecture) : ARM64(aarch64)
 2. OS : Ubuntu 24.04 LTS
## Submitter User Guide Steps
Step1) Build System Set-up  
Step2) Interface Implementation  
Step3) Build and Start BMT

## Step 1) Build System Set-up (Installation Guide for Ubuntu)
**1. Install Packages**
- Open a terminal and run the following commands to install CMake, g++ compiler, Ninja Build System, and EGL Library.
  ```bash
  sudo apt update
  sudo apt install cmake                     # CMake
  sudo apt install build-essential           # GCC, G++, Make
  sudo apt-get install ninja-build           # Ninja
  sudo apt install libgl1 libgl1-mesa-dev    # EGL and OpenGL
  sudo apt install unzip                     # unzip
  ```

**2. Verify the Installation**

- You can check the versions of the installed tools by running the following commands. If these commands return version information for each tool, the installation was successful.
  ```bash
  cmake --version
  gcc --version
  ninja --version
  dpkg -l | grep -E 'libgl1|libgl1-mesa-dev'
  ```

## Step2) Interface Implementation
- Implement the overridden functions in the `Virtual_Submitter_Implementation` class, which inherits from the `SNU_BMT_Interface` interface, within `main.cpp`.
- Ensure that these functions operate correctly on the intended computing unit (e.g., CPU, GPU, NPU).
- To ensure that external headers (.h), source files (.cpp), or libraries required for implementing inference on the target hardware are included during the build process, make sure to update the **CMakeLists.txt** accordingly.
![SNU_BMT_Interface_Diagram_For_README](https://github.com/user-attachments/assets/a67c4ca7-2b40-451d-9d91-3202fdf2a673)

```cpp
#ifndef SNU_BMT_INTERFACE_H
#define SNU_BMT_INTERFACE_H
#include "label_type.h"
using namespace std;

// Represents the result of the inference process for a single batch.
// Fields:
// - Classification_ImageNet2012_PredictedIndex_0_to_999: An integer representing the predicted class index (0-999) for the ImageNet dataset.
struct EXPORT_SYMBOL BMTResult
{
    // While conducting Classification BMT, if the value is not between 0 and 999, it indicates that the result has not been updated and will be treated as an error.
    int Classification_ImageNet2012_PredictedIndex_0_to_999 = -1;

    // While conducting Object Detection BMT
    vector<Coco17DetectionResult> objectDetectionResult;
};

// Stores optional system configuration data provided by the Submitter.
// These details will be uploaded to the database along with the performance data.
struct EXPORT_SYMBOL Optional_Data
{
    string cpu_type; // e.g., Intel i7-9750HF
    string accelerator_type; // e.g., DeepX M1(NPU)
    string submitter; // e.g., DeepX
    string cpu_core_count; // e.g., 16
    string cpu_ram_capacity; // e.g., 32GB
    string cooling; // e.g., Air, Liquid, Passive
    string cooling_option; // e.g., Active, Passive (Active = with fan/pump, Passive = without fan)
    string cpu_accelerator_interconnect_interface; // e.g., PCIe Gen5 x16
    string benchmark_model; // e.g., ResNet-50
    string operating_system; // e.g., Ubuntu 20.04.5 LTS
};

// A variant can store and manage values only from a fixed set of types determined at compile time.
// Since variant manages types statically, it can be used with minimal runtime type-checking overhead.
// std::get<DataType>(variant) checks if the requested type matches the stored type and returns the value if they match.
using VariantType = variant<uint8_t*, uint16_t*, uint32_t*,
                            int8_t*,int16_t*,int32_t*,
                            float*, // Define variant pointer types
                            vector<uint8_t>, vector<uint16_t>, vector<uint32_t>,
                            vector<int8_t>, vector<int16_t>, vector<int32_t>,
                            vector<float>>; // Define variant vector types

class EXPORT_SYMBOL SNU_BMT_Interface
{
public:
   virtual ~SNU_BMT_Interface(){}

   // This is not mandatory but can be implemented if needed.
   // The virtual function getOptionalData() returns an Optional_Data object,
   // which includes fields like CPU_Type and Accelerator_Type.
   // By default, these fields are initialized as empty strings.
   virtual Optional_Data getOptionalData()
   {
       Optional_Data data;
       data.cpu_type = ""; // e.g., Intel i7-9750HF
       data.accelerator_type = ""; // e.g., DeepX M1(NPU)
       data.submitter = ""; // e.g., DeepX
       data.cpu_core_count = ""; // e.g., 16
       data.cpu_ram_capacity = ""; // e.g., 32GB
       data.cooling = ""; // e.g., Air, Liquid, Passive
       data.cooling_option = ""; // e.g., Active, Passive (Active = with fan/pump, Passive = without fan)
       data.cpu_accelerator_interconnect_interface = ""; // e.g., PCIe Gen5 x16
       data.benchmark_model = ""; // e.g., ResNet-50
       data.operating_system = ""; // e.g., Ubuntu 20.04.5 LTS
       return data;
   }

   // It is recommended to use this instead of a constructor,
   // as it allows handling additional errors that cannot be managed within the constructor.
   // The Initialize function is guaranteed to be called before convertToData and runInference are executed.
   virtual void Initialize() = 0;

   // Performs preprocessing before AI inference to convert data into the format required by the AI Processing Unit.
   // This method prepares model input data and is excluded from latency/throughput performance measurements.
   // The converted data is loaded into RAM prior to invoking the runInference(..) method.
   virtual VariantType convertToPreprocessedDataForInference(const string& imagePath) = 0;

   // Returns the final BMTResult value of the batch required for performance evaluation in the App.
   virtual vector<BMTResult> runInference(const vector<VariantType>& data) = 0;
};
#endif // SNU_BMT_INTERFACE_H
```


## Step3) Build and Start BMT

**1. Open the Linux terminal**

- Clone and navigate to the build directory using the following command
  ```bash
  git clone https://github.com/kinsingo/SNU_BMT_GUI_Submitter_Linux.git
  cd SNU_BMT_GUI_Submitter_Windows/build
  ```

**2. Generate the Ninja build system using cmake**

- Run the following command to remove existing cache
  ```bash
  rm -rf CMakeCache.txt CMakeFiles SNU_BMT_GUI_Submitter
  ```
- Run the following command to execute CMake in the current directory (usually the build directory). This command will generate the Ninja build system based on the CMakeLists.txt file located in the parent directory. Once successfully executed, the project will be ready to be built using Ninja.
  ```bash
  cmake -G "Ninja" ..
  ```

**3. Build the project**

- Run the following command to build the project using the build system configured by CMake in the current directory. This will compile the project and create the executable SNU_BMT_GUI_Submitter.exe in the build folder.
  ```bash
  cmake --build .
  ```

**4. Setting Library Path for Executable in Current Directory**

- Run the following command to make the executable(SNU_BMT_GUI_Submitter) can reference the libraries located in the lib folder of the current directory.
  ```bash
  export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
  ```

**5. Start Performance Analysis**

- Run the following command to start created excutable. When the GUI Popup, Click [Start BMT] button to start AI Performance Analysis.
  ```bash
  ./SNU_BMT_GUI_Submitter
  ```

**Run all commands at once (For Initial Build)**
```bash
git clone https://github.com/kinsingo/SNU_BMT_GUI_Submitter_Linux.git
cd SNU_BMT_GUI_Submitter_Linux/build/
sudo apt update
sudo apt install cmake
sudo apt install build-essential
sudo apt-get install ninja-build
sudo apt-get install libgl1 libgl1-mesa-dev
sudo apt install unzip
rm -rf CMakeCache.txt CMakeFiles SNU_BMT_GUI_Submitter
cmake -G "Ninja" ..
cmake --build .
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
./SNU_BMT_GUI_Submitter
```

**Run all commands at once (For Rebuild)**
```bash
rm -rf CMakeCache.txt CMakeFiles SNU_BMT_GUI_Submitter
cmake -G "Ninja" ..
cmake --build .
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
./SNU_BMT_GUI_Submitter
```

**Execute AI-BMT App**
```bash
./SNU_BMT_GUI_Submitter
```

