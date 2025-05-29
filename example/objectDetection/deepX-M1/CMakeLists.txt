cmake_minimum_required(VERSION 3.14)

project(AI_BMT_GUI_Submitter_App VERSION 0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PROJECT_SOURCES main.cpp)

# Create the executable
add_executable(AI_BMT_GUI_Submitter ${PROJECT_SOURCES})

#DX-RT
#Step 1: 
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake) # using function & cfg cmake file
include(dxapp.cfg)
include(dxapp.function)
include(toolchain.aarch64)

#Step 2: Classification
add_dxrt_lib()
add_opencv()

# Set RPATH to include the lib directory during the build and install phases
set_target_properties(AI_BMT_GUI_Submitter PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Ensure RPATH is always used (not stripped out)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Step 3
LIST(APPEND link_libs ${CMAKE_CURRENT_SOURCE_DIR}/build/libSNU_AI_BMT_GUI_Library.so)
add_target(AI_BMT_GUI_Submitter)


#Classification
set(classification 
objectDetection_singleCore.cpp)
add_executable(run_classifier ${classification})
add_target(run_classifier)