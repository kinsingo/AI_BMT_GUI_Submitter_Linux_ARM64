cmake_minimum_required(VERSION 3.14)

project(AI_BMT_GUI_Submitter_App VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compile options
set(COMPILE_OPTIONS -Wall -std=gnu++2a -Werror -O3)

# Add the source files
set(PROJECT_SOURCES main.cpp)
file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
)

# Create the executable
add_executable(AI_BMT_GUI_Submitter ${SOURCES})
# Add the directory where the headers are located
target_include_directories(AI_BMT_GUI_Submitter PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link the Hailo library and OpenCV
find_package(HailoRT REQUIRED)
find_package(Threads)
find_package(OpenCV REQUIRED)

message(STATUS "Found OpenCV: " ${OpenCV_INCLUDE_DIRS})

# Ensure OpenCV and HailoRT are linked properly
target_include_directories(AI_BMT_GUI_Submitter PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${HailoRT_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

# Link the libraries to the executable
target_link_libraries(AI_BMT_GUI_Submitter PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/build/libAI_BMT_GUI_Library.so
    HailoRT::libhailort 
    Threads::Threads
    ${OpenCV_LIBS}
)

# Set RPATH to include the lib directory during the build and install phases
set_target_properties(AI_BMT_GUI_Submitter PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Ensure RPATH is always used (not stripped out)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
