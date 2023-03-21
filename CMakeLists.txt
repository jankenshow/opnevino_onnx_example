cmake_minimum_required(VERSION 3.10)

project(openvino_inference)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV and OpenVINO packages
find_package(OpenCV REQUIRED)
find_package(InferenceEngine REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS} ${InferenceEngine_INCLUDE_DIRS})

# Add the executable
add_executable(${PROJECT_NAME} inference.cpp)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} ${InferenceEngine_LIBRARIES})
