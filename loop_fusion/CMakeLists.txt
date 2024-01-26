cmake_minimum_required(VERSION 3.0)
project(loop_fusion  LANGUAGES C CXX CUDA)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    nav_msgs
    camera_models
    cv_bridge
    roslib
    )


SET(OpenCV_DIR "/home/sy/my_lib/opencv_3.4.10_cuda11.7")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_DIR}/include)
link_directories(${OpenCV_DIR}/build/lib)

#tensorrt
set(tensorrt_dir "/home/sy/my_lib/TensorRT-8.6.1.6_cu11")

include_directories(${tensorrt_dir}/include)
link_directories(${tensorrt_dir}/lib)


#cuda
set(CUDA_DIR  "/usr/local/cuda")
find_package(CUDA REQUIRED)
include_directories(${CUDA_DIR}/include)
link_directories(${CUDA_DIR}/lib64)


find_package(Ceres REQUIRED)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)

include_directories(        /usr/local/include/
        ./include
        ${catkin_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${tensorrt_dir}/include)

catkin_package()



add_executable(loop_fusion_node
    src/pose_graph_node.cpp
    src/pose_graph.cpp
    src/keyframe.cpp
    src/utility/CameraPoseVisualization.cpp
    src/ThirdParty/DBoW/BowVector.cpp
    src/ThirdParty/DBoW/FBrief.cpp
    src/ThirdParty/DBoW/FeatureVector.cpp
    src/ThirdParty/DBoW/QueryResults.cpp
    src/ThirdParty/DBoW/ScoringObject.cpp
    src/ThirdParty/DUtils/Random.cpp
    src/ThirdParty/DUtils/Timestamp.cpp
    src/ThirdParty/DVision/BRIEF.cpp
    src/ThirdParty/VocabularyBinary.cpp

    src/deep_net/deep_net.h
    src/deep_net/deep_net.cpp
    src/deep_net/tensorrt_utils.h
        src/deep_net/tensorrt_tools/ilogger.cpp
        src/deep_net/tensorrt_tools/cuda_tools.cpp
        src/deep_net/tensorrt_tools/preprocess_kernel.cu
        src/deep_net/tensorrt_tools/trt_infer.cpp
        src/deep_net/tensorrt_tools/trt_tensor.cpp
        src/deep_net/tensorrt_utils.cpp
)

target_link_libraries(loop_fusion_node  nvinfer cudart faiss dw  ${catkin_LIBRARIES}  ${OpenCV_LIBS} ${CERES_LIBRARIES})

# reference：https://developer.nvidia.com/cuda-gpus#compute
set_property(TARGET loop_fusion_node PROPERTY CUDA_ARCHITECTURES 75 )

#set(CUDA_ARCHITECTURES 75)
target_compile_options(loop_fusion_node PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread -lineinfo --use_fast_math --disable-warnings>)
