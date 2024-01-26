//
// Created by sy on 23-12-4.
//

#ifndef VINS_FUSION_TENSORRT_UTILS_H
#define VINS_FUSION_TENSORRT_UTILS_H

#pragma once
//tensorrt
#include "NvInfer.h"
#include "NvInferRuntime.h"

//cuda
#include "cuda.h"
#include <cuda_runtime.h>

//sys
#include "vector"
#include "fstream"
#include <memory>

//tensorrt_tools
#include "tensorrt_tools/trt_infer.hpp"
#include "tensorrt_tools/trt_tensor.hpp"
#include "tensorrt_tools/cuda_tools.hpp"
#include "tensorrt_tools/ilogger.hpp"

#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__ ,__LINE__)

using namespace std;

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line);


inline const char* severity_string(nvinfer1::ILogger::Severity t);


class TRTLogger : public nvinfer1::ILogger{
public:

    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg)
    noexcept override;
};

template<typename _T>
static shared_ptr<_T> make_nvshared(_T* ptr){

    //通过智能指针管理 nv 返回的指针参数，内存自动释放，避免泄漏

    return shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
}


vector<unsigned char> load_file(const string& file);
#endif //VINS_FUSION_TENSORRT_UTILS_H
