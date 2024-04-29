//
// Created by sy on 23-11-23.
//

#include "tensorrt_utils.h"


using namespace std;

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){

    if(code != cudaSuccess){
        const char* err_name = cudaGetErrorName(code);
        const char* err_message = cudaGetErrorString(code);

        printf("runtime error %s:%d %s failed. \n code = %s, message =%s\n", file, line, op, err_name, err_message);

        return false;
    }
    return true;

}

inline const char* severity_string(nvinfer1::ILogger::Severity t){

    switch(t){

        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR: return "error";
        case nvinfer1::ILogger::Severity::kWARNING: return "warning";
        case nvinfer1::ILogger::Severity::kINFO: return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";

        default: return "unknow";

    }
}

void TRTLogger::log(nvinfer1::ILogger::Severity severity, const nvinfer1::AsciiChar *msg) noexcept {

    if(severity <= Severity::kINFO){

        if(severity == Severity::kWARNING){

            printf("\033[33m%s: %s\033[0m\n", severity_string(severity),msg);
        }
        else if(severity <= Severity::kERROR){

            printf("\033[31m%s: %s\033[0m\n", severity_string(severity),msg);
        }
        else{

            printf("%s: %s\n", severity_string(severity), msg);

        }

    }
}


vector<unsigned char> load_file(const string& file){
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;

    if (length > 0){
        in.seekg(0, ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;

}

