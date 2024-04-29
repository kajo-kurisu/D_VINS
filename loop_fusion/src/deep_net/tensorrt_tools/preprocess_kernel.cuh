#ifndef PREPROCESS_KERNEL_CUH
#define PREPROCESS_KERNEL_CUH

#include "cuda_tools.hpp"

namespace CUDAKernel{

    enum class NormType : int{
        None      = 0,
        MeanStd   = 1,
        AlphaBeta = 2
    };

    enum class ChannelType : int{
        None          = 0,
        Invert        = 1
    };

    struct Norm{
        float mean[3];
        float std[3];
        float alpha, beta;
        NormType type = NormType::None;
        ChannelType channel_type = ChannelType::None;

        // out = (x * alpha - mean) / std
        static Norm mean_std(const float mean[3], const float std[3], float alpha = 1/255.0f, ChannelType channel_type=ChannelType::None);

        // out = x * alpha + beta
        static Norm alpha_beta(float alpha, float beta = 0, ChannelType channel_type=ChannelType::None);

        // None
        static Norm None();
    };


//    void warp_affine_bilinear_and_normalize_plane(
//            uint8_t* src, int src_line_size, int src_width, int src_height,
//            float* dst  , int dst_width, int dst_height,
//            float* matrix_2_3, uint8_t const_value, const Norm& norm,
//            cudaStream_t stream);

    void warp_affine_bilinear_and_normalize_plane(
            int channels, uint8_t* src, int src_line_size, int src_width, int src_height,
            float* dst  , int dst_width, int dst_height,
            float* matrix_2_3, uint8_t const_value, const Norm& norm,
            cudaStream_t stream);

    void warp_affine_bilinear_and_normalize_plane_mix(
            int channels, uint8_t* src, int src_line_size, int src_width, int src_height,
            float* dst  , int dst_width, int dst_height,
            float* matrix_2_3, uint8_t const_value, const Norm& norm,
            cudaStream_t stream);

    void normalize_kpts(
            int* src,  float* dst, int jobs,
            float shift_width,float shift_height,float scale,
            cudaStream_t stream
    );

    void normalize_kpts(
            float* src,  float* dst, int jobs,
            float shift_width,float shift_height,float scale,
            cudaStream_t stream
    );

    void matches_post_process(
            int *kpts0,
            int *kpts1,
            int *matches,
            float *mkpts0,
            float *mkpts1,
            float scale_width,
            float scale_height,
            int kpts_num,
            int matches_num,
            cudaStream_t stream);

    void matches_post_process(
            float *kpts0,
            float *kpts1,
            int *matches,
            float *mkpts0,
            float *mkpts1,
            float resize_w, float resize_h,
            float scale_width_0, float scale_height_0,
            float scale_width_1, float scale_height_1,
            int kpts_num,
            int matches_num,
            cudaStream_t stream);

    void kpts_post_process(
            int *kpts0,  int *kpts1,
            float *mkpts0,  float *mkpts1,
            float scale_width,  float scale_height,
            int kpts_num,
            cudaStream_t stream
    );

};

#endif // PREPROCESS_KERNEL_CUH