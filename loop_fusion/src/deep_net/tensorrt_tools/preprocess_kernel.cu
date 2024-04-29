
#include <iostream>
#include "preprocess_kernel.cuh"


namespace CUDAKernel{

    Norm Norm::mean_std(const float mean[3], const float std[3], float alpha, ChannelType channel_type){

        Norm out;
        out.type  = NormType::MeanStd;
        out.alpha = alpha;
        out.channel_type = channel_type;
        memcpy(out.mean, mean, sizeof(out.mean));
        memcpy(out.std,  std,  sizeof(out.std));
        return out;
    }

    Norm Norm::alpha_beta(float alpha, float beta, ChannelType channel_type){

        Norm out;
        out.type = NormType::AlphaBeta;
        out.alpha = alpha;
        out.beta = beta;
        out.channel_type = channel_type;
        return out;
    }

    Norm Norm::None(){
        return Norm();
    }


    /// ---------------------- 核函数定义 ---------------------


    __global__ void normalize_kpts_kernal(int* src ,float* dst, float shift_width,float shift_height,float scale,int edge)
    {

        int position = blockIdx.x * blockDim.x + threadIdx.x;
        if (position >= edge) return;

        if( position % 2 == 0)
        {
            *(dst+position) = (*(src+position) - shift_width) / scale;
        } else
        {
            *(dst+position) = (*(src+position) - shift_height) / scale;
        }
    };

    __global__ void normalize_kpts_kernal(float* src ,float* dst, float shift_width,float shift_height,float scale,int edge)
    {

        int position = blockIdx.x * blockDim.x + threadIdx.x;
        if (position >= edge) return;

        if( position % 2 == 0)
        {
            *(dst+position) = (*(src+position) - shift_width) / scale;
        } else
        {
            *(dst+position) = (*(src+position) - shift_height) / scale;
        }
    };


    __global__ void kpts_post_process_kernal(
            float *kpts0,
            float *kpts1,
            float *mkpts0,
            float *mkpts1,
            float scale_width_0,
            float scale_height_0,
            float scale_width_1,
            float scale_height_1,
            int edge)
    {
        int position = blockIdx.x * blockDim.x + threadIdx.x;
        if (position >= edge) return;

        if (position % 2 ==0)
        {
//            printf("%f\n",*(mkpts0 + position)/scale_width -0.5);
//            printf("mkpts0 %d %f %f\n",position ,*(mkpts0 + position),(*(mkpts0 + position) + 0.5) / scale_width_0  -0.5);
//            printf("mkpts1 %d %f %f\n",position ,*(mkpts1 + position),(*(mkpts1 + position) + 0.5) / scale_width_1  -0.5);

            *(mkpts0 + position) = (*(mkpts0 + position) + 0.5f) / scale_width_0  - 0.5f;
            *(mkpts1 + position) = (*(mkpts1 + position) + 0.5f) / scale_width_1  - 0.5f;

        } else{
//            printf("%f\n",(*(mkpts0 + position))/scale_height -0.5);
//            printf("mkpts0 %d %f %f\n",position ,*(mkpts0 + position),(*(mkpts0 + position) + 0.5) / scale_height_0  -0.5);
//            printf("mkpts1 %d %f %f\n",position ,*(mkpts1 + position),(*(mkpts1 + position) + 0.5) / scale_height_1  -0.5);

            *(mkpts0 + position) = (*(mkpts0 + position) + 0.5f) / scale_height_0  - 0.5f;
            *(mkpts1 + position) = (*(mkpts1 + position) + 0.5f) / scale_height_1  - 0.5f;
//            printf("%f",*(mkpts0 + position));
        }

    };


    __global__ void recover_normkpts_kernal(
            float *kpts0,
            float *kpts1,
            int *matches,
            float *mkpts0,
            float *mkpts1,
            float shift_width,
            float shift_height,
            int edge)
    {
        int position = blockIdx.x * blockDim.x + threadIdx.x;
        if (position >= edge) return;

        float scale = max(shift_height,shift_width);

        if (position % 2 ==0)
        {
            *(mkpts0 + position) = (*(kpts0 + *(matches + position)*2)) * scale + shift_width;
            *(mkpts0 + position+1) = (*(kpts0 + *(matches+position)*2+1))* scale + shift_height;
//
//            *(mkpts0 + position) = (*(kpts0 + *(matches + position)*2)) ;
//            *(mkpts0 + position+1) = (*(kpts0 + *(matches + position)*2+1));
//            printf(" mkpts0 %d  %f  %f\n",position, *(kpts0 + *(matches+position)*2),(*(kpts0 + *(matches+position)*2+1)));

        } else{
            *(mkpts1 + position-1) = (*(kpts1 + *(matches+position)*2))* scale +shift_width ;
            *(mkpts1 + position) = (*(kpts1 + *(matches+position)*2+1))* scale + shift_height ;

//            *(mkpts1 + position-1) = (*(kpts1 + *(matches+position)*2)) ;
//            *(mkpts1 + position) = (*(kpts1 + *(matches+position)*2+1));
//            printf("mkpts1 %d  %f  %f\n",position, *(kpts1 + *(matches+position)*2),(*(kpts1 + *(matches+position)*2+1)));
        }

    };



    __global__ void kpts_post_process_kernal(
            int *kpts0,
            int *kpts1,
            float *mkpts0,
            float *mkpts1,
            float scale_width,
            float scale_height,
            int edge)
    {
        int position = blockIdx.x * blockDim.x + threadIdx.x;
        if (position >= edge) return;

        if (position % 2 ==0)
        {
            printf("%d %f %f %f\n",position ,mkpts0,(*(mkpts0 + position) + 0.5) / scale_width  -0.5);
            *(mkpts0 + position) = (*(mkpts0 + position) + 0.5) / scale_width  -0.5;
            *(mkpts1 + position) = (*(mkpts1 + position) + 0.5) / scale_width  -0.5;
        } else{
            printf("%d %f %f %f\n",position ,mkpts1,(*(mkpts1 + position) + 0.5) / scale_width  -0.5 , (*(mkpts1 + position) + 0.5) / scale_height  -0.5);

            *(mkpts0 + position) = (*(mkpts0 + position) + 0.5) / scale_height  -0.5;
            *(mkpts1 + position) = (*(mkpts1 + position) + 0.5) / scale_height  -0.5;
        }

    };


    __global__ void matches_post_process_kernal(
            int *kpts0,
            int *kpts1,
            int *matches,
            float *mkpts0,
            float *mkpts1,
            float scale_width,
            float scale_height,
            int edge)
    {
        int position = blockIdx.x * blockDim.x + threadIdx.x;
        if (position >= edge) return;

        if (position % 2 ==0)
        {
            *(mkpts0 + position) = *(kpts0 + *(matches + position)*2);
            *(mkpts0 + position+1) = *(kpts0 + *(matches+position)*2+1);
        } else{
            *(mkpts1 + position) = *(kpts1 + *(matches+position)*2);
            *(mkpts1 + position+1) = *(kpts1 + *(matches+position)*2+1);
        }

    };


    __global__ void warp_affine_bilinear_and_normalize_plane_kernel(int channels,uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height,
                                                                    uint8_t const_value_st, float* warp_affine_matrix_2_3, Norm norm, int edge){

        if(channels ==3)
        {
            int position = blockIdx.x * blockDim.x + threadIdx.x;
            if (position >= edge) return;

            float m_x1 = warp_affine_matrix_2_3[0];
            float m_y1 = warp_affine_matrix_2_3[1];
            float m_z1 = warp_affine_matrix_2_3[2];
            float m_x2 = warp_affine_matrix_2_3[3];
            float m_y2 = warp_affine_matrix_2_3[4];
            float m_z2 = warp_affine_matrix_2_3[5];

            int dx      = position % dst_width;
            int dy      = position / dst_width;
            float src_x = m_x1 * dx + m_y1 * dy + m_z1;
            float src_y = m_x2 * dx + m_y2 * dy + m_z2;
            float c0, c1, c2;

            if(src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height){
                // out of range
                c0 = const_value_st;
                c1 = const_value_st;
                c2 = const_value_st;
            }else{
                int y_low = floorf(src_y);
                int x_low = floorf(src_x);
                int y_high = y_low + 1;
                int x_high = x_low + 1;

                uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
                float ly    = src_y - y_low;
                float lx    = src_x - x_low;
                float hy    = 1 - ly;
                float hx    = 1 - lx;
                float w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
                uint8_t* v1 = const_value;
                uint8_t* v2 = const_value;
                uint8_t* v3 = const_value;
                uint8_t* v4 = const_value;
                if(y_low >= 0){
                    if (x_low >= 0)
                        v1 = src + y_low * src_line_size + x_low * channels;

                    if (x_high < src_width)
                        v2 = src + y_low * src_line_size + x_high * channels;
                }

                if(y_high < src_height){
                    if (x_low >= 0)
                        v3 = src + y_high * src_line_size + x_low * channels;

                    if (x_high < src_width)
                        v4 = src + y_high * src_line_size + x_high * channels;
                }

                // same to opencv
                c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
                c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
                c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
            }

            if(norm.channel_type == ChannelType::Invert){
                float t = c2;
                c2 = c0;  c0 = t;
            }

            if(norm.type == NormType::MeanStd){
                c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
                c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];
                c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];
            }else if(norm.type == NormType::AlphaBeta){
                c0 = c0 * norm.alpha + norm.beta;
                c1 = c1 * norm.alpha + norm.beta;
                c2 = c2 * norm.alpha + norm.beta;
            }

            int area = dst_width * dst_height;
            float* pdst_c0 = dst + dy * dst_width + dx;
//            float* pdst_c1 = pdst_c0 + area;
//            float* pdst_c2 = pdst_c1 + area;
//
//            *pdst_c0 = c0;
//            *pdst_c1 = c1;
//            *pdst_c2 = c2;
            *pdst_c0 = 0.299 * c0 + 0.587 * c1 + 0.114 * c2;
        }
        else if(channels == 1 )
        {
            int position = blockIdx.x * blockDim.x + threadIdx.x;
            if (position >= edge) return;

            float m_x1 = warp_affine_matrix_2_3[0];
            float m_y1 = warp_affine_matrix_2_3[1];
            float m_z1 = warp_affine_matrix_2_3[2];
            float m_x2 = warp_affine_matrix_2_3[3];
            float m_y2 = warp_affine_matrix_2_3[4];
            float m_z2 = warp_affine_matrix_2_3[5];

            int dx      = position % dst_width;
            int dy      = position / dst_width;
            float src_x = m_x1 * dx + m_y1 * dy + m_z1;
            float src_y = m_x2 * dx + m_y2 * dy + m_z2;
            float c0;

            if(src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height){
                // out of range
                c0 = const_value_st;
            }else{
                int y_low = floorf(src_y);
                int x_low = floorf(src_x);
                int y_high = y_low + 1;
                int x_high = x_low + 1;

                uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
                float ly    = src_y - y_low;
                float lx    = src_x - x_low;
                float hy    = 1 - ly;
                float hx    = 1 - lx;
                float w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
                uint8_t* v1 = const_value;
                uint8_t* v2 = const_value;
                uint8_t* v3 = const_value;
                uint8_t* v4 = const_value;
                if(y_low >= 0){
                    if (x_low >= 0)
                        v1 = src + y_low * src_line_size + x_low * channels;

                    if (x_high < src_width)
                        v2 = src + y_low * src_line_size + x_high * channels;
                }

                if(y_high < src_height){
                    if (x_low >= 0)
                        v3 = src + y_high * src_line_size + x_low * channels;

                    if (x_high < src_width)
                        v4 = src + y_high * src_line_size + x_high * channels;
                }
                // same to opencv
                c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
            }
            if(norm.type == NormType::MeanStd){
                c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
            }else if(norm.type == NormType::AlphaBeta){
                c0 = c0 * norm.alpha + norm.beta;
            }

            float * pdst_c0 = dst + dy * dst_width + dx;
            *pdst_c0 = c0;
        }
    }

    __global__ void warp_affine_bilinear_and_normalize_plane_kernel_mix(int channels,uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height,
                                                                    uint8_t const_value_st, float* warp_affine_matrix_2_3, Norm norm, int edge){
        if(channels ==3)
        {
            int position = blockIdx.x * blockDim.x + threadIdx.x;
            if (position >= edge) return;

            float m_x1 = warp_affine_matrix_2_3[0];
            float m_y1 = warp_affine_matrix_2_3[1];
            float m_z1 = warp_affine_matrix_2_3[2];
            float m_x2 = warp_affine_matrix_2_3[3];
            float m_y2 = warp_affine_matrix_2_3[4];
            float m_z2 = warp_affine_matrix_2_3[5];

            int dx      = position % dst_width;
            int dy      = position / dst_width;
            float src_x = m_x1 * dx + m_y1 * dy + m_z1;
            float src_y = m_x2 * dx + m_y2 * dy + m_z2;
            float c0, c1, c2;

            if(src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height){
                // out of range
                c0 = const_value_st;
                c1 = const_value_st;
                c2 = const_value_st;
            }else{
                int y_low = floorf(src_y);
                int x_low = floorf(src_x);
                int y_high = y_low + 1;
                int x_high = x_low + 1;

                uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
                float ly    = src_y - y_low;
                float lx    = src_x - x_low;
                float hy    = 1 - ly;
                float hx    = 1 - lx;
                float w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
                uint8_t* v1 = const_value;
                uint8_t* v2 = const_value;
                uint8_t* v3 = const_value;
                uint8_t* v4 = const_value;
                if(y_low >= 0){
                    if (x_low >= 0)
                        v1 = src + y_low * src_line_size + x_low * channels;

                    if (x_high < src_width)
                        v2 = src + y_low * src_line_size + x_high * channels;
                }

                if(y_high < src_height){
                    if (x_low >= 0)
                        v3 = src + y_high * src_line_size + x_low * channels;

                    if (x_high < src_width)
                        v4 = src + y_high * src_line_size + x_high * channels;
                }

                // same to opencv
                c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
                c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
                c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
            }

            if(norm.channel_type == ChannelType::Invert){
                float t = c2;
                c2 = c0;  c0 = t;
            }

            if(norm.type == NormType::MeanStd){
                c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
                c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];
                c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];
            }else if(norm.type == NormType::AlphaBeta){
                c0 = c0 * norm.alpha + norm.beta;
                c1 = c1 * norm.alpha + norm.beta;
                c2 = c2 * norm.alpha + norm.beta;
            }

            int area = dst_width * dst_height;
            float* pdst_c0 = dst + dy * dst_width + dx;
            float* pdst_c1 = pdst_c0 + area;
            float* pdst_c2 = pdst_c1 + area;

            *pdst_c0 = c0;
            *pdst_c1 = c1;
            *pdst_c2 = c2;

        }
        else if(channels == 1 )
        {
            int position = blockIdx.x * blockDim.x + threadIdx.x;
            if (position >= edge) return;

            float m_x1 = warp_affine_matrix_2_3[0];
            float m_y1 = warp_affine_matrix_2_3[1];
            float m_z1 = warp_affine_matrix_2_3[2];
            float m_x2 = warp_affine_matrix_2_3[3];
            float m_y2 = warp_affine_matrix_2_3[4];
            float m_z2 = warp_affine_matrix_2_3[5];

            int dx      = position % dst_width;
            int dy      = position / dst_width;
            float src_x = m_x1 * dx + m_y1 * dy + m_z1;
            float src_y = m_x2 * dx + m_y2 * dy + m_z2;
            float c0;

            if(src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height){
                // out of range
                c0 = const_value_st;
            }else{
                int y_low = floorf(src_y);
                int x_low = floorf(src_x);
                int y_high = y_low + 1;
                int x_high = x_low + 1;

                uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
                float ly    = src_y - y_low;
                float lx    = src_x - x_low;
                float hy    = 1 - ly;
                float hx    = 1 - lx;
                float w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
                uint8_t* v1 = const_value;
                uint8_t* v2 = const_value;
                uint8_t* v3 = const_value;
                uint8_t* v4 = const_value;
                if(y_low >= 0){
                    if (x_low >= 0)
                        v1 = src + y_low * src_line_size + x_low * channels;

                    if (x_high < src_width)
                        v2 = src + y_low * src_line_size + x_high * channels;
                }

                if(y_high < src_height){
                    if (x_low >= 0)
                        v3 = src + y_high * src_line_size + x_low * channels;

                    if (x_high < src_width)
                        v4 = src + y_high * src_line_size + x_high * channels;
                }
                // same to opencv
                c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
            }
            if(norm.type == NormType::MeanStd){
                c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
            }else if(norm.type == NormType::AlphaBeta){
                c0 = c0 * norm.alpha + norm.beta;
            }

            float * pdst_c0 = dst + dy * dst_width + dx;
            *pdst_c0 = c0;
        }
    }


    /// ----------------------- 调用核函数 ----------------------------

    void warp_affine_bilinear_and_normalize_plane(
            int channels, uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height,
            float* matrix_2_3, uint8_t const_value, const Norm& norm,
            cudaStream_t stream) {

        int jobs   = dst_width * dst_height;
        auto grid  = CUDATools::grid_dims(jobs);
        auto block = CUDATools::block_dims(jobs);

        checkCudaKernel(warp_affine_bilinear_and_normalize_plane_kernel <<<grid, block, 0, stream >>> (
                channels, src, src_line_size,
                src_width, src_height, dst,
                dst_width, dst_height, const_value, matrix_2_3, norm, jobs
        ));
    }

    void warp_affine_bilinear_and_normalize_plane_mix(
            int channels, uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height,
            float* matrix_2_3, uint8_t const_value, const Norm& norm,
            cudaStream_t stream) {

        int jobs   = dst_width * dst_height;
        auto grid  = CUDATools::grid_dims(jobs);
        auto block = CUDATools::block_dims(jobs);

        checkCudaKernel(warp_affine_bilinear_and_normalize_plane_kernel_mix <<<grid, block, 0, stream >>> (
                channels, src, src_line_size,
                src_width, src_height, dst,
                dst_width, dst_height, const_value, matrix_2_3, norm, jobs
        ));
    }


    void normalize_kpts(
            int* src,  float* dst, int jobs,
            float shift_width,float shift_height,float scale,
            cudaStream_t stream
    )
    {
        auto grid  = CUDATools::grid_dims(jobs);
        auto block = CUDATools::block_dims(jobs);

        checkCudaKernel(normalize_kpts_kernal <<<grid, block, 0, stream >>> (
                src,
                dst,
                shift_width,
                shift_height,
                scale,
                jobs

        ));
    }


    void normalize_kpts(
            float* src,  float* dst, int jobs,
            float shift_width,float shift_height,float scale,
            cudaStream_t stream
    )
    {
        auto grid  = CUDATools::grid_dims(jobs);
        auto block = CUDATools::block_dims(jobs);

        checkCudaKernel(normalize_kpts_kernal <<<grid, block, 0, stream >>> (
                src,
                dst,
                shift_width,
                shift_height,
                scale,
                jobs

        ));
    }


    void kpts_post_process(
            int *kpts0,  int *kpts1,
            float *mkpts0,  float *mkpts1,
            float scale_width,  float scale_height,
            int kpts_num,
            cudaStream_t stream
    )
    {
        auto grid_kpts  = CUDATools::grid_dims(kpts_num);
        auto block_kpts = CUDATools::block_dims(kpts_num);

        checkCudaKernel(kpts_post_process_kernal <<<grid_kpts, block_kpts, 0, stream >>> (
                kpts0,kpts1,mkpts0,mkpts1,
                scale_width,
                scale_height,
                kpts_num
        ));

    }

    void matches_post_process(
            int *kpts0,  int *kpts1,   int *matches,
            float *mkpts0,  float *mkpts1,
            float scale_width,   float scale_height,
            int kpts_num,   int matches_num,
            cudaStream_t stream)
    {
        auto grid_kpts  = CUDATools::grid_dims(kpts_num);
        auto block_kpts = CUDATools::block_dims(kpts_num);

        auto grid_match  = CUDATools::grid_dims(matches_num);
        auto block_match = CUDATools::block_dims(matches_num);

        checkCudaKernel(matches_post_process_kernal<<<grid_match,block_match,0,stream>>>(
                kpts0,kpts1,matches,mkpts0,mkpts1,
                scale_width,scale_height,
                matches_num
        ))
        checkCudaRuntime(cudaStreamSynchronize(stream));
        checkCudaKernel(kpts_post_process_kernal <<<grid_kpts, block_kpts, 0, stream >>> (
                kpts0,kpts1,mkpts0,mkpts1,
                scale_width,
                scale_height,
                kpts_num

        ));
    }


    void matches_post_process(
            float *kpts0,  float *kpts1,   int *matches,
            float *mkpts0,  float *mkpts1,
            float resize_w,   float resize_h,
            float scale_width_0,   float scale_height_0,
            float scale_width_1,   float scale_height_1,
            int kpts_num,   int matches_num,
            cudaStream_t stream)
    {

//        printf("matches num = %d \n",matches_num);
        if(matches_num > 1)
        {
//            auto grid_kpts  = CUDATools::grid_dims(kpts_num);
//            auto block_kpts = CUDATools::block_dims(kpts_num);

            auto grid_match  = CUDATools::grid_dims(matches_num);
            auto block_match = CUDATools::block_dims(matches_num);

            //恢复特征点坐标
            checkCudaKernel(recover_normkpts_kernal<<<grid_match,block_match,0,stream>>>(
                    kpts0,kpts1,matches,mkpts0,mkpts1,
                    resize_w,resize_h,
                    matches_num
            ))
            checkCudaRuntime(cudaStreamSynchronize(stream));

            //建立匹配关系
            checkCudaKernel(kpts_post_process_kernal <<<grid_match, block_match, 0, stream >>> (
                    kpts0,kpts1,mkpts0,mkpts1,
                    scale_width_0,
                    scale_height_0,
                    scale_width_1,
                    scale_height_1,
                    kpts_num

            ));
        }
    }

};

