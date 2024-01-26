//
// Created by sy on 23-12-4.
//
//
// Created by sy on 23-10-7.
//

#include "deep_net.h"

//opencv
#include "opencv2/core.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/highgui.hpp"
#include "cv_bridge/cv_bridge.h"


//faiss
#include "faiss/IndexFlat.h"
#include "faiss/IndexHNSW.h"

// 64-bit int
using idx_t = faiss::Index::idx_t;
using namespace std;


MixVPR::MixVPR( const std::string engine_path) {
//    TRTLogger logger;
    // 创建两个事件，一个开始事件，一个结束事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 在默认的stream记录开始事件
    cudaEventRecord(start);

    //-------------------------- 1.加载模型 ----------------------------
    static auto engine_data = load_file(engine_path);

    //------------------------ 2.创建 runtime --------------------------
    static auto runtime = make_nvshared(nvinfer1::createInferRuntime(Logger));

    //-------------------------- 3.反序列化 ---------------------------
    static auto engine = make_nvshared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));

    if (engine == nullptr) {
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return ;
    }

    //-------------------- 4.从engine上创建 执行上下文
    static auto execution_context = make_nvshared(engine->createExecutionContext());


    //------------------------- 5.创建CUDA 流 ------------------------
    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));
}




void MixVPR::inference(const cv::String& filename,
                       std::vector<frame> &frame_set,
                       std::vector<float> &des_db,
                       const char *engine_path
) {

    float time;
    frame frame;

    TRTLogger logger;
    // 创建两个事件，一个开始事件，一个结束事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 在默认的stream记录开始事件
    cudaEventRecord(start);

    //-------------------------- 1.加载模型 ----------------------------
    static auto engine_data = load_file(engine_path);

    //------------------------ 2.创建 runtime --------------------------
    static auto runtime = make_nvshared(nvinfer1::createInferRuntime(logger));

    //-------------------------- 3.反序列化 ---------------------------
    static auto engine = make_nvshared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));

    if (engine == nullptr) {
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return ;
    }

    //-------------------- 4.从engine上创建 执行上下文
    static auto execution_context = make_nvshared(engine->createExecutionContext());


    //------------------------- 5.创建CUDA 流 ------------------------
    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));


    //-------------------------6.准备数据----------------------------
    //图像格式
    int input_batch= 1;
    int input_channel = 3;
    int input_height = 320;
    int input_width = 320;
    int input_numel = input_batch * input_channel * input_height * input_width;
    int image_num=1;

    float* input_data_host= nullptr;
    float* input_data_device = nullptr;
    checkRuntime(cudaMallocHost(&input_data_host, image_num*input_numel * sizeof(float)));

    checkRuntime(cudaMalloc(&input_data_device, image_num*input_numel * sizeof(float)));
    float mean[] = {0.406, 0.456, 0.485};
    float std[]= {0.225, 0.224, 0.229};

    bool test_in_dataset = true;
    bool test_in_callback = false;
    cv::Mat image;

    // image to float
    if(test_in_dataset)
    {
        image = cv::imread(filename);
    }

    cv::resize(image, image, cv::Size(input_width, input_height));
    int image_area = image.cols * image.rows;
    unsigned char* pimage = image.data; //BGRBGRBGR TO BBBGGGRRR   用地址偏移做索引

    //------------------使用多张图片做测试------------------------
    for(int i =0;i<image_num;i++)
    {
        float* phost_b = input_data_host + image_area *(3*i);
        float* phost_g = input_data_host + image_area *(3*i+1) ;
        float* phost_r = input_data_host + image_area *(3*i+2) ;
        pimage = image.data;  //执行完一张图片后，下一个循环重新指向图片的收个像素的指针

        for(int j = 0; j < image_area; ++j, pimage += 3){
            //注意这里的顺序 rgb 调换了

            *phost_r++ = (pimage[0] / 255.0f - mean[0]) / std[0];
            *phost_g++ = (pimage[1] / 255.0f - mean[1]) / std[1];
            *phost_b++ = (pimage[2] / 255.0f - mean[2]) / std[2];
        }

    }

    checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host,
                                 image_num*input_numel * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));

    const int des_dim = 512;
    float output_data_host[des_dim];
    float* output_data_device = nullptr;

    checkRuntime(cudaMalloc(&output_data_device,
                            sizeof(output_data_host)));

    //明确当前推理时，使用的数据输入大小
    auto input_dims = execution_context->getBindingDimensions(0);
    input_dims.d[0] = input_batch;
    execution_context->setBindingDimensions(0, input_dims);


    //用一个指针数组指定 input 和 output 在 GPU 内存中的指针
    float* bindings[] = {input_data_device, output_data_device};


    //------------------------ 7.推理 --------------------------
    bool success = execution_context->enqueueV2((void**)bindings, stream,
                                                nullptr);


    checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device,
                                 sizeof(output_data_host), cudaMemcpyDeviceToHost, stream));

    checkRuntime(cudaStreamSynchronize(stream));

    //解算描述子

//    for(int i =0;i<sizeof(output_data_host)/sizeof (float);i++)
//    {
//        frame.global_descriptor[i]=*(output_data_host+i);

//        frame.img_global_des_vec.push_back(*(output_data_host+i));
//        des_db.push_back(*(output_data_host+i));
//    }
//    cout<<"mix_des size = "<<frame.img_des_vec.size()<<endl;


//    std::vector<float> a (output_data_host,output_data_host+1);

    frame.img_global_des_vec.insert(frame.img_global_des_vec.begin(),output_data_host, output_data_host + 512);
    des_db.insert(des_db.end(),output_data_host,output_data_host+512);

//--------------------8.按照创建相反的顺序释放内存 ----------------------

    // 推理完成后，记录结束事件
    cudaEventRecord(stop);
    // 等待结束事件完成
    cudaEventSynchronize(stop);


    // 计算两个event之间的间隔时间
//    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("time = %f\n",time);

    frame.extract_time = time;
    frame_set.push_back(frame);


    // 销毁事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFreeHost(input_data_host));
    checkRuntime(cudaFree(input_data_device));
    checkRuntime(cudaFree(output_data_device));

}





void MixVPR::test_in_datasets(const std::string filepath,
                              std::vector<cv::String > &namearray,
                              std::vector<frame> &frame_set,
                              std::map<int,frame> &frame_des_dataset,
                              std::vector<float> &des_db
){

    cv::glob(filepath,namearray);

    std::sort(namearray.begin(), namearray.end());

    for(size_t i=0 ; i < namearray.size();i++) {
        auto image = cv::imread(namearray[i]);
//        cv::imshow("666", image);
        inference(namearray[i],frame_set,des_db,"/home/sy/sy/Mix_ws/src/mixvpr/model/mix/mix_512.engine");
//        cv::waitKey(0);
        frame_set.at(i).raw_image = image;
        frame_set.at(i).frame_id = std::to_string(i);
        frame_set.at(i).filename = namearray[i];
        frame_des_dataset[i] = frame_set[i];
    }

}

void MixVPR::img_callback(const sensor_msgs::CompressedImage &msg,
                          frame &_frame,
                          std::vector<frame> &frame_set,
                          std::map<int,frame> &frame_des_dataset,
                          std::vector<float> &des_db
)
{
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::TYPE_8UC3);
    _frame.raw_image =cv_ptr->image;
    _frame.frame_id =  msg.header.frame_id;
    _frame.timestamp = msg.header.stamp;
    cv::imshow("img",cv_ptr->image);
    cv::waitKey(0);

}

void MixVPR::sort_vec_faiss(vector<frame> &frame_set, std::vector<float> &des_db )
{
    int d = 512;      // dimension
    int nb = frame_set.size(); // database size
//    int nb = 3; // database size

    int nq = 1;  // nb of queries

    float* xb;
    float* xq;

    //构造数据集db
    xb = des_db.data();
    //选择xq

    int nlist=1;
//    faiss::IndexFlatL2 index(d); // call constructor
    faiss::IndexFlatIP index(d); // call constructor

//    faiss::IndexFlatL2 quantizer(d);
//    faiss::IndexIVFFlat index(&quantizer,d,10);
//
//    faiss::IndexIVFPQ index(&quantizer,d,6,4,8);

//    index.train(nb,xb);
//    index.add(nb,xb);


//    index.train(nb,xb);
    printf("is_trained = %s\n", index.is_trained ? "true" : "false");
    index.add(nb, xb); // add vectors to the index
    printf("ntotal = %zd\n", index.ntotal);
    int k = 3;


    for(auto & img : frame_set)
    {
        xq = img.img_global_des_vec.data();
        { // search xq
            idx_t* I = new idx_t[k * nq];
            float* D = new float[k * nq];

//        index.search(nq, xq, k, D, I);

//        index.search(nq,xq,k,D,I);

//        index.nprobe = 10;
            index.search(nq,xq,k,D,I);
            // print results
            //打印最相似的几个向量索引
            cout<<"----------- frame"<<img.frame_id<<"-----------"<<endl;
            printf("I (top k  first results)=\n");
            for (int i = 0; i < nq; i++) {
                for (int j = 0; j < k; j++)
                    printf("%5zd ", I[i * k + j]);
                printf("\n");
            }

            img.top_k_ind.push_back(I[1]);
            printf("D=\n");
            for (int i = 0; i < nq; i++) {
                for (int j = 0; j < k; j++)
                    printf("%7g ", D[i * k + j]);
                printf("\n");
            }
            delete[] I;
            delete[] D;
        }
    }
    //开始真正的向量检索，nq表示待检索的向量数量，xq表示待检索的向量，k表示top k ,D  和 I 表示返回的距离和索引Index


}


void MixVPR::run(std::string &datapath)
{
    //加载数据集

    std::vector<cv::String > namearray;
    std::vector<frame> frame_set;
    std::map<int,frame> frame_des_dataset;
    std::vector<float> des_db;


    auto startTime = std::chrono::high_resolution_clock::now();

    //数据集中进行测试
    test_in_datasets(datapath, namearray,frame_set,frame_des_dataset,des_db);

    auto endTime = std::chrono::high_resolution_clock::now();
    float cpu_time = std::chrono::duration<float, std::milli>(endTime - startTime).count();

    //--------------------打印数据-------------------
    int round = frame_set.size();

    float stream_time =0 ;
    for(int i =0;i<frame_set.size();i++ )
    {
        stream_time += frame_set.at(i).extract_time;
    }

    printf("time_total in stream = %f\n",(stream_time)/(round));
    printf("time_total_cpu = %f\n",cpu_time/(round));

    //faiss sort

    auto startTime2 = std::chrono::high_resolution_clock::now();
    sort_vec_faiss(frame_set,des_db);

    auto endTime2 = std::chrono::high_resolution_clock::now();
    float faiss_sort_time = std::chrono::duration<float, std::milli>(endTime2 - startTime2).count();
    cout<<"total time in faiss = "<<faiss_sort_time<<endl;



    for(int i =0;i<frame_set.size();i++)
    {
        auto img0 = cv::imread(namearray[frame_set[i].top_k_ind[0]]);
        auto img1 = cv::imread(namearray[i]);
        cv::Mat img2;
        cv::hconcat(img0,img1,img2);
        cv::imshow("666",img2);
        cv::waitKey(0);
    }
}

namespace Estimator_net {

    using namespace std;

    using RMEigen = Eigen::Matrix<float, 16, 16, Eigen::RowMajor>;

    Eigen::MatrixXf change(const Eigen::MatrixXf &r) {
        return r.cwiseMax(r.cwiseInverse());
    }

    Eigen::MatrixXf sz(const Eigen::MatrixXf &w, const Eigen::MatrixXf &h) {
        Eigen::MatrixXf pad = (w + h) * 0.5;
        Eigen::MatrixXf sz2 = (w + pad).cwiseProduct(h + pad);
        return sz2.cwiseSqrt();
    }

    float sz(const float w, const float h) {
        float pad = (w + h) * 0.5;
        float sz2 = (w + pad) * (h + pad);
        return std::sqrt(sz2);
    }

    Eigen::MatrixXf mxexp(Eigen::MatrixXf mx) {
        for (int i = 0; i < mx.rows(); ++i) {
            for (int j = 0; j < mx.cols(); ++j) {
                mx(i, j) = std::exp(mx(i, j));
            }
        }
        return mx;
    }


    struct AffineMatrix{
        float i2d[6];       // image to dst(network), 2x3 matrix
        float d2i[6];       // dst to image, 2x3 matrix

        void compute(const cv::Size& from, const cv::Size& to){
            float scale_x = to.width / (float)from.width;
            float scale_y = to.height / (float)from.height;
            float scale = std::min(scale_x, scale_y);

            i2d[0] = scale_x;  i2d[1] = 0;  i2d[2] = 0;
            i2d[3] = 0;  i2d[4] = scale_y;  i2d[5] = 0;

            cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
            cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
        }

        cv::Mat d2i_mat(){
            return {2, 3, CV_32F, d2i};
        }
    };


    class EstimatorImpl : public Estimator{
    public:
        ~EstimatorImpl() = default;

        bool startup(const std::string &superpoint_engine_path, const std::string &lightglue_engine_path, int gpuid){
            gpu_ = gpuid;
            TRT::set_device(gpuid);

            //////////////////////////////////////////////// 1.
            superpoint_model_ = TRT::load_infer(superpoint_engine_path);
            if(superpoint_model_ == nullptr){
                INFOE("Load model failed: %s", superpoint_engine_path.c_str());
                return false;
            }
            superpoint_model_->print();

            lightglue_model_ = TRT::load_infer(lightglue_engine_path);
            if(lightglue_model_ == nullptr){
                INFOE("Load model failed: %s", lightglue_engine_path.c_str());
                return false;
            }
            lightglue_model_->print();
//            //////////////////////////////////////////////////// !!!!
            stream_ = lightglue_model_->get_stream();

            return true;
        }

        bool startup(const std::string &superpoint_re_engine_path,int gpuid){
            gpu_ = gpuid;
            TRT::set_device(gpuid);

            //////////////////////////////////////////////// 1.
            superpoint_re_model_ = TRT::load_infer(superpoint_re_engine_path);
            if(superpoint_re_model_ == nullptr){
                INFOE("Load model failed: %s", superpoint_re_engine_path.c_str());
                return false;
            }
            superpoint_re_model_->print();

//            //////////////////////////////////////////////////// !!!!
            stream_re_ = superpoint_re_model_->get_stream();

            return true;
        }

        void sp_extractor(const cv::Mat &image) override {
            auto startTime = std::chrono::high_resolution_clock::now();
            auto startTime_sp = std::chrono::high_resolution_clock::now();
            auto startTime_sp_pre = std::chrono::high_resolution_clock::now();

            // 绑定输入输出
            sp_in_image = superpoint_model_->tensor("image");

            //--------------------SP预处理--------------------------
            //imread读到的图片为BGR
//            cv::Mat image_gray =  image.clone();

            height = image.rows;
            width = image.cols;

            auto tensor = sp_in_image;

            TRT::CUStream preprocess_stream = tensor->get_stream();

            //调这两个参数可以改变图像大小
            int width_adj,height_adj;
            width_adj = 752;
            height_adj = 480;

            AffineMatrix affineMatrix;
            cv::Size input_size(width_adj, 480);
            affineMatrix.compute(image.size(), input_size);

            tensor->resize(1, 1, 480, width_adj).to_gpu();

            size_t size_image = image.cols * image.rows * image.channels();
            // 对齐 32 字节
            size_t size_matrix = iLogger::upbound(sizeof(affineMatrix.d2i), 32);
            auto workspace = tensor->get_workspace();
            auto* gpu_workspace           = (uint8_t*)workspace->gpu(size_matrix + size_image);
            auto* affine_matrix_device    = (float*)gpu_workspace;
            uint8_t* image_device         = size_matrix + gpu_workspace;

            auto* cpu_workspace           = (uint8_t*)workspace->cpu(size_matrix + size_image);
            auto* affine_matrix_host      = (float*)cpu_workspace;
            uint8_t* image_host           = size_matrix + cpu_workspace;

            // speed up
            memcpy(image_host, image.data, size_image);
            memcpy(affine_matrix_host, affineMatrix.d2i, sizeof(affineMatrix.d2i));
            checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, preprocess_stream));
            checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affineMatrix.d2i), cudaMemcpyHostToDevice, preprocess_stream));

            auto normalize_ = CUDAKernel::Norm::alpha_beta(1 / 255.f, 0.f, CUDAKernel::ChannelType::Invert);

            auto cuda_resize_time_start = std::chrono::high_resolution_clock::now();

            CUDAKernel::warp_affine_bilinear_and_normalize_plane(
                    image.channels(),image_device, image.cols * image.channels(), image.cols, image.rows,
                    tensor->gpu<float>(), width_adj, 480,
                    affine_matrix_device, 114,
                    normalize_, preprocess_stream
            );

            auto cuda_resize_time_end = std::chrono::high_resolution_clock::now();
            float cuda_resize_time = std::chrono::duration<float, std::milli>(cuda_resize_time_end - cuda_resize_time_start).count();
            float resize_pre_time = std::chrono::duration<float, std::milli>(cuda_resize_time_start - startTime_sp_pre).count();


            auto copy_time_start = std::chrono::high_resolution_clock::now();
            //CV_32FC1能显示不能保存  CV_8UC1
            cv::Mat image_gray(480, width_adj, CV_32FC1);
            bool DEBUG_IMG = false;
            //DEBUG图像的话
            if(DEBUG_IMG)
            {
                checkCudaRuntime(cudaMemcpyAsync(image_gray.data, tensor->gpu<float>(), 480*width_adj*sizeof(float), cudaMemcpyDeviceToHost, tensor->get_stream()));
                checkCudaRuntime(cudaStreamSynchronize(tensor->get_stream()));
                cv::imwrite("../src/mixvpr/11111111.jpg", image_gray);
                cv::imshow("666",image_gray);
                cv::waitKey(0);
            }

            auto copy_time_end = std::chrono::high_resolution_clock::now();
            float copy_time = std::chrono::duration<float, std::milli>(copy_time_end - copy_time_start).count();

            auto endTime_sp_pre = std::chrono::high_resolution_clock::now();
            auto startTime_sp_inf = std::chrono::high_resolution_clock::now();
            //----------------推理-----------------------------------
//            sp_in_image->set_norm_mat_gray(0,image_gray);

            superpoint_model_->synchronize();
            superpoint_model_->forward();

            auto endTime_sp_inf = std::chrono::high_resolution_clock::now();
            auto startTime_sp_post = std::chrono::high_resolution_clock::now();

            sp_out_descriptors = superpoint_model_->output(0);
            sp_out_keypoints = superpoint_model_->output(1);
            sp_out_scores = superpoint_model_->output(2);

            //----------------------获取指针----------------------------
            float*  des = sp_out_descriptors->cpu<float>();
            int*  keypoints = sp_out_keypoints->cpu<int>();
            float*  scores =  sp_out_scores->cpu<float>();

            //---------------------提取数据 && SP后处理----------------------
            float shift_width = width_adj/2;        //float shift_width = image_gray.cols / 2;
            float shift_height = height_adj/2;       //float shift_height = image_gray.rows / 2;
            float scale = max(shift_width,shift_height);              //float scale = max(image_gray.cols,image_gray.rows)/2;

            auto kpts_dim = superpoint_model_->run_dims("keypoints");
            auto desc_dim = superpoint_model_->run_dims("descriptors");
            auto scores_dim = superpoint_model_->run_dims("scores");

            sp_out_keypoints->resize(kpts_dim);
            sp_out_descriptors->resize(desc_dim);
            sp_out_scores->resize(scores_dim);

            cv::Point2f norm;
            cv::Point2f ori;

            //normalize keypoints 标准化特征点 为lg的输入做准备
            //TODO!!!!!写成核函数的形式，再拷贝，可以实现优化加速
            for(int i = 0 ;i< sp_out_keypoints->shape(1)*2 ;i+=2)
            {
                norm.x = (*(keypoints+i) - shift_width) / scale;
                norm.y = (*(keypoints+i+1) - shift_height) / scale;
                sp_kpts_norm.push_back(norm);

                ori.x = *(keypoints+i);
                ori.y = *(keypoints+i+1);
                sp_kpts.push_back(ori);
            }

            sp_desc = vector<float>(des, des + sp_out_descriptors->shape(1)*256);
            sp_scores = vector<float>(scores,scores + sp_out_scores->shape(1));

            auto endTime_sp = std::chrono::high_resolution_clock::now();
            auto endTime_sp_post = std::chrono::high_resolution_clock::now();

            float sp_time = std::chrono::duration<float, std::milli>(endTime_sp - startTime_sp).count();
            float sp_time_pre = std::chrono::duration<float, std::milli>(endTime_sp_pre - startTime_sp_pre).count();
            float sp_time_post = std::chrono::duration<float, std::milli>(endTime_sp_post - startTime_sp_post).count();
            float sp_time_inf = std::chrono::duration<float, std::milli>(endTime_sp_inf- startTime_sp_inf).count();

            sp_time_tmp += sp_time;
            sp_num++;
            printf("time_total_sp = %f\n",sp_time);
            printf("time_aver_sp = %f\n",sp_time_tmp / sp_num);

//            printf("cuda_resize_time  = %f\n",cuda_resize_time);
//            printf("resize_pre_time  = %f\n",resize_pre_time);
//
//            cout<<"copy resized img from gpu time = "<<copy_time<<endl;
//
//            printf("sp_pre time = %f\n",sp_time_pre);
//            printf("sp_inf time = %f\n",sp_time_inf);
//            printf("sp_post time = %f\n",sp_time_post);

            cout<<"----------------------------------"<<endl;
            checkCudaRuntime(cudaDeviceSynchronize());
        }

        void sp_extractor( const cv::Mat& img, vector<cv::Point2f> &sp_kpts ) override{

            auto startTime = std::chrono::high_resolution_clock::now();
            auto startTime_sp_re = std::chrono::high_resolution_clock::now();
            auto startTime_sp_re_pre = std::chrono::high_resolution_clock::now();

            superpoint_re_model_->set_run_dims("keypoints_r",{1,(int)sp_kpts.size(),2});

            // 绑定输入输出
            sp_re_in_image = superpoint_re_model_->tensor("image_r");
            sp_re_in_keypoints = superpoint_re_model_->tensor("keypoints_r");


            sp_re_in_keypoints->set_kpts_to_tensor(sp_kpts);


            //--------------------sp_re预处理--------------------------
            //imread读到的图片为BGR

            height = img.rows;
            width = img.cols;

            auto tensor = sp_re_in_image;

            TRT::CUStream preprocess_stream = tensor->get_stream();

            //调这两个参数可以改变图像大小
            int width_adj,height_adj;
            width_adj = 752;
            height_adj = 480;


            AffineMatrix affineMatrix;
            cv::Size input_size(width_adj, height_adj);
            affineMatrix.compute(img.size(), input_size);

            tensor->resize(1, 1, height_adj, width_adj).to_gpu();

            size_t size_image = img.cols * img.rows * img.channels();
            // 对齐 32 字节
            size_t size_matrix = iLogger::upbound(sizeof(affineMatrix.d2i), 32);
            auto workspace = tensor->get_workspace();
            auto* gpu_workspace           = (uint8_t*)workspace->gpu(size_matrix + size_image);
            auto* affine_matrix_device    = (float*)gpu_workspace;
            uint8_t* image_device         = size_matrix + gpu_workspace;

            auto* cpu_workspace           = (uint8_t*)workspace->cpu(size_matrix + size_image);
            auto* affine_matrix_host      = (float*)cpu_workspace;
            uint8_t* image_host           = size_matrix + cpu_workspace;

            // sp_reeed up
            memcpy(image_host, img.data, size_image);
            memcpy(affine_matrix_host, affineMatrix.d2i, sizeof(affineMatrix.d2i));
            checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, preprocess_stream));
            checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affineMatrix.d2i), cudaMemcpyHostToDevice, preprocess_stream));

            auto normalize_ = CUDAKernel::Norm::alpha_beta(1 / 255.f, 0.f, CUDAKernel::ChannelType::Invert);

            auto cuda_resize_time_start = std::chrono::high_resolution_clock::now();
            checkCudaRuntime(cudaDeviceSynchronize());

            CUDAKernel::warp_affine_bilinear_and_normalize_plane(
                    img.channels(),image_device, img.cols * img.channels(), img.cols, img.rows,
                    tensor->gpu<float>(), width_adj, height_adj,
                    affine_matrix_device, 114,
                    normalize_, preprocess_stream
            );

            auto cuda_resize_time_end = std::chrono::high_resolution_clock::now();
            auto copy_time_start = std::chrono::high_resolution_clock::now();

            cv::Mat image_gray(480, width_adj, CV_32FC1);
            bool DEBUG_IMG = false;

            //DEBUG图像的话
            if(DEBUG_IMG)
            {
                checkCudaRuntime(cudaMemcpyAsync(image_gray.data, tensor->gpu<float>(), height_adj*width_adj*sizeof(float), cudaMemcpyDeviceToHost, tensor->get_stream()));
                checkCudaRuntime(cudaStreamSynchronize(tensor->get_stream()));
                cv::imwrite("../src/mixvpr/2.jpg", image_gray);
                cv::imshow("666",image_gray);
                cv::waitKey(0);
            }

            auto copy_time_end = std::chrono::high_resolution_clock::now();
            auto endTime_sp_re_pre = std::chrono::high_resolution_clock::now();
            auto startTime_sp_re_inf = std::chrono::high_resolution_clock::now();

            //----------------推理-----------------------------------
            superpoint_re_model_->synchronize();
            superpoint_re_model_->forward();

            sp_re_out_scores = superpoint_re_model_->output(0);
            sp_re_out_descriptors = superpoint_re_model_->output(1);


            auto scores_re_dim = superpoint_re_model_->run_dims("scores_r");
            auto des_re_dim = superpoint_re_model_->run_dims("des_r");

            sp_re_out_descriptors->resize(des_re_dim);
            sp_re_out_scores->resize(scores_re_dim);

            sp_re_desc = std::vector<float>(sp_re_out_descriptors->cpu<float>(),
                                            sp_re_out_descriptors->cpu<float>() +sp_re_out_descriptors->numel()
            );

            auto endTime_sp_re = std::chrono::high_resolution_clock::now();

            float sp_re_time = std::chrono::duration<float, std::milli>(endTime_sp_re - startTime_sp_re).count();

            sp_re_num++;

            sp_re_time_tmp += sp_re_time;
            cout<<"sp_re_total_time= " << sp_re_time <<endl;
            cout<<"sp_re_aver_time= " << sp_re_time_tmp / sp_re_num <<endl;

        }

        void lg_matcher(std::vector<cv::Point2f> &lg_in_kpts0_,
                        std::vector<cv::Point2f> &lg_in_kpts1_,
                        std::vector<float> &lg_in_desc0_,
                        std::vector<float> &lg_in_desc1_ ,
                        int height_0,int width_0,
                        int height_1,int width_1) override {

            auto startTime = std::chrono::high_resolution_clock::now();
            auto startTime_lg = std::chrono::high_resolution_clock::now();
            auto startTime_lg_pre = std::chrono::high_resolution_clock::now();

            //---------------------提取数据 && SP后处理----------------------
            //640*480
//            float shift_width = 320;    // image.width/2
//            float shift_height = 240;   //image.height/2
//            float scale = 320;          //max(image.height,image.width)/2

            //如果是vins的480x752.
            float shift_width = 376;    // image.width/2
            float shift_height = 240;   //image.height/2
            float scale = 376;          //max(image.height,image.width)/2

            lightglue_model_->set_run_dims("kpts0",{1,(int)lg_in_kpts0_.size(),2});
            lightglue_model_->set_run_dims("kpts1",{1,(int)lg_in_kpts1_.size(),2});
            lightglue_model_->set_run_dims("desc0",{1,(int)lg_in_kpts0_.size(),256});
            lightglue_model_->set_run_dims("desc1",{1,(int)lg_in_kpts1_.size(),256});

//            lg_in_kpts0_norm = lightglue_model_->input(0);
//            lg_in_kpts1_norm = lightglue_model_->input(1);

            lg_in_kpts0 = lightglue_model_->input(0);
            lg_in_kpts1 = lightglue_model_->input(1);
            lg_in_desc0 = lightglue_model_->input(2);
            lg_in_desc1 = lightglue_model_->input(3);

//            lg_in_kpts0 = lg_in_kpts0_norm->clone();
//            lg_in_kpts1 = lg_in_kpts1_norm->clone();


            lg_in_kpts0->set_kpts_to_tensor(lg_in_kpts0_);
            lg_in_kpts1->set_kpts_to_tensor(lg_in_kpts1_);

//            lg_in_kpts0_norm->set_kpts_to_tensor(lg_in_kpts0_);
//            lg_in_kpts1_norm->set_kpts_to_tensor(lg_in_kpts1_);


            lg_in_desc0->set_vec_to_tensor(lg_in_desc0_);
            lg_in_desc1->set_vec_to_tensor(lg_in_desc1_);

            CUDAKernel::normalize_kpts(lg_in_kpts0->gpu<float>(),lg_in_kpts0->gpu<float>(),lg_in_kpts0->numel(),
                                       shift_width , shift_height , scale,
                                       lightglue_model_->get_stream());

            CUDAKernel::normalize_kpts(lg_in_kpts1->gpu<float>(),lg_in_kpts1->gpu<float>(),lg_in_kpts1->numel(),
                                       shift_width , shift_height , scale,
                                       lightglue_model_->get_stream());


//            CUDAKernel::normalize_kpts(lg_in_kpts0_norm->gpu<float>(),lg_in_kpts0_norm->gpu<float>(),lg_in_kpts0_norm->numel(),
//                                       shift_width , shift_height , scale,
//                                       lightglue_model_->get_stream());
//
//            CUDAKernel::normalize_kpts(lg_in_kpts1_norm->gpu<float>(),lg_in_kpts1_norm->gpu<float>(),lg_in_kpts1_norm->numel(),
//                                       shift_width , shift_height , scale,
//                                       lightglue_model_->get_stream());


//            lg_in_desc0->copy_from_gpu(0,sp_out_descriptors->gpu(),lg_in_desc0->numel());
//            lg_in_desc1->copy_from_gpu(0,sp_out_descriptors->gpu(),lg_in_desc1->numel());

            auto endTime_lg_pre = std::chrono::high_resolution_clock::now();
            auto startTime_lg_inf = std::chrono::high_resolution_clock::now();

            lightglue_model_->synchronize();
            lightglue_model_->forward();

            auto endTime_lg_inf = std::chrono::high_resolution_clock::now();
            auto startTime_lg_post = std::chrono::high_resolution_clock::now();

            //----------------后处理---------------------

            //原图大小731h x 1024w
            // 匹配之后用到的 0.625 = 640/1024   0.656634748 = 480 / 731
            //scale = [w  ,h ]
//            float scale_width = 640.f/width_0;
//            float scale_height = 480.f/height_0;
//
//            float scale_width_0 = 640.f/width_0;
//            float scale_height_0 = 480.f/height_0;
//
//            float scale_width_1 = 640.f/width_1;
//            float scale_height_1 = 480.f/height_1;

            float scale_width = 752.f/width_0;
            float scale_height = 480.f/height_0;

            float scale_width_0 = 752.f/width_0;
            float scale_height_0 = 480.f/height_0;

            float scale_width_1 = 752.f/width_1;
            float scale_height_1 = 480.f/height_1;


            //绑定输入输出
            lg_out_matches0 = lightglue_model_->output(0);
            lg_out_mscores0 = lightglue_model_->output(1);

            auto lg_match_dims = lightglue_model_->run_dims("matches0");
            auto lg_scores_dims = lightglue_model_->run_dims("mscores0");

            shared_ptr<TRT::Tensor> a =lg_out_matches0;
            auto * test = a->cpu<int>();
            lg_out_matches0->resize(lg_match_dims).to_gpu(false);
            lg_out_mscores0->resize(lg_scores_dims).to_gpu(false);
            mkpts0 = lg_out_matches0->clone();
            mkpts1 = lg_out_matches0->clone();

//            for(int i =0;i<lg_in_kpts0->shape(1);i++)
//            {
//                cout<<lg_in_kpts0->cpu<float>()[i]<<endl;
////                cout<<sp_out_keypoints->cpu<int>()[i]<<endl;
//            }
//            CUDAKernel::matches_post_process(sp_out_keypoints->gpu<int>(),sp_out_keypoints->gpu<int>(),
//                                             lg_out_matches0->gpu<int>(),
//                                             mkpts0->gpu<float>(), mkpts1->gpu<float>(),
//                                             scale_width,scale_height,
//                                             sp_out_keypoints->shape(1)*2,
//                                             lg_out_matches0->shape(0)*2,
//                                             lightglue_model_->get_stream());

            CUDAKernel::matches_post_process(lg_in_kpts0->gpu<float>(),lg_in_kpts1->gpu<float>(),
                                             lg_out_matches0->gpu<int>(),
                                             mkpts0->gpu<float>(), mkpts1->gpu<float>(),

                                             shift_width,shift_height,

                                             scale_width_0,scale_height_0,
                                             scale_width_1,scale_height_1,

                                             lg_in_kpts0_.size()*2,
                                             lg_out_matches0->shape(0)*2,
                                             lightglue_model_->get_stream());


//            for(int i =0;i<lg_out_matches0->shape(0);i++)
//            {
////                cout<<lg_out_matches0->cpu<int>()[i]<<endl;
//                cout<<lg_out_matches0->cpu<int>()[i]<<endl;
//            }

            float * c = mkpts0->cpu<float>();
            float * d = mkpts1->cpu<float>();

            cv::Point2f m0;
            cv::Point2f m1;
            for(int i=0; i<mkpts0->shape(0)*2; i=i+2)
            {
                m0.x  = *(c+i);
                m0.y  = *(c+i+1);

                m1.x  = *(d+i);
                m1.y  = *(d+i+1);

                lg_mkpts0.push_back(m0);
                lg_mkpts1.push_back(m1);
            }

            lg_scores.insert(lg_scores.begin(),
                             lg_out_mscores0->cpu<float>(),
                             lg_out_mscores0->cpu<float>()+lg_out_matches0->shape(0));
            lg_matches.insert(lg_matches.begin(),
                              lg_out_matches0->cpu<int>(),
                              lg_out_matches0->cpu<int>()+lg_out_matches0->numel());


            auto endTime_lg_post = std::chrono::high_resolution_clock::now();
            auto endTime_lg = std::chrono::high_resolution_clock::now();
            auto endTime = std::chrono::high_resolution_clock::now();

            float lg_time = std::chrono::duration<float, std::milli>(endTime_lg - startTime_lg).count();
            float lg_time_pre = std::chrono::duration<float, std::milli>(endTime_lg_pre - startTime_lg_pre).count();
            float lg_time_post = std::chrono::duration<float, std::milli>(endTime_lg_post - startTime_lg_post).count();
            float lg_time_inf = std::chrono::duration<float, std::milli>(endTime_lg_inf- startTime_lg_inf).count();

            lg_num++;
            lg_time_tmp += lg_time;
            printf("time_total_lg = %f\n",lg_time);
            printf("time_aver_lg = %f\n",lg_time_tmp / lg_num);
//            printf("lg_pre time = %f\n",lg_time_pre);
//            printf("lg_inf time = %f\n",lg_time_inf);
//            printf("lg_post time = %f\n",lg_time_post);

            cout<<"----------------------------------"<<endl;
        }


        void lg_matcher() override {

            auto startTime = std::chrono::high_resolution_clock::now();
            auto startTime_lg = std::chrono::high_resolution_clock::now();
            auto startTime_lg_pre = std::chrono::high_resolution_clock::now();

            //---------------------提取数据 && SP后处理----------------------
            float shift_width = 320;    // image.width/2
            float shift_height = 240;   //image.height/2
            float scale = 320;          //max(image.height,image.width)/2

            auto kpts_dim = superpoint_model_->run_dims("keypoints");
            auto desc_dim = superpoint_model_->run_dims("descriptors");
//            auto kpts0_dim;

            lightglue_model_->set_run_dims("kpts0",kpts_dim);
            lightglue_model_->set_run_dims("kpts1",kpts_dim);
            lightglue_model_->set_run_dims("desc0",desc_dim);
            lightglue_model_->set_run_dims("desc1",desc_dim);

            lg_in_kpts0 = lightglue_model_->input(0);
            lg_in_kpts1 = lightglue_model_->input(1);
            lg_in_desc0 = lightglue_model_->input(2);
            lg_in_desc1 = lightglue_model_->input(3);


            CUDAKernel::normalize_kpts(sp_out_keypoints->gpu<int>(),lg_in_kpts0->gpu<float>(),1024,
                                       shift_width , shift_height , scale,
                                       superpoint_model_->get_stream());

            CUDAKernel::normalize_kpts(sp_out_keypoints->gpu<int>(),lg_in_kpts1->gpu<float>(),1024,
                                       shift_width , shift_height , scale,
                                       superpoint_model_->get_stream());

            lg_in_desc0->copy_from_gpu(0,sp_out_descriptors->gpu(),lg_in_desc0->numel());
            lg_in_desc1->copy_from_gpu(0,sp_out_descriptors->gpu(),lg_in_desc1->numel());

            auto endTime_lg_pre = std::chrono::high_resolution_clock::now();
            auto startTime_lg_inf = std::chrono::high_resolution_clock::now();

            lightglue_model_->synchronize();
            lightglue_model_->forward();

            auto endTime_lg_inf = std::chrono::high_resolution_clock::now();
            auto startTime_lg_post = std::chrono::high_resolution_clock::now();

//            cout<<lg_in_kpts0->cpu<float>()[0]<<endl;
//            cout<<lg_in_kpts0->cpu<float>()[1]<<endl;
//            cout<<lg_in_kpts0->cpu<float>()[2]<<endl;
//            cout<<lg_in_kpts0->cpu<float>()[3]<<endl;

            cout<<lg_in_desc0->shape(0) << " "<<lg_in_desc0->shape(1)<<" " << lg_in_desc0->shape(2)<<endl;
            cout<<lg_in_desc1->shape(0) << " "<<lg_in_desc1->shape(1)<<" " << lg_in_desc1->shape(2)<<endl;

            //----------------后处理---------------------

            //原图大小731h x 1024w
            // 匹配之后用到的 0.625 = 640/1024   0.656634748 = 480 / 731
            //scale = [w  ,h ]
            float scale_width = 640.f/width;
            float scale_height = 480.f/height;

            //绑定输入输出
            lg_out_matches0 = lightglue_model_->output(0);
            lg_out_mscores0 = lightglue_model_->output(1);

            auto lg_match_dims = lightglue_model_->run_dims("matches0");
            auto lg_scores_dims = lightglue_model_->run_dims("mscores0");

            lg_out_matches0->resize(lg_match_dims);
            lg_out_mscores0->resize(lg_scores_dims);
            mkpts0 = lg_out_matches0->clone();
            mkpts1 = lg_out_matches0->clone();

//            for(int i=0;i<sp_out_keypoints->shape(1);i++)
//            {
//                cout<<sp_out_keypoints->cpu<int>()[i]<<endl;
//            }

            CUDAKernel::matches_post_process(sp_out_keypoints->gpu<int>(),sp_out_keypoints->gpu<int>(),
                                             lg_out_matches0->gpu<int>(),
                                             mkpts0->gpu<float>(), mkpts1->gpu<float>(),
                                             scale_width,scale_height,
                                             sp_out_keypoints->shape(1)*2,
                                             lg_out_matches0->shape(0)*2,
                                             lightglue_model_->get_stream());

            float * c = mkpts0->cpu<float>();
            float * d = mkpts1->cpu<float>();

            cv::Point2f m0;
            cv::Point2f m1;
            for(int i=0; i<mkpts0->shape(0)*2; i=i+2)
            {
                m0.x  = *(c+i);
                m0.y  = *(c+i+1);

                m1.x  = *(d+i);
                m1.y  = *(d+i+1);

                lg_mkpts0.push_back(m0);
                lg_mkpts1.push_back(m1);
            }

            lg_scores.insert(lg_scores.begin(),
                             lg_out_mscores0->cpu<float>(),
                             lg_out_mscores0->cpu<float>()+lg_out_matches0->shape(0));
            lg_matches.insert(lg_matches.begin(),
                              lg_out_matches0->cpu<int>(),
                              lg_out_matches0->cpu<int>()+lg_out_matches0->shape(0)*lg_out_matches0->shape(1));

            cout<<"mkps size = "<<lg_mkpts0.size()<<endl;
            cout<<lg_matches.size()<<endl;

            auto endTime_lg_post = std::chrono::high_resolution_clock::now();
            auto endTime_lg = std::chrono::high_resolution_clock::now();
            auto endTime = std::chrono::high_resolution_clock::now();

            float lg_time = std::chrono::duration<float, std::milli>(endTime_lg - startTime_lg).count();
            float lg_time_pre = std::chrono::duration<float, std::milli>(endTime_lg_pre - startTime_lg_pre).count();
            float lg_time_post = std::chrono::duration<float, std::milli>(endTime_lg_post - startTime_lg_post).count();
            float lg_time_inf = std::chrono::duration<float, std::milli>(endTime_lg_inf- startTime_lg_inf).count();

            printf("time_total_lg = %f\n",lg_time);
            printf("time_aver_lg = %f\n",lg_time);
            printf("lg_pre time = %f\n",lg_time_pre);
            printf("lg_inf time = %f\n",lg_time_inf);
            printf("lg_post time = %f\n",lg_time_post);

            cout<<"----------------------------------"<<endl;
        }

    private:

        //模型
        shared_ptr<TRT::Infer> superpoint_model_;
        shared_ptr<TRT::Tensor> sp_in_image;

        shared_ptr<TRT::Tensor> sp_out_keypoints;
        shared_ptr<TRT::Tensor> sp_out_scores;
        shared_ptr<TRT::Tensor> sp_out_descriptors;


        shared_ptr<TRT::Infer> superpoint_re_model_;
        shared_ptr<TRT::Tensor> sp_re_in_image;

        shared_ptr<TRT::Tensor> sp_re_in_keypoints;
        shared_ptr<TRT::Tensor> sp_re_out_scores;
        shared_ptr<TRT::Tensor> sp_re_out_descriptors;


        shared_ptr<TRT::Infer> lightglue_model_;
        shared_ptr<TRT::Tensor> lg_in_kpts0;
        shared_ptr<TRT::Tensor> lg_in_kpts1;

        shared_ptr<TRT::Tensor> lg_in_kpts0_norm;
        shared_ptr<TRT::Tensor> lg_in_kpts1_norm;

        shared_ptr<TRT::Tensor> lg_in_desc0;
        shared_ptr<TRT::Tensor> lg_in_desc1;
        shared_ptr<TRT::Tensor> lg_out_matches0;
        shared_ptr<TRT::Tensor> lg_out_mscores0;

        shared_ptr<TRT::Tensor> mkpts0;
        shared_ptr<TRT::Tensor> mkpts1;

        TRT::CUStream stream_ = nullptr;
        TRT::CUStream stream_re_ = nullptr;

        double sp_time_tmp = 0;
        double sp_re_time_tmp = 0;
        double lg_time_tmp = 0;

        int sp_num = 0;
        int sp_re_num = 0;
        int lg_num = 0;


        int gpu_ = 0;

    };

    shared_ptr<Estimator> creat_estimator(const std::string & superpoint_engine_path,const std::string &lightglue_engine_path, int gpuid){
        shared_ptr<EstimatorImpl> instance(new EstimatorImpl{});
        if(!instance->startup(superpoint_engine_path, lightglue_engine_path , gpuid))
            instance.reset();
        return instance;
    }

    shared_ptr<Estimator> recover_estimator(const std::string & superpoint_re_engine_path, int gpuid){
        shared_ptr<EstimatorImpl> instance(new EstimatorImpl{});
        if(!instance->startup(superpoint_re_engine_path,gpuid))
            instance.reset();
        return instance;
    }

    void Estimator::img_callback(const sensor_msgs::CompressedImage msg) {

        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::TYPE_8UC3);
        image =cv_ptr->image;
//            image.frame_id =  msg.header.frame_id;
//            image.timestamp = msg.header.stamp;
//            cv::imshow("img",cv_ptr->image);
//            cv::waitKey(0);

        sp_extractor(image);


    }
}
