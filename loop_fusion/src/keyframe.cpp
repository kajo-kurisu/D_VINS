/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "keyframe.h"

// 64-bit int
using idx_t = faiss::Index::idx_t;
std::vector<float> descriptors_database;
using namespace std;

double PNP_INFLATION;
int MIN_LOOP_NUM;
bool USE_SP;
double MAX_THETA_DIFF;
double MAX_POSE_DIFF;

//shared_ptr<Estimator_net::Estimator> deep_net_estimator;

namespace backward{
    backward::SignalHandling sh;
}


template <typename Derived>
static void reduceVector(vector<Derived> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

// create keyframe online
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
                   vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_norm,
                   vector<double> &_point_id, int _sequence)
{
    time_stamp = _time_stamp;
    index = _index;
    vio_T_w_i = _vio_T_w_i;
    vio_R_w_i = _vio_R_w_i;
    T_w_i = vio_T_w_i;
    R_w_i = vio_R_w_i;
    origin_vio_T = vio_T_w_i;
    origin_vio_R = vio_R_w_i;
    image = _image.clone();
    cv::resize(image, thumbnail, cv::Size(80, 60));
    point_3d = _point_3d;
    point_2d_uv = _point_2d_uv;
    point_2d_norm = _point_2d_norm;
    point_id = _point_id;
    has_loop = false;
    loop_index = -1;
    has_fast_point = false;
    loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
    sequence = _sequence;

//    deep_net_estimator = Estimator_net::creat_estimator(sp_engine_path,lg_engine_path,0);
    computeWindowSuperpoint();
    computeSuperpoint();

//    computeWindowBRIEFPoint();
//    computeBRIEFPoint();

    compute_mix_des();
    sort_vec_faiss();



//    if(!DEBUG_IMAGE)
//        image.release();
}

// load previous keyframe
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
                   cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1 > &_loop_info,
                   vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors)
{
    time_stamp = _time_stamp;
    index = _index;
    //vio_T_w_i = _vio_T_w_i;
    //vio_R_w_i = _vio_R_w_i;
    vio_T_w_i = _T_w_i;
    vio_R_w_i = _R_w_i;
    T_w_i = _T_w_i;
    R_w_i = _R_w_i;
    if (DEBUG_IMAGE)
    {
        image = _image.clone();
        cv::resize(image, thumbnail, cv::Size(80, 60));
    }
    if (_loop_index != -1)
        has_loop = true;
    else
        has_loop = false;
    loop_index = _loop_index;
    loop_info = _loop_info;
    has_fast_point = false;
    sequence = 0;
    keypoints = _keypoints;
    keypoints_norm = _keypoints_norm;
    brief_descriptors = _brief_descriptors;
}



void KeyFrame::compute_mix_des() {
    float time;
    TRTLogger logger;
    frame frame;
    cv::Mat image_mix;

    // 创建两个事件，一个开始事件，一个结束事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 在默认的stream记录开始事件
    cudaEventRecord(start);


    //-------------------------- 1.加载模型 ----------------------------
    static auto engine_data = load_file(mix_engine_path);

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

    cv::resize(image, image_mix, cv::Size(input_width, input_height));

    cv::cvtColor(image_mix,image_mix,CV_GRAY2BGR,3);

    int image_area = image_mix.cols * image_mix.rows;
    unsigned char* pimage = image_mix.data; //BGRBGRBGR TO BBBGGGRRR   用地址偏移做索引


//    checkRuntime(cudaMalloc(&input_data_device, image_num*input_numel * sizeof(float)));

//    cout<<"channel = "<<image_mix.channels()<<endl;
    //------------------使用多张图片做测试------------------------
    for(int i =0;i<image_num;i++)
    {
        float* phost_b = input_data_host + image_area *(3*i);
        float* phost_g = input_data_host + image_area *(3*i+1) ;
        float* phost_r = input_data_host + image_area *(3*i+2) ;
//        pimage = image.data;  //执行完一张图片后，下一个循环重新指向图片的收个像素的指针

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

//    cout<<"start infrence"<<endl;
    //------------------------ 7.推理 --------------------------
    bool success = execution_context->enqueueV2((void**)bindings, stream,
                                                nullptr);


    checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device,
                                 sizeof(output_data_host), cudaMemcpyDeviceToHost, stream));

    checkRuntime(cudaStreamSynchronize(stream));

//    mix_descriptors.insert(mix_descriptors.begin(),output_data_host, output_data_host + 512);
    mix_descriptors = std::vector<float> (output_data_host, output_data_host + 512);
    descriptors_database.insert(descriptors_database.end(),output_data_host,output_data_host+512);

//--------------------8.按照创建相反的顺序释放内存 ----------------------

    // 推理完成后，记录结束事件
    cudaEventRecord(stop);
    // 等待结束事件完成
    cudaEventSynchronize(stop);


    // 计算两个event之间的间隔时间
//    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("time for extracting mix des= %f\n",time);

    // 销毁事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFreeHost(input_data_host));
    checkRuntime(cudaFree(input_data_device));

    checkRuntime(cudaFree(output_data_device));

}


void KeyFrame::sort_vec_faiss( )
{
    int d = 512;      // dimension
    int nb;  // database size
    int nq = 1;  // nb of queries

    float* xb;
    float* xq;
//    std::vector<float> test;
    //构造数据集db
//    xb = descriptors_database.data();

//6400=50*128

//    cout<<"index = "<<index<<endl;
//    cout<<"size of xb = "<<endl;
    std::vector<float> test;

    int k;
    int window_size =25;
    if(index>=window_size)
    {
        test.resize((index-window_size+1)*512);
        test = vector<float>(descriptors_database.begin(),descriptors_database.begin()+(index-window_size+1)*512);
//        memcpy(test.data(),descriptors_database.data(),(index-24)*512);
        xb = test.data();
        nb = test.size()/512;
//        k = index-24;
//        cout<<"k= "<<k<<" nb = "<<nb<<endl;
    }
    else
    {
        xb = descriptors_database.data();
        nb = index+1;
//        k=nb;
    }


    //k=nb表示对全部向量进行索引，用于DEBUG
//    k=nb;

    k = 5;
    //选择xq
    xq = mix_descriptors.data();
    printf("now index = %d\n", index);


    auto startTime = std::chrono::high_resolution_clock::now();
//    faiss::IndexFlatL2 index(d); // call constructor
    faiss::IndexFlatIP index(d); // call constructor

//    faiss::IndexFlatCodes index(d);
//    printf("is_trained = %s\n", index.is_trained ? "true" : "false");
    index.add(nb, xb); // add vectors to the index
//    printf("ntotal = %zd\n", index.ntotal);

    //开始真正的向量检索，nq表示待检索的向量数量，xq表示待检索的向量，k表示top k ,D  和 I 表示返回的距离和索引Index
    { // search xq
        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];

        index.search(nq, xq, k, D, I);

        // print results
        //打印最相似的几个向量索引
        printf("I (top k  first results)=\n");
        for (int i = 0; i < nq; i++) {
            for (int j = 0; j < k; j++)
            {
                printf("%5zd ", I[i * k + j]);
                top_sim_index.push_back(I[i * k + j]);
            }
            printf("\n");
        }

        printf("D=\n");

        for (int i = 0; i < nq; i++) {
            for (int j = 0; j < k; j++)
            {
                printf("%7g  ", D[i * k + j]);
                top_sim.push_back(D[i * k + j]);
            }

            printf("\n");
        }

        delete[] I;
        delete[] D;
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    float faiss_sort_time = std::chrono::duration<float, std::milli>(endTime - startTime).count();
    cout<<"total time in faiss = "<<faiss_sort_time<<endl;

}


void KeyFrame::computeWindowSuperpoint()
{
    for(auto & i : point_2d_uv)
    {
        cv::KeyPoint key;
        key.pt.y = i.y;
        key.pt.x = i.x;
        window_keypoints.push_back(key);
    }

    cv::Mat test;
    image.convertTo(test,CV_8UC1);

    static auto deep_net_estimator_window = Estimator_net::recover_estimator(sp_re_engine_path,0);

    deep_net_estimator_window->sp_re_desc.clear();

    cout<<"---in sp re ---"<<endl;
    cout<<"sp re window "<<index<<endl;
    cout<<"point_2d_uv.size= "<<point_2d_uv.size()<<endl;

    if(point_2d_uv.size()>0)
        deep_net_estimator_window->sp_extractor(image,point_2d_uv);

    window_local_descriptors = deep_net_estimator_window->sp_re_desc;

}

void KeyFrame::computeSuperpoint(){

    vector<cv::Point2f> _keypoints;

    static auto  deep_net_estimator = Estimator_net::creat_estimator(sp_engine_path,sp_engine_path,0);

    deep_net_estimator->sp_kpts.clear();
    deep_net_estimator->sp_kpts_norm.clear();
    deep_net_estimator->sp_scores.clear();
    deep_net_estimator->sp_desc.clear();

//    local_descriptors.clear();


//    cout<<image.size<<endl;  480 x 752

    cv::Mat test;
    image.convertTo(test,CV_8UC1);

    cout<<"---in normal sp---"<<endl;
    cout<<"sp "<<index<<endl;

    deep_net_estimator->sp_extractor(test);

    _keypoints = deep_net_estimator->sp_kpts;
    scores = deep_net_estimator->sp_scores;
    local_descriptors = deep_net_estimator->sp_desc;

    if(USE_SP)
    {
        //把 superpoint 提取的点输入 keypoints 中，不需要归一化，统一在lg中进行归一化
        for(auto & i : _keypoints)
        {
            cv::KeyPoint key;
            key.pt.y = i.y;
            key.pt.x = i.x;
            keypoints.push_back(key);
        }
    }
    else
        local_descriptors.clear();

    //直接把前端的特征点输入keypoints中，没有归一化
    // push back the uvs used in vio
    float shift_width = image.cols/2;
    float shift_height = image.rows/2;
    float scale = max(shift_width,shift_height);

//    cout<<image.size<<endl;
    //pt_2d_uv 原始图像大小为480x752
    for(auto & i : point_2d_uv)
    {
        cv::KeyPoint key;
        cv::KeyPoint key_test;
//        key.pt.x = (i.x - shift_width) / scale ;
//        key.pt.y = (i.y - shift_height) / scale ;

        key_test.pt.y = i.y ;
        key_test.pt.x = i.x ;
        keypoints.push_back(key_test);
    }

    cout<<"keypoints.size() ="<<keypoints.size() <<" sp_keypoints.size() " <<_keypoints.size()<<"+"<<" point2d_uv.size"<<point_2d_uv.size()<<endl;



    local_descriptors.insert(local_descriptors.end(),
                             window_local_descriptors.begin(),
                             window_local_descriptors.end());

//    local_descriptors = window_local_descriptors;

    for (int i = 0; i < (int)keypoints.size(); i++)
    {
        Eigen::Vector3d tmp_p;
        m_camera->liftProjective(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
        cv::KeyPoint tmp_norm;
        tmp_norm.pt = cv::Point2f(tmp_p.x()/tmp_p.z(), tmp_p.y()/tmp_p.z());
        keypoints_norm.push_back(tmp_norm);
    }

}

void KeyFrame::debug(KeyFrame* old_kf) {

    static auto sp = Estimator_net::creat_estimator(sp_engine_path,lg_engine_path,0);

    static auto sp_re = Estimator_net::recover_estimator(sp_re_engine_path,0);

    static auto sp_1 = Estimator_net::creat_estimator(sp_engine_path, lg_engine_path, 0);

    cout<<"-------------------in debug--------------------"<<endl;
    sp->lg_matches.clear();
    sp->lg_scores.clear();
    sp->lg_mkpts0.clear();
    sp->lg_mkpts1.clear();

    sp->sp_kpts.clear();
    sp_1->sp_kpts.clear();


//    cout<<image.empty()<<endl;
    sp->sp_extractor(image);

    sp_1->sp_extractor(old_kf->image);

    cout<<sp->sp_kpts.size()<<endl;
    cout<<sp_1->sp_kpts.size()<<endl;

    sp_re->sp_extractor(image,old_kf->point_2d_uv);

    sp->lg_matcher(point_2d_uv,old_kf->point_2d_uv,
                   local_descriptors, sp_re->sp_re_desc,
                   image.rows,image.cols,
                   image.rows,image.cols);

    cout<<"debug lg_mkpts size = "<<sp->lg_matches.size()<<endl;

    vector<cv::Point2f> _keypoints, _keypoints_old;

    _keypoints = point_2d_uv;
    _keypoints_old = old_kf->point_2d_uv;

    vector<int> match_index(_keypoints.size(),-1);

    //获取匹配对 yuanbende
//    for(int i =0;i<deep_net_estimator_lg->lg_mkpts0.size();i++)
//    {
//        match_index[i] = deep_net_estimator_lg->lg_matches[2*i];
//    }

    for(int i =0;i<sp->lg_mkpts0.size();i++)
    {
        match_index[sp->lg_matches[2*i+1]] = sp->lg_matches[2*i+1];
    }

//    for (int i : match_index){
//        cv::Point2f pt(0.f, 0.f);
//        cv::Point2f pt_norm(0.f, 0.f);
//
//        status[i] = 1;
//
//        pt = _keypoints_old[i];
//        pt_norm = keypoints_old_norm[i].pt;
//
//        matched_2d_old.push_back(pt);
//        matched_2d_old_norm.push_back(pt_norm);
//
//    }

    vector<cv::Point2f> matched_2d_old, matched_2d_old_norm,matched_2d_cur;
    vector<cv::KeyPoint> keypoints_old_norm;
    vector<uchar> status ;

    matched_2d_cur = point_2d_uv;

    for (int i = 0; i < (int)_keypoints_old.size(); i++)
    {
        Eigen::Vector3d tmp_p;
        m_camera->liftProjective(Eigen::Vector2d(_keypoints_old[i].x, _keypoints_old[i].y), tmp_p);
        cv::KeyPoint tmp_norm;
        tmp_norm.pt = cv::Point2f(tmp_p.x()/tmp_p.z(), tmp_p.y()/tmp_p.z());
        keypoints_old_norm.push_back(tmp_norm);
    }

    for(int i : match_index)
    {
        cv::Point2f pt(0.f, 0.f);
        cv::Point2f pt_norm(0.f, 0.f);
        if (i >= 0){
            status.push_back(1);
            pt = _keypoints_old[i];
            pt_norm = keypoints_old_norm[i].pt;
        }
        else
            status.push_back(0);
        matched_2d_old.push_back(pt);
        matched_2d_old_norm.push_back(pt_norm);
    }

    vector<cv::Point3f> matched_3d = point_3d;

//    cout<<"before reduce"<<matched_3d.size()<<" "<<matched_2d_old_norm.size()<<endl;
    reduceVector(matched_2d_old, status);
    reduceVector(matched_2d_cur, status);
    reduceVector(matched_2d_old_norm, status);
    reduceVector(matched_3d, status);

    Eigen::Vector3d PnP_T_old;
    Eigen::Matrix3d PnP_R_old;

    status.clear();
    if(matched_2d_old_norm.size()>4 && matched_3d.size()>4)
        PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old);

    Eigen::Vector3d relative_t;
    Quaterniond relative_q;
    double relative_yaw;

    relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
    relative_q = PnP_R_old.transpose() * origin_vio_R;
    relative_yaw = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(PnP_R_old).x());
    //printf("PNP relative\n");
    cout << "pnp relative_t " << relative_t.norm() << endl;
    cout << "pnp relative_yaw " << relative_yaw << endl;


    reduceVector(matched_2d_cur, status);
//    cout<<"after reduce vec matched size  = "<<(int)matched_2d_cur.size()<<endl;
    reduceVector(matched_2d_old, status);
    reduceVector(matched_2d_old_norm, status);
    reduceVector(matched_3d, status);



    cout<<"-------------------in debug--------------------"<<endl;

}

void KeyFrame::light_glue_matcher(vector<cv::Point2f> &matched_2d_old, vector<cv::Point2f> &matched_2d_old_norm, vector<uchar> &status,
                                  std::vector<float> &local_descriptors_old, vector<float> &scores_old,
                                  const vector<cv::KeyPoint> &keypoints_old,
                                  const vector<cv::KeyPoint> &keypoints_old_norm,
                                  const int height_old, const int width_old){

    static auto deep_net_estimator_lg = Estimator_net::creat_estimator(lg_engine_path,lg_engine_path,0);

    deep_net_estimator_lg->lg_matches.clear();
    deep_net_estimator_lg->lg_scores.clear();

    deep_net_estimator_lg->lg_mkpts0.clear();
    deep_net_estimator_lg->lg_mkpts1.clear();

    cout<<"--------------------"<<endl;
    vector<cv::Point2f> _keypoints, _keypoints_old;

    float shift_width = 376;
    float shift_height = 240;
    float scale = 376;

    for(auto & i : window_keypoints)
    {
//        cv::Point2f key;
//        key.x = (i.pt.x - shift_width) / scale ;
//        key.y = (i.pt.y - shift_height) / scale ;

        //不需要归一化，统一在lg中进行
        _keypoints.push_back(i.pt);
    }

    for(auto & i : keypoints_old)
    {
        _keypoints_old.push_back(i.pt);
    }
    //kpts0,kpts1,des0,des1
    deep_net_estimator_lg->lg_matcher(_keypoints, _keypoints_old, window_local_descriptors, local_descriptors_old,480,752,480,752);

    cout<<"lg_mkpts0.size() = "<<deep_net_estimator_lg->lg_mkpts0.size()<<endl;

    vector<int> match_index0(deep_net_estimator_lg->lg_mkpts0.size());
    vector<int> match_index1(deep_net_estimator_lg->lg_mkpts1.size());
//    vector<int> match_index(_keypoints.size(),-1);
//    vector<float> match_score;

    //获取匹配对 yuanbende
    for(int i =0; i<deep_net_estimator_lg->lg_mkpts0.size(); i++)
    {
        match_index0[i] = deep_net_estimator_lg->lg_matches[2*i];

        match_index1[i] = deep_net_estimator_lg->lg_matches[2*i+1];
    }
//    for(int i =0;i<deep_net_estimator_lg->lg_mkpts0.size();i++)
//    {
//        match_index[deep_net_estimator_lg->lg_matches[2*i]] = deep_net_estimator_lg->lg_matches[2*i];
//    }


    status = std::vector<uchar>(_keypoints.size(),0);
//    status = std::vector<uchar>(_keypoints.size());

    for (int i  = 0;i< match_index0.size();i++){
        cv::Point2f pt(0.f, 0.f);
        cv::Point2f pt_norm(0.f, 0.f);

        status[match_index0[i]] = 1;

        pt = _keypoints_old[match_index1[i]];
        pt_norm = keypoints_old_norm[match_index1[i]].pt;

        matched_2d_old.push_back(pt);
        matched_2d_old_norm.push_back(pt_norm);

    }

//    for(int i : match_index)
//    {
//        cv::Point2f pt(0.f, 0.f);
//        cv::Point2f pt_norm(0.f, 0.f);
//        if (i >= 0){
//            status.push_back(1);
//            pt = _keypoints_old[i];
//            pt_norm = keypoints_old_norm[i].pt;
//        }
//        else
//            status.push_back(0);
//        matched_2d_old.push_back(pt);
//        matched_2d_old_norm.push_back(pt_norm);
//    }

//    for(auto &i:status )
//        cout<<(int)i<<endl;

    //这两个是相等的
    cout<<"matched_2d_old.size() = "<<matched_2d_old.size()<<endl;
//    cout<<"matched_2d_old_norm.size() = "<<matched_2d_old_norm.size()<<endl;

}

void KeyFrame::computeWindowBRIEFPoint()
{
    BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());

    for(int i = 0; i < (int)point_2d_uv.size(); i++)
    {
        cv::KeyPoint key;
        key.pt = point_2d_uv[i];
        window_keypoints.push_back(key);
    }
    extractor(image, window_keypoints, window_brief_descriptors);
}




void KeyFrame::computeBRIEFPoint()
{
    BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
    const int fast_th = 20; // corner detector response threshold
    if(1)
        cv::FAST(image, keypoints, fast_th, true);
    else
    {
        vector<cv::Point2f> tmp_pts;
        cv::goodFeaturesToTrack(image, tmp_pts, 500, 0.01, 10);
        for(int i = 0; i < (int)tmp_pts.size(); i++)
        {
            cv::KeyPoint key;
            key.pt = tmp_pts[i];
            keypoints.push_back(key);
        }
    }

    extractor(image, keypoints, brief_descriptors);
    for (int i = 0; i < (int)keypoints.size(); i++)
    {
        Eigen::Vector3d tmp_p;
        m_camera->liftProjective(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
        cv::KeyPoint tmp_norm;
        tmp_norm.pt = cv::Point2f(tmp_p.x()/tmp_p.z(), tmp_p.y()/tmp_p.z());
        keypoints_norm.push_back(tmp_norm);
    }
}

void BriefExtractor::operator() (const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const
{
    m_brief.compute(im, keys, descriptors);
}


bool KeyFrame::searchInAera(const BRIEF::bitset window_descriptor,
                            const std::vector<BRIEF::bitset> &descriptors_old,
                            const std::vector<cv::KeyPoint> &keypoints_old,
                            const std::vector<cv::KeyPoint> &keypoints_old_norm,
                            cv::Point2f &best_match,
                            cv::Point2f &best_match_norm)
{
    cv::Point2f best_pt;
    int bestDist = 128;
    int bestIndex = -1;
    for(int i = 0; i < (int)descriptors_old.size(); i++)
    {

        int dis = HammingDis(window_descriptor, descriptors_old[i]);
        if(dis < bestDist)
        {
            bestDist = dis;
            bestIndex = i;
        }
    }
    //printf("best dist %d", bestDist);
    if (bestIndex != -1 && bestDist < 80)
    {
        best_match = keypoints_old[bestIndex].pt;
        best_match_norm = keypoints_old_norm[bestIndex].pt;
        return true;
    }
    else
        return false;
}

void KeyFrame::searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
                                std::vector<cv::Point2f> &matched_2d_old_norm,
                                std::vector<uchar> &status,
                                const std::vector<BRIEF::bitset> &descriptors_old,
                                const std::vector<cv::KeyPoint> &keypoints_old,
                                const std::vector<cv::KeyPoint> &keypoints_old_norm)
{
    for(int i = 0; i < (int)window_brief_descriptors.size(); i++)
    {
        cv::Point2f pt(0.f, 0.f);
        cv::Point2f pt_norm(0.f, 0.f);
        if (searchInAera(window_brief_descriptors[i], descriptors_old, keypoints_old, keypoints_old_norm, pt, pt_norm))
            status.push_back(1);
        else
            status.push_back(0);
        matched_2d_old.push_back(pt);
        matched_2d_old_norm.push_back(pt_norm);
    }

}


void KeyFrame::FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                      const std::vector<cv::Point2f> &matched_2d_old_norm,
                                      vector<uchar> &status)
{
    int n = (int)matched_2d_cur_norm.size();
    for (int i = 0; i < n; i++)
        status.push_back(0);
    if (n >= 8)
    {
        vector<cv::Point2f> tmp_cur(n), tmp_old(n);
        for (int i = 0; i < (int)matched_2d_cur_norm.size(); i++)
        {
            double FOCAL_LENGTH = 460.0;
            double tmp_x, tmp_y;
            tmp_x = FOCAL_LENGTH * matched_2d_cur_norm[i].x + COL / 2.0;
            tmp_y = FOCAL_LENGTH * matched_2d_cur_norm[i].y + ROW / 2.0;
            tmp_cur[i] = cv::Point2f(tmp_x, tmp_y);

            tmp_x = FOCAL_LENGTH * matched_2d_old_norm[i].x + COL / 2.0;
            tmp_y = FOCAL_LENGTH * matched_2d_old_norm[i].y + ROW / 2.0;
            tmp_old[i] = cv::Point2f(tmp_x, tmp_y);
        }
        cv::findFundamentalMat(tmp_cur, tmp_old, cv::FM_RANSAC, 3.0, 0.9, status);
    }
}

void KeyFrame::PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                         const std::vector<cv::Point3f> &matched_3d,
                         std::vector<uchar> &status,
                         Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old)
{
    //for (int i = 0; i < matched_3d.size(); i++)
    //	printf("3d x: %f, y: %f, z: %f\n",matched_3d[i].x, matched_3d[i].y, matched_3d[i].z );
    //printf("match size %d \n", matched_3d.size());
    cv::Mat r, rvec, t, D, tmp_r;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
    Matrix3d R_inital;
    Vector3d P_inital;
    Matrix3d R_w_c = origin_vio_R * qic;
    Vector3d T_w_c = origin_vio_T + origin_vio_R * tic;

    R_inital = R_w_c.inverse();
    P_inital = -(R_inital * T_w_c);

    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);

    cv::Mat inliers;
    TicToc t_pnp_ransac;

    int flags = cv::SOLVEPNP_EPNP;
//    cout<<"matched_3d num ="<<matched_3d.size()<<" matched_2d_old_norm = "<<matched_2d_old_norm.size()<<endl;
//    for(int i =0; i <4; i++)
//        cout<<matched_3d[i]<<matched_2d_old_norm[i]<<endl;

    solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 200, PNP_INFLATION/460.0, 0.99, inliers,flags);

//    if (CV_MAJOR_VERSION < 3)
//        solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 100, inliers);
//    else
//    {
//        if (CV_MINOR_VERSION < 2)
//            solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, sqrt(10.0 / 460.0), 0.99, inliers);
//        else
//            solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0/460.0, 0.99, inliers);
//
//    }

    for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
        status.push_back(0);

    for( int i = 0; i < inliers.rows; i++)
    {
        int n = inliers.at<int>(i);
        status[n] = 1;
    }

    cv::Rodrigues(rvec, r);
    Matrix3d R_pnp, R_w_c_old;
    cv::cv2eigen(r, R_pnp);
    R_w_c_old = R_pnp.transpose();
    Vector3d T_pnp, T_w_c_old;
    cv::cv2eigen(t, T_pnp);
    T_w_c_old = R_w_c_old * (-T_pnp);

    PnP_R_old = R_w_c_old * qic.transpose();
    PnP_T_old = T_w_c_old - PnP_R_old * tic;

}


bool KeyFrame::findConnection(KeyFrame* old_kf)
{
    DEBUG_IMAGE = 1;
    TicToc tmp_t;
    //printf("find Connection\n");
    vector<cv::Point2f> matched_2d_cur, matched_2d_old;
    vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
    vector<cv::Point3f> matched_3d;
    vector<double> matched_id;
    vector<uchar> status;

    // re-undistort with the latest intrinsic values
    for (int i = 0; i < (int)point_2d_uv.size(); i++) {
        Eigen::Vector3d tmp_p;
        m_camera->liftProjective(Eigen::Vector2d(point_2d_uv[i].x, point_2d_uv[i].y), tmp_p);
        point_2d_norm.push_back(cv::Point2f(tmp_p.x()/tmp_p.z(), tmp_p.y()/tmp_p.z()));
    }
    old_kf->keypoints_norm.clear();
    for (int i = 0; i < (int)old_kf->keypoints.size(); i++) {
        Eigen::Vector3d tmp_p;
        m_camera->liftProjective(Eigen::Vector2d(old_kf->keypoints[i].pt.x, old_kf->keypoints[i].pt.y), tmp_p);
        cv::KeyPoint tmp_norm;
        tmp_norm.pt = cv::Point2f(tmp_p.x()/tmp_p.z(), tmp_p.y()/tmp_p.z());
        old_kf->keypoints_norm.push_back(tmp_norm);
    }

    matched_3d = point_3d;
    matched_2d_cur = point_2d_uv;
    matched_2d_cur_norm = point_2d_norm;
    matched_id = point_id;

    TicToc t_match;
#if 0
    if (DEBUG_IMAGE)
	    {
	        cv::Mat gray_img, loop_match_img;
	        cv::Mat old_img = old_kf->image;
	        cv::hconcat(image, old_img, gray_img);
	        cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)point_2d_uv.size(); i++)
	        {
	            cv::Point2f cur_pt = point_2d_uv[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)old_kf->keypoints.size(); i++)
	        {
	            cv::Point2f old_pt = old_kf->keypoints[i].pt;
	            old_pt.x += COL;
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        ostringstream path;
	        path << "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "0raw_point.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
#endif
    //printf("search by des\n");
//    searchByBRIEFDes(matched_2d_old, matched_2d_old_norm, status, old_kf->brief_descriptors, old_kf->keypoints, old_kf->keypoints_norm);
    cout<<"loop index =  "<< old_kf->index <<" keypoints.size ="<<old_kf->keypoints.size()<<" "
        <<" now index = "<<index<<endl;

    if( point_2d_uv.size()>20 && keypoints.size()>20 && old_kf->keypoints.size() > 20)
    {
         light_glue_matcher(matched_2d_old, matched_2d_old_norm, status, old_kf->local_descriptors, old_kf->scores, old_kf->keypoints, old_kf->keypoints_norm, old_kf->image.rows, old_kf->image.cols);
    }

    if(!status.empty())
    {
        reduceVector(matched_2d_cur, status);
        //        reduceVector(matched_2d_old, status);
        //        reduceVector(matched_2d_cur_norm, status);
        //        reduceVector(matched_2d_old_norm, status);
        reduceVector(matched_3d, status);
        reduceVector(matched_id, status);
    }

    //printf("search by des finish\n");
#if 0
    if (DEBUG_IMAGE)
	    {
			int gap = 10;
        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap);
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }

	        ostringstream path, path1, path2;
	        path <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	        /*
	        path1 <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match_1.jpg";
	        cv::imwrite( path1.str().c_str(), image);
	        path2 <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match_2.jpg";
	        cv::imwrite( path2.str().c_str(), old_img);	        
	        */
	        
	    }
#endif
    status.clear();
    /*
    FundmantalMatrixRANSAC(matched_2d_cur_norm, matched_2d_old_norm, status);
    reduceVector(matched_2d_cur, status);
    reduceVector(matched_2d_old, status);
    reduceVector(matched_2d_cur_norm, status);
    reduceVector(matched_2d_old_norm, status);
    reduceVector(matched_3d, status);
    reduceVector(matched_id, status);
    */
#if 0
    if (DEBUG_IMAGE)
	    {
			int gap = 10;
        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap) ;
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }

	        ostringstream path;
	        path <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "2fundamental_match.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
#endif
    Eigen::Vector3d PnP_T_old;
    Eigen::Matrix3d PnP_R_old;
    Eigen::Vector3d relative_t;
    Quaterniond relative_q;
    double relative_yaw;
//    cout<<"matched_2d_cur_size = "<<(int)matched_2d_cur.size()<<endl;
//    cout<<"matched_3d_size = "<<(int)matched_3d.size()<<endl;
//    cout<<"matched_2d_old_norm .size = "<<(int)matched_2d_old_norm.size()<<endl;
    if ((int)matched_2d_cur.size() > MIN_LOOP_NUM && (matched_2d_old_norm.size() == matched_3d.size()))
    {
        status.clear();
        PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old);
//        cout<<"after pnp matched_2d_cur_size = "<<(int)matched_2d_cur.size()<<endl;
        reduceVector(matched_2d_cur, status);
        cout<<"after reduce vec matched size  = "<<(int)matched_2d_cur.size()<<endl;
        reduceVector(matched_2d_old, status);
//        reduceVector(matched_2d_cur_norm, status);
        reduceVector(matched_2d_old_norm, status);
        reduceVector(matched_3d, status);
        reduceVector(matched_id, status);

#if 1
        if (DEBUG_IMAGE)
        {
            int gap = 10;
            cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
            for(int i = 0; i< (int)matched_2d_cur.size(); i++)
            {
                cv::Point2f cur_pt = matched_2d_cur[i];
                cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
            }
            for(int i = 0; i< (int)matched_2d_old.size(); i++)
            {
                cv::Point2f old_pt = matched_2d_old[i];
                old_pt.x += (COL + gap);
                cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
            }
            for (int i = 0; i< (int)matched_2d_cur.size(); i++)
            {
                cv::Point2f old_pt = matched_2d_old[i];
                old_pt.x += (COL + gap) ;
                cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 2, 8, 0);
            }
            cv::Mat notation(50, COL + gap + COL, CV_8UC3, cv::Scalar(255, 255, 255));
            putText(notation, "current frame: " + to_string(index) + "  sequence: " + to_string(sequence), cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);

            putText(notation, "previous frame: " + to_string(old_kf->index) + "  sequence: " + to_string(old_kf->sequence), cv::Point2f(20 + COL + gap, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
            cv::vconcat(notation, loop_match_img, loop_match_img);

            /*
            ostringstream path;
            path <<  "/home/tony-ws1/raw_data/loop_image/"
                    << index << "-"
                    << old_kf->index << "-" << "3pnp_match.jpg";
            cv::imwrite( path.str().c_str(), loop_match_img);
            */
            if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
            {
                /*
                cv::imshow("loop connection",loop_match_img);
                cv::waitKey(10);
                */
                cv::Mat thumbimage;
                cv::resize(loop_match_img, thumbimage, cv::Size(loop_match_img.cols / 2, loop_match_img.rows / 2));
                sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", thumbimage).toImageMsg();
                msg->header.stamp = ros::Time(time_stamp);
                pub_match_img.publish(msg);
            }
        }
#endif
    }
//    cout<<"matched_2d_cur_size = "<<(int)matched_2d_cur.size()<<endl;
    if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
    {
        relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
        relative_q = PnP_R_old.transpose() * origin_vio_R;
        relative_yaw = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(PnP_R_old).x());
        //printf("PNP relative\n");
        cout << "pnp relative_t " << relative_t.norm() << endl;
        cout << "pnp relative_yaw " << abs(relative_yaw) << endl;


        if (abs(relative_yaw) < MAX_THETA_DIFF && relative_t.norm() < MAX_POSE_DIFF)
        {
            cout<<"has loop!!!!!!!!!!!!!!!!!!!!"<<endl;
            has_loop = true;
            loop_index = old_kf->index;
            loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
                    relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
                    relative_yaw;
           return true;
        }
    }
    //printf("loop final use num %d %lf--------------- \n", (int)matched_2d_cur.size(), t_match.toc());
    return false;
}


int KeyFrame::HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b)
{
    BRIEF::bitset xor_of_bitset = a ^ b;
    int dis = xor_of_bitset.count();
    return dis;
}

void KeyFrame::getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = vio_T_w_i;
    _R_w_i = vio_R_w_i;
}

void KeyFrame::getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = T_w_i;
    _R_w_i = R_w_i;
}

void KeyFrame::updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
    T_w_i = _T_w_i;
    R_w_i = _R_w_i;
}

void KeyFrame::updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
    vio_T_w_i = _T_w_i;
    vio_R_w_i = _R_w_i;
    T_w_i = vio_T_w_i;
    R_w_i = vio_R_w_i;
}

Eigen::Vector3d KeyFrame::getLoopRelativeT()
{
    return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2));
}

Eigen::Quaterniond KeyFrame::getLoopRelativeQ()
{
    return Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5), loop_info(6));
}

double KeyFrame::getLoopRelativeYaw()
{
    return loop_info(7);
}

void KeyFrame::updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info)
{
    if (abs(_loop_info(7)) < 30.0 && Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0)
    {
        //printf("update loop info\n");
        loop_info = _loop_info;
    }
}




BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
    // The DVision::BRIEF extractor computes a random pattern by default when
    // the object is created.
    // We load the pattern that we used to build the vocabulary, to make
    // the descriptors compatible with the predefined vocabulary

    // loads the pattern
    cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
    if(!fs.isOpened()) throw string("Could not open file ") + pattern_file;

    vector<int> x1, y1, x2, y2;
    fs["x1"] >> x1;
    fs["x2"] >> x2;
    fs["y1"] >> y1;
    fs["y2"] >> y2;

    m_brief.importPairs(x1, y1, x2, y2);
}


