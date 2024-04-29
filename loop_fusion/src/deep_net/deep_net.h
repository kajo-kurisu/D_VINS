//
// Created by sy on 23-12-4.
//

#ifndef VINS_FUSION_DEEP_NET_H
#define VINS_FUSION_DEEP_NET_H

#pragma once
#include "tensorrt_utils.h"

//sys
#include "iostream"
#include "vector"

#include "opencv2/opencv.hpp"

//eigen
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Geometry"

//ros
#include "ros/ros.h"
#include <ros/time.h>
#include "sensor_msgs/CompressedImage.h"

#include "../keyframe.h"

#include "tensorrt_tools/preprocess_kernel.cuh"


//4SEASONS
#define mix_engine_path "/home/sy/sy/Mix_ws/src/mixvpr/model/mix/mix_512.engine"
#define sp_engine_path "/home/sy/sy/Mix_ws/src/mixvpr/model/sim_800x400_512/superpoint_800x400_512.engine"
//#define lg_engine_path "/home/sy/sy/Mix_ws/src/mixvpr/model/sim_640x480_512/superpoint_lightglue_10_512.engine"
#define lg_engine_path "/home/sy/sy/Mix_ws/src/mixvpr/model/sim_640x480_512/superpoint_lightglue_10_1024.engine"
#define sp_re_engine_path "/home/sy/sy/Mix_ws/src/mixvpr/model/sp_re_800x400_512/superpoint_recover_des_800x400.engine"

//EUROC
//#define mix_engine_path "/home/sy/sy/Mix_ws/src/mixvpr/model/mix/mix_512.engine"
//#define sp_engine_path "/home/sy/sy/Mix_ws/src/mixvpr/model/sim_752x480_512/superpoint_752x480_512.engine"
////#define lg_engine_path "/home/sy/sy/Mix_ws/src/mixvpr/model/sim_640x480_512/superpoint_lightglue_10_512.engine"
//#define lg_engine_path "/home/sy/sy/Mix_ws/src/mixvpr/model/sim_640x480_512/superpoint_lightglue_10_1024.engine"
//#define sp_re_engine_path "/home/sy/sy/Mix_ws/src/mixvpr/model/sp_re_752x480_512/superpoint_recover_des_480x752.engine"


//kaist
//#define mix_engine_path "/home/sy/sy/Mix_ws/src/mixvpr/model/mix/mix_512.engine"
//#define sp_engine_path "/home/sy/sy/Mix_ws/src/mixvpr/model/sim_1280x560_512/superpoint_1280x560_512.engine"
//#define lg_engine_path "/home/sy/sy/Mix_ws/src/mixvpr/model/sim_640x480_512/superpoint_lightglue_10_1024.engine"
//#define sp_re_engine_path "/home/sy/sy/Mix_ws/src/mixvpr/model/sp_re_1280x560_512/superpoint_recover_des_560x1280.engine"


class frame{
public:
    frame()=default;

    std::string frame_id;
    ros::Time timestamp;

    float extract_time;

    int image_width;
    int image_height;

    cv::String filename;
    cv::Mat raw_image;

    //dim = 512
    std::vector<float> img_global_des_vec;

    std::vector<float> local_descriptor;
    std::vector<float> kpoints;
    std::vector<float> similarity;
    std::vector<int> top_k_ind;

    std::vector<std::pair<float,float>> landmarks;

};

class MixVPR{
public:

    TRTLogger Logger;


    explicit MixVPR(const std::string engine_path);
    explicit MixVPR();
    void inference(const cv::String& filename,
                   std::vector<frame> &frame_set,
                   std::vector<float> &des_db,
                   const char *engine_path
    );

    void test_in_datasets(const std::string filepath,
                          std::vector<cv::String > &namearray,
                          std::vector<frame> &frame_set,
                          std::map<int,frame> &frame_des_dataset,
                          std::vector<float> &des_db
    );

    void img_callback(const sensor_msgs::CompressedImage &msg,
                      frame &_frame,
                      std::vector<frame> &frame_set,
                      std::map<int,frame> &frame_des_dataset,
                      std::vector<float> &des_db
    );

    void sort_vec_faiss(std::vector<frame> &frame_set, std::vector<float> &des_db );

    void run(std::string &datapath);

};

namespace MixVPR_net{
    using namespace std;
    class MixVPR{
    public:

        MixVPR() = default;
        virtual void mix_extractor(const cv::Mat &img) = 0;
        virtual void test_in_dataset(const std::string filepath) = 0;
        virtual void sort_in_faiss(float* db,
                                   float* xq,int n) = 0;

        std::vector<float> descriptors_database;

        //mix_des dim = 512
        std::vector<float> mix_des;
        std::vector<int> top_sim_index;
        std::vector<float> top_sim;

        std::map<int,vector<int>> sim_map;

    };
    shared_ptr<MixVPR> creat_mix(const std::string & superpoint_engine_path, const int &engine_type, int gpuid = 0);
}

namespace Estimator_net {
    using namespace std;
    class Estimator{
    public:
        Estimator() = default;
        virtual void sp_extractor( const cv::Mat& img ) = 0;
        virtual void sp_extractor( const cv::Mat& img, vector<cv::Point2f> &sp_kpts ) = 0;
        virtual void lg_matcher(std::vector<cv::Point2f> &lg_in_kpts0,
                                std::vector<cv::Point2f> &lg_in_kpts1,
                                std::vector<float> &lg_in_desc0,
                                std::vector<float> &lg_in_desc1,
                                const int& height_0,const int& width_0,
                                const int& height_1,const int& width_1) = 0;

        virtual void lg_matcher() = 0;

        int width;
        int height;
        vector<float> sp_desc;
        vector<float> sp_scores;
        vector<cv::Point2f> sp_kpts_norm;
        vector<cv::Point2f> sp_kpts;

        vector<float> sp_re_desc;
        vector<float> sp_re_scores;

        vector<int> lg_matches;
        vector<float> lg_scores;
        vector<cv::Point2f> lg_mkpts0;
        vector<cv::Point2f> lg_mkpts1;
        cv::Mat image;

    };

    shared_ptr<Estimator> creat_estimator(const std::string & superpoint_engine_path,const std::string &lightglue_engine_path, int gpuid = 0);
    shared_ptr<Estimator> recover_estimator(const std::string & superpoint_engine_path, int gpuid = 0);
    shared_ptr<Estimator> single_init(const std::string & superpoint_engine_path, const int &engine_type, int gpuid = 0);

};


#endif //VINS_FUSION_DEEP_NET_H
