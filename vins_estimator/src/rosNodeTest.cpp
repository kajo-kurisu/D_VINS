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

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "utility/visualization.h"

//USE_BAG_LOAD
#include <rosbag/bag.h>
#include <rosbag/view.h>

#define USE_BAG_LOAD 1

Estimator estimator;

queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
std::mutex m_buf;
double time_last;

double last_image_time_r = -1;
double last_image_time_l = -1;


rosbag::Bag bag;
rosbag::View view_full;
rosbag::View view;
std::vector<std::string> topics;



void new_sequence()
{
    printf("new sequence\n");
    m_buf.lock();
    while(!img0_buf.empty())
        img0_buf.pop();
    while(!img1_buf.empty())
        img1_buf.pop();

    estimator.clearState();
    estimator.setParameter();

    m_buf.unlock();
}

void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img0_buf.push(img_msg);
    m_buf.unlock();

    if (last_image_time_l == -1)
        last_image_time_l = img_msg->header.stamp.toSec();
    else if (img_msg->header.stamp.toSec() - last_image_time_l > 1.0 || img_msg->header.stamp.toSec() < last_image_time_l)
    {
        ROS_WARN("image discontinue! detect a new sequence!");
        new_sequence();
    }
    last_image_time_l = img_msg->header.stamp.toSec();

}

void img1_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img1_buf.push(img_msg);
    m_buf.unlock();

    if (last_image_time_r == -1)
        last_image_time_r = img_msg->header.stamp.toSec();
    else if (img_msg->header.stamp.toSec() - last_image_time_r > 1.0 || img_msg->header.stamp.toSec() < last_image_time_r)
    {
        ROS_WARN("image discontinue! detect a new sequence!");
        new_sequence();
    }
    last_image_time_r = img_msg->header.stamp.toSec();
}


cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();

    return img;
}

// extract images with same timestamp from two topics
void sync_process()
{
    while(1)
    {
        if(STEREO)
        {
            cv::Mat image0, image1;
            std_msgs::Header header;
            double time = 0;
            m_buf.lock();
            if (!img0_buf.empty() && !img1_buf.empty())
            {
                double time0 = img0_buf.front()->header.stamp.toSec();
                double time1 = img1_buf.front()->header.stamp.toSec();
                // 0.003s sync tolerance
                if(time0 < time1 - 0.003)
                {
                    img0_buf.pop();
                    printf("throw img0\n");
                }
                else if(time0 > time1 + 0.003)
                {
                    img1_buf.pop();
                    printf("throw img1\n");
                }
                else
                {
                    time = img0_buf.front()->header.stamp.toSec();
                    header = img0_buf.front()->header;
                    image0 = getImageFromMsg(img0_buf.front());
                    img0_buf.pop();
                    image1 = getImageFromMsg(img1_buf.front());
                    img1_buf.pop();


                    //printf("find img0 and img1\n");
                }
            }
            m_buf.unlock();
            if(!image0.empty())
                estimator.inputImage(time, image0, image1);
        }
        else
        {
            cv::Mat image;
            std_msgs::Header header;
            double time = 0;
            m_buf.lock();
            if(!img0_buf.empty())
            {
                time = img0_buf.front()->header.stamp.toSec();
                header = img0_buf.front()->header;
                image = getImageFromMsg(img0_buf.front());
                img0_buf.pop();
            }
            m_buf.unlock();

//            if(!image.empty())
//                estimator.inputImage(time, image);
            if(!image.empty()){
                if(time-time_last>1){
                    ROS_INFO("time jump detected, restart estimation!");
                    estimator.clearState();
                    estimator.setParameter();
                    time_last = 10000000000000000;
                }
                else{
                    //std::cout<<"time:"<<time<<std::endl;
                    time_last = time;
                    estimator.inputImage(time, image);
                }
            }

        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}


void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);
    estimator.inputIMU(t, acc, gyr);
    return;
}

//fast_bag_play
void imu_process() {
    while (!imu_buf.empty()) {
        imu_callback(imu_buf.front());
        imu_buf.pop();
    }
}


void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (unsigned int i = 0; i < feature_msg->points.size(); i++)
    {
        int feature_id = feature_msg->channels[0].values[i];
        int camera_id = feature_msg->channels[1].values[i];
        double x = feature_msg->points[i].x;
        double y = feature_msg->points[i].y;
        double z = feature_msg->points[i].z;
        double p_u = feature_msg->channels[2].values[i];
        double p_v = feature_msg->channels[3].values[i];
        double velocity_x = feature_msg->channels[4].values[i];
        double velocity_y = feature_msg->channels[5].values[i];
        if(feature_msg->channels.size() > 5)
        {
            double gx = feature_msg->channels[6].values[i];
            double gy = feature_msg->channels[7].values[i];
            double gz = feature_msg->channels[8].values[i];
            pts_gt[feature_id] = Eigen::Vector3d(gx, gy, gz);
            //printf("receive pts gt %d %f %f %f\n", feature_id, gx, gy, gz);
        }
        ROS_ASSERT(z == 1);
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }
    double t = feature_msg->header.stamp.toSec();
    estimator.inputFeature(t, featureFrame);
    return;
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        estimator.clearState();
        estimator.setParameter();
    }
    return;
}

void imu_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true)
    {
        //ROS_WARN("use IMU!");
        estimator.changeSensorType(1, STEREO);
    }
    else
    {
        //ROS_WARN("disable IMU!");
        estimator.changeSensorType(0, STEREO);
    }
    return;
}

void cam_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true)
    {
        //ROS_WARN("use stereo!");
        estimator.changeSensorType(USE_IMU, 1);
    }
    else
    {
        //ROS_WARN("use mono camera (left)!");
        estimator.changeSensorType(USE_IMU, 0);
    }
    return;
}

int main(int argc, char **argv)
{
    time_last = 10000000000000000;
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
//    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

//    if(argc != 2)
//    {
//        printf("please intput: rosrun vins vins_node [config file] \n"
//               "for example: rosrun vins vins_node "
//               "~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml \n");
//        return 1;
//    }

//    n.setParam("config_file",);
//    string config_file;
    string config_file = argv[1];


//    n.param<std::string>("config_file",config_file,"/home/sy/sy/vins_fusion_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml");
//    n.param<std::string>("config_file",config_file,"/home/sy/sy/vins_fusion_ws/src/VINS-Fusion/config/4seasons/4seasons_stereo_imu_config.yaml");


    readParameters(config_file);
    estimator.setParameter();

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    ros::Subscriber sub_imu;

//#define FAST_BAG
#ifdef FAST_BAG
    bag.open(BAG_PATH, rosbag::bagmode::Read);
    view_full.addQuery(bag);
    ros::Time time_st = view_full.getBeginTime();
    ros::Time time_end = view_full.getEndTime();

    topics.push_back(IMAGE0_TOPIC);
    topics.push_back(IMAGE1_TOPIC);
    topics.push_back(IMU_TOPIC);

    view.addQuery(bag, rosbag::TopicQuery(topics), time_st, time_end);
    for (rosbag::View::iterator it = view.begin(); it != view.end(); ++it) {
        auto data_tmp = *it;
        if (data_tmp.getTopic() == IMAGE0_TOPIC) {
            img0_buf.emplace(data_tmp.instantiate<sensor_msgs::Image>());
        }
        if (data_tmp.getTopic() == IMAGE1_TOPIC) {
            img1_buf.emplace(data_tmp.instantiate<sensor_msgs::Image>());
        }
        if (data_tmp.getTopic() == IMU_TOPIC) {
            imu_buf.emplace(data_tmp.instantiate<sensor_msgs::Imu>());
        }
    }

    ROS_INFO("time start = %.6f", time_st.toSec());
    ROS_INFO("time end   = %.6f", time_end.toSec());
    ROS_INFO("Image0_Size = %lu", img0_buf.size());
    ROS_INFO("Image1_Size = %lu", img1_buf.size());
    ROS_INFO("Imu_Size = %lu", imu_buf.size());
    std::thread imu(imu_process);
    usleep(5000000);

#else
    if(USE_IMU)
    {
        sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    }

    ros::Subscriber sub_feature = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_img0 = n.subscribe(IMAGE0_TOPIC, 100, img0_callback);
    ros::Subscriber sub_img1;
    if(STEREO)
    {
        sub_img1 = n.subscribe(IMAGE1_TOPIC, 100, img1_callback);
    }
    ros::Subscriber sub_restart = n.subscribe("/vins_restart", 100, restart_callback);
    ros::Subscriber sub_imu_switch = n.subscribe("/vins_imu_switch", 100, imu_switch_callback);
    ros::Subscriber sub_cam_switch = n.subscribe("/vins_cam_switch", 100, cam_switch_callback);
#endif

    std::thread sync_thread{sync_process};
    ros::spin();
    return 0;
}
