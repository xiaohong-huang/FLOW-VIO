#include <csignal>
#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "swf/swf.h"
#include "parameter/parameters.h"
#include "utility/visualization.h"
#include "utility/utility.h"
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <std_msgs/ByteMultiArray.h>
#include <sensor_msgs/MagneticField.h>
#include <sensor_msgs/NavSatFix.h>
#include<random>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>

SWFOptimization* swf_optimization;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
double start_timestamp = 0;//1896.28
TicToc system_time;
double last_system_time;



cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr& img_msg) {
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1") {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    } else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
    cv::Mat img = ptr->image.clone();
    return img;
}


void processoneimage() {
    cv::Mat image0, image1;
    double time = 0;

    if (!img0_buf.empty() ) {
        time = img0_buf.front()->header.stamp.toSec();
        image0 = getImageFromMsg(img0_buf.front());
        img0_buf.pop();
    }

    if (!image0.empty())
        swf_optimization->InputImage(time, image0, image1);
}


void img0_callback(const sensor_msgs::ImageConstPtr& img_msg) {
    if (img_msg->header.stamp.toSec() < start_timestamp)
        return;
    img0_buf.push(img_msg);
    processoneimage();
}


void imu_callback(const sensor_msgs::ImuConstPtr& imu_msg) {

    if (imu_msg->header.stamp.toSec() < start_timestamp - 1)
        return;
    double t = imu_msg->header.stamp.toSec();
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);
    acc = acc;
    gyr = gyr;
    swf_optimization->InputIMU(t, acc, gyr);
    return;
}





void bind_cpu(std::vector<int>cpu_set) {

}



void sig_handler( int sig ) {
    std::cout << "\tabort_\r\n";
    exit( 0 );
}





int main(int argc, char** argv) {


    string config_file = argv[1];
    ROS_PATH = argv[2];
    RESULT_PATH = argv[3];
    printf("config_file: %s\n", argv[1]);
    readParameters(config_file);

    ros::init(argc, argv, "flow_vio");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    registerPub(n);

    LEAK_NUM = SWF_SIZE_IN - 1;


#if DEBUG
    time_t t = time(nullptr);
    struct tm* now = localtime(&t);
    std::stringstream timeStr;
    timeStr << now->tm_year + 1900 << "-";
    timeStr << now->tm_mon + 1 << "-";
    timeStr << now->tm_mday << " ";
    timeStr << now->tm_hour << ":";
    timeStr << now->tm_min << ":";
    timeStr << now->tm_sec << ".log";
    std::string LOG_PATH = "log/" + RESULT_PATH + "--" + timeStr.str();
    LOG_OUT = std::ofstream(LOG_PATH, std::ios::out);
    LOG_OUT.precision(10);
#endif


    swf_optimization = new SWFOptimization();
    swf_optimization->SetParameter();
    printf("waiting for image and imu...");
    rosbag::Bag bag;
    bag.open(ROS_PATH, rosbag::bagmode::Read);
    rosbag::View view(bag);

    signal( SIGINT, sig_handler );
    for ( rosbag::View::iterator it = view.begin(); it != view.end(); ++it) {
        auto m = *it;
        if (m.getTopic() == IMU_TOPIC)
            imu_callback(m.instantiate<sensor_msgs::Imu>());
        else if (m.getTopic() == IMAGE0_TOPIC)
            img0_callback(m.instantiate<sensor_msgs::Image>());
        processoneimage();
    }
    while (swf_optimization->feature_buf.size() > 10);


    //As same as ORB-SLAM3, we generate the final resutls for evaluation.
    std::vector<PosInfo> camera_poses = swf_optimization->RetriveAllPose();
    ofstream foutC(RESULT_PATH, ios::out);
    foutC.setf(ios::fixed, ios::floatfield);

    Eigen::Matrix3d R_WI_WC = swf_optimization->R_WI_WC;
    double scale_factor = swf_optimization->scale_factor;
    static double old_time = 0;
    for (int i = 0; i < (int)camera_poses.size(); i++) {
        Eigen::Matrix3d Ri = R_WI_WC * camera_poses[i].R * RIC[0].transpose();
        Eigen::Vector3d ti = R_WI_WC * camera_poses[i].t * scale_factor - Ri * TIC[0] - Ri * Pgb;
        Eigen::Quaterniond Qi = Eigen::Quaterniond(Ri);
        double time_stamp = camera_poses[i].time_stamp;
        if (old_time)assert(time_stamp > old_time);

        foutC.precision(9);
        foutC << time_stamp << " ";
        foutC.precision(5);
        foutC << ti.x() << " "
              << ti.y() << " "
              << ti.z() << " "
              << Qi.x() << " "
              << Qi.y() << " "
              << Qi.z() << " "
              << Qi.w() << endl;
        old_time = time_stamp;

    }
    foutC.close();

    std::cout << "finish\n";
    std::cout << "finish\n";
    return 0;
}




