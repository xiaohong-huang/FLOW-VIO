
#include "parameters.h"
#include <ros/ros.h>
#include "../utility/utility.h"
#include <fstream>
#include <map>

cv::Mat mask_global;
std::vector<std::unordered_map<int, double>> idepths_all;
bool fix_scale = false;
bool has_excitation = false;
double ACC_N, INT_N, ACC_W, GYR_NONLINEARITY;
double GYR_N, GYR_W;
int ESTIMATE_EXTRINSIC = 1;
double FOCAL_LENGTH;
double F_NOISE_SIGMA;
double SKIP_TIME = 1;
int OPTICAL_WINSIZE = 11;
int FRAME_NUM_FOR_REJECTION=50;
std::set<double*>idepth_pointers;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Quaterniond> QIC;
std::vector<Eigen::Vector3d> TIC;


Eigen::Vector3d G{0.0, 0.0, 9.8};
Eigen::Vector3d Pgb;

int MIN_DIST;
int SHOW_TRACK;
int FLOW_BACK = 0;

std::string RESULT_PATH;
std::string IMU_TOPIC;
std::string IMAGE0_TOPIC;
std::vector<std::string> CAM_NAMES;
std::string ROS_PATH;



std::ofstream LOG_OUT;

int LEAK_NUM;

omp_lock_t omp_lock;
std::string exposure_time_file;

bool have_hist = false;
bool enable_output = false;

void readParameters(std::string config_file) {

    omp_init_lock(&omp_lock);
    FILE* fh = fopen(config_file.c_str(), "r");
    if (fh == NULL) {
        printf("config_file dosen't exist; wrong config_file path");
        abort();
        return;
    }
    fclose(fh);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened())
        std::cerr << "ERROR: Wrong path to settings" << std::endl;

    //required parameters
    fsSettings["image0_topic"] >> IMAGE0_TOPIC;
    fsSettings["imu_topic"] >> IMU_TOPIC;
    fsSettings["FOCAL_LENGTH"] >> FOCAL_LENGTH;
    MIN_DIST = fsSettings["feature_min_dist"];
    SHOW_TRACK = fsSettings["SHOW_TRACK"];
    ACC_N = fsSettings["acc_n"];
    INT_N = fsSettings["int_n"];
    FLOW_BACK = fsSettings["FLOW_BACK"];
    GYR_NONLINEARITY = fsSettings["gyr_nonlinearity"];
    ACC_W = fsSettings["acc_w"];
    GYR_N = fsSettings["gyr_n"];
    GYR_W = fsSettings["gyr_w"];
    G.z() = fsSettings["g_norm"];

    cv::Mat cv_T;
    fsSettings["body_T_cam0"] >> cv_T;
    Eigen::Matrix4d T;
    cv::cv2eigen(cv_T, T);
    RIC.push_back(T.block<3, 3>(0, 0));
    TIC.push_back(T.block<3, 1>(0, 3));


    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);
    std::string cam0Calib;
    fsSettings["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    CAM_NAMES.push_back(cam0Path);


    for (int i = 0; i < (int)RIC.size(); i++)
        QIC.push_back(Eigen::Quaterniond(RIC[i]));


    //optional parameters
    cv::FileNode node;
    Pgb.setZero();

    if (!fsSettings["SKIP_TIME"].empty())SKIP_TIME = fsSettings["SKIP_TIME"];
    if (!fsSettings["FLOW_BACK"].empty())FLOW_BACK = fsSettings["FLOW_BACK"];
    if (!fsSettings["ESTIMATE_EXTRINSIC"].empty())ESTIMATE_EXTRINSIC = fsSettings["ESTIMATE_EXTRINSIC"];
    if (!fsSettings["FRAME_NUM_FOR_REJECTION"].empty())FRAME_NUM_FOR_REJECTION = fsSettings["FRAME_NUM_FOR_REJECTION"];
    if (!fsSettings["Pgb"].empty()) {
        cv::Mat cv_V;
        Eigen::Vector3d V;
        fsSettings["Pgb"] >> cv_V;
        cv::cv2eigen(cv_V, V);
        Pgb = V;
    }
    if (!fsSettings["MASK_NAME"].empty()) {
        std::string mask_name;
        fsSettings["MASK_NAME"] >> mask_name;
        mask_global = cv::imread(configPath + "/" + mask_name, cv::IMREAD_GRAYSCALE);
    }



    if (FLOW_BACK) {
        OPTICAL_WINSIZE = 17;//setting a larger searching window for flow back strategy.
    } else
        OPTICAL_WINSIZE = 11;

    F_NOISE_SIGMA = (std::max(OPTICAL_WINSIZE, 10) - 7) * 0.1;



    fsSettings.release();
}
int localSize(int size)  {
    return size == 7 ? 6 : size;
}

int globalSize(int size)  {
    return size == 6 ? 7 : size;
}
