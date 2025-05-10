

#include "visualization.h"
#include "../parameter/parameters.h"
#include <fstream>
#include <std_msgs/Header.h>
#include <sensor_msgs/PointCloud.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include "camera_pose_visualization.h"

using namespace ros;
using namespace Eigen;

nav_msgs::Path path;

ros::Publisher pub_odometry, pub_latest_odometry;
ros::Publisher pub_path;
ros::Publisher pub_point_cloud_short, pub_margin_cloud, pub_point_cloud_long;
ros::Publisher pub_key_poses;
ros::Publisher pub_camera_pose;
ros::Publisher pub_camera_pose_right;
ros::Publisher pub_rectify_pose_left;
ros::Publisher pub_rectify_pose_right;
ros::Publisher pub_camera_pose_visual;
ros::Publisher pub_keyframe_pose;
ros::Publisher pub_keyframe_point;
ros::Publisher pub_extrinsic;

camera_pose_visualization cameraposevisual(1, 0, 0, 1);
Vector3d        Ps0;

#define PUB_INDEX swf_optimization.image_count - 1

void registerPub(ros::NodeHandle& n) {
    pub_latest_odometry = n.advertise<nav_msgs::Odometry>("imu_propagate", 1000);
    pub_path = n.advertise<nav_msgs::Path>("path", 1000);
    pub_odometry = n.advertise<nav_msgs::Odometry>("odometry", 1000);
    pub_point_cloud_short = n.advertise<sensor_msgs::PointCloud>("point_cloud", 1000);

    pub_margin_cloud = n.advertise<sensor_msgs::PointCloud>("margin_cloud", 1000);
    pub_camera_pose = n.advertise<nav_msgs::Odometry>("camera_pose", 1000);
    pub_camera_pose_right = n.advertise<nav_msgs::Odometry>("camera_pose_right", 1000);
    pub_rectify_pose_left = n.advertise<geometry_msgs::PoseStamped>("rectify_pose_left", 1000);
    pub_rectify_pose_right = n.advertise<geometry_msgs::PoseStamped>("rectify_pose_right", 1000);
    pub_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("camera_pose_visual", 1000);
    pub_keyframe_pose = n.advertise<nav_msgs::Odometry>("keyframe_pose", 1000);
    pub_keyframe_point = n.advertise<sensor_msgs::PointCloud>("keyframe_point", 1000);
    pub_extrinsic = n.advertise<nav_msgs::Odometry>("extrinsic", 1000);
}



void printStatistics(const SWFOptimization& swf_optimization, double t) {

    if (swf_optimization.solver_flag != SWFOptimization::SolverFlag::NonLinear)
        return;

    int index = PUB_INDEX;
    LOG_OUT <<  "time:" << swf_optimization.headers[index] << std::endl;
    LOG_OUT << "pos: " << (swf_optimization.Ps[index]).transpose() << " " << (swf_optimization.Ps[index]).transpose() << std::endl;
    LOG_OUT << "vel: " << (swf_optimization.Vs[index]).transpose() << std::endl;
    LOG_OUT << "orientation: " << Utility::R2ypr(swf_optimization.Rs[index]).transpose() << std::endl;
    LOG_OUT << "gyro bias: " << swf_optimization.Bgs[index].transpose() << std::endl;
    LOG_OUT << "acc bias: " << swf_optimization.Bas[index].transpose() << std::endl;
    if (ESTIMATE_TD) LOG_OUT << "td: " << swf_optimization.td << std::endl;
    if (ESTIMATE_EXTRINSIC) LOG_OUT << "extrinsic: " << TIC[0].transpose() << ",\t" << Utility::R2ypr(RIC[0]).transpose() << std::endl;
    if (ESTIMATE_GYR_SCALE || ESTIMATE_ACC_SCALE) LOG_OUT << "imu scale: " << swf_optimization.acc_scale.transpose() << ",\t" << swf_optimization.gyr_scale.transpose() << std::endl;
    LOG_OUT << "image_count:" << swf_optimization.image_count << std::endl;

}

void resetpot(const SWFOptimization& swf_optimization, const std_msgs::Header& header) {
    Ps0 = swf_optimization.Ps[0];
}

void pubOdometry(const SWFOptimization& swf_optimization, const std_msgs::Header& header) {
    std::vector<Vector3d>        Ps(swf_optimization.image_count);


    for (int i = 0; i < swf_optimization.image_count; i++)
        Ps[i] = swf_optimization.Ps[i] - Ps0;
    if (swf_optimization.solver_flag == SWFOptimization::SolverFlag::NonLinear) {
        nav_msgs::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.child_frame_id = "world";
        Quaterniond tmp_Q;
        tmp_Q = Quaterniond(swf_optimization.Rs[PUB_INDEX]);
        Vector3d tmpP = Ps[PUB_INDEX];

        Vector3d tmpv = swf_optimization.Vs[PUB_INDEX];

        odometry.pose.pose.position.x = tmpP.x();
        odometry.pose.pose.position.y = tmpP.y();
        odometry.pose.pose.position.z = tmpP.z();
        odometry.pose.pose.orientation.x = tmp_Q.x();
        odometry.pose.pose.orientation.y = tmp_Q.y();
        odometry.pose.pose.orientation.z = tmp_Q.z();
        odometry.pose.pose.orientation.w = tmp_Q.w();
        odometry.twist.twist.linear.x = tmpv.x();
        odometry.twist.twist.linear.y = tmpv.y();
        odometry.twist.twist.linear.z = tmpv.z();
        pub_odometry.publish(odometry);

        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = header;
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose = odometry.pose.pose;
        path.header = header;
        path.header.frame_id = "world";
        path.poses.push_back(pose_stamped);
        pub_path.publish(path);
        if (PUB_INDEX < 0)
            return;


        {
            nav_msgs::Odometry odometry;
            odometry.header = header;
            odometry.header.frame_id = "world";
            odometry.pose.pose.position.x = TIC[0].x();
            odometry.pose.pose.position.y = TIC[0].y();
            odometry.pose.pose.position.z = TIC[0].z();
            Quaterniond tmp_q{RIC[0]};
            odometry.pose.pose.orientation.x = tmp_q.x();
            odometry.pose.pose.orientation.y = tmp_q.y();
            odometry.pose.pose.orientation.z = tmp_q.z();
            odometry.pose.pose.orientation.w = tmp_q.w();
            pub_extrinsic.publish(odometry);
        }
    }
}


void pubCameraPose(const SWFOptimization& swf_optimization, const std_msgs::Header& header) {

    std::vector<Vector3d>        Ps(swf_optimization.image_count);


    for (int i = 0; i < swf_optimization.image_count; i++)
        Ps[i] = swf_optimization.Ps[i] - Ps0;

    int idx2 = PUB_INDEX;
    if (swf_optimization.solver_flag == SWFOptimization::SolverFlag::NonLinear) {
        int i = idx2;
        Vector3d P = (Ps[i] + swf_optimization.Rs[i] * TIC[0]);
        Quaterniond R = Quaterniond((swf_optimization.Rs[i] * RIC[0]));

        nav_msgs::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();

        pub_camera_pose.publish(odometry);

        cameraposevisual.reset();
        cameraposevisual.add_pose(P, R);
        cameraposevisual.publish_by(pub_camera_pose_visual, odometry.header);
    }
}


void pubPointCloud(const SWFOptimization& swf_optimization, const std_msgs::Header& header) {
    std::vector<Vector3d>        Ps(swf_optimization.image_count);


    for (int i = 0; i < swf_optimization.image_count; i++)
        Ps[i] = swf_optimization.Ps[i] - Ps0;

    sensor_msgs::PointCloud point_cloud;
    point_cloud.header = header;

    for (auto& it_per_id : swf_optimization.f_manager.feature) {
        if (!it_per_id.valid)continue;
        if (it_per_id.start_frame > SWF_SIZE_IN * (SWF_SIZE_OUT - 1))
            continue;
        Eigen::Vector3d ptsInW;

        bool is_cross = (it_per_id.start_frame / SWF_SIZE_IN != (/*it_per_id.endFrame()*/it_per_id.start_frame + (int)it_per_id.feature_per_frame.size() - 1/*it_per_id.endFrame()*/ - 1) / SWF_SIZE_IN) || it_per_id.start_frame == 0;
        if (is_cross) {
            int imu_i = it_per_id.start_frame;
            if  (it_per_id.start_frame % SWF_SIZE_IN != 0)
                imu_i = (it_per_id.start_frame / SWF_SIZE_IN + 1) * SWF_SIZE_IN;
            ASSERT(idepths_all[imu_i / SWF_SIZE_IN][it_per_id.feature_id] != 0);
            ptsInW = swf_optimization.Rs[imu_i] * (
                         RIC[0] * (it_per_id.feature_per_frame[imu_i - it_per_id.start_frame].point /
                                   idepths_all[imu_i / SWF_SIZE_IN][it_per_id.feature_id]) + TIC[0]
                     ) + swf_optimization.Ps[imu_i];
        } else {
            ptsInW = swf_optimization.Rs[it_per_id.start_frame] * (
                         RIC[0] * (it_per_id.feature_per_frame[0].point /
                                   idepths_all[it_per_id.start_frame / SWF_SIZE_IN][it_per_id.feature_id]) + TIC[0]
                     ) + swf_optimization.Ps[it_per_id.start_frame];
        }

        Vector3d w_pts_i = (ptsInW - Ps0);

        geometry_msgs::Point32 p;
        p.x = w_pts_i(0);
        p.y = w_pts_i(1);
        p.z = w_pts_i(2);

        point_cloud.points.push_back(p);


    }
    pub_point_cloud_short.publish(point_cloud);

}

