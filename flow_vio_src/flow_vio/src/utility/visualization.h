

#pragma once

#include <ros/ros.h>

#include <eigen3/Eigen/Dense>

#include "../swf/swf.h"



void registerPub(ros::NodeHandle& n);

void printStatistics(const SWFOptimization& swf_optimization, double t);

void pubOdometry(const SWFOptimization& swf_optimization, const std_msgs::Header& header);

void pubCameraPose(const SWFOptimization& swf_optimization, const std_msgs::Header& header);

void pubPointCloud(const SWFOptimization& swf_optimization, const std_msgs::Header& header);

void resetpot(const SWFOptimization& swf_optimization, const std_msgs::Header& header);

