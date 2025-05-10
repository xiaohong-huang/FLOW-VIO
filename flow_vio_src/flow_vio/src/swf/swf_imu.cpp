#include "swf.h"
#include "../utility/visualization.h"
#include<fstream>
#include "../factor/initial_factor.h"

void SWFOptimization::InitializePos() {
    for (int i = 0; i < SWF_WINDOW_SIZE + 1; i++)
        Bgs[i].setZero();

    LOG_OUT << "initial bgs:" << Bgs[0];
    printf("averge acc %f %f %f\n", acc_mean.x(), acc_mean.y(), acc_mean.z());

    Vector3d mag_mean;
    mag_mean(0) = 0;
    mag_mean(1) = 1;
    mag_mean(2) = 0;

    Matrix3d Rwb0;
    Eigen::Vector3d z0 = acc_mean.normalized();
    Eigen::Vector3d x0 = (Utility::skewSymmetric(mag_mean) * z0).normalized();
    Eigen::Vector3d y0 = (Utility::skewSymmetric(z0) * x0).normalized();
    Rwb0.block(0, 0, 1, 3) = x0.transpose();
    Rwb0.block(1, 0, 1, 3) = y0.transpose();
    Rwb0.block(2, 0, 1, 3) = z0.transpose();


    for (int i = 0; i < SWF_WINDOW_SIZE + 1; i++)
        Rs[i] = Rwb0;
    LOG_OUT << "init R0: " << endl
            << Utility::R2ypr(Rs[0]).transpose() << ","
            << Utility::R2ypr(Rwb0).transpose() << ","
            << endl;
}



void SWFOptimization::InputIMU(double t, const Vector3d& linearAcceleration, const Vector3d& angularVelocity) {
    if (first_observe_time == 0)
        first_observe_time = t;
    if (t < first_observe_time + SKIP_TIME)
        return;
    if (t < first_observe_time + SKIP_TIME + AVERAGE_TIME) {
        acc_mean += linearAcceleration;
        acc_count++;
        return;
    }
    static bool imu_initialize = false;
    if (!imu_initialize) {
        imu_initialize = true;
        acc_mean /= acc_count;
        InitializePos();
    }

    acc_buf.push(make_pair(t, linearAcceleration));
    gyr_buf.push(make_pair(t, angularVelocity));
}


bool SWFOptimization::GetImuInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>>& accVector,
                                     vector<pair<double, Eigen::Vector3d>>& gyrVector) {
    if (acc_buf.empty()) {
        printf("not receive imu\n");
        return false;
    }
    if (t1 <= acc_buf.back().first) {
        while (acc_buf.front().first <= t0) {
            acc_buf.pop();
            gyr_buf.pop();
        }
        while (acc_buf.front().first < t1) {
            accVector.push_back(acc_buf.front());
            acc_buf.pop();
            gyrVector.push_back(gyr_buf.front());
            gyr_buf.pop();
        }
        accVector.push_back(acc_buf.front());
        gyrVector.push_back(gyr_buf.front());
    } else {
        printf("wait for imu\n");
        return false;
    }
    return true;
}


bool SWFOptimization::ImuAvailable(double t) {

    double tend = acc_buf.back().first;
    if (!acc_buf.empty() && t <= tend)
        return true;
    else
        return false;
}


void SWFOptimization::ImuIntegrate() {
    if (AMP) {
        static double ACC_N0 = ACC_N, INT_N0 = INT_N, ACC_W0 = ACC_W, GYR_N0 = GYR_N, GYR_W0 = GYR_W;
        if (image_count < SWF_WINDOW_SIZE / AMP_NUM) {
            ACC_N = ACC_N0 * AMP;
            ACC_W = ACC_W0 * AMP;
            GYR_N = GYR_N0 * AMP;
            GYR_W = GYR_W0 * AMP;
        } else {
            static bool is_first = true;
            if (is_first) {
                is_first = false;
                ACC_N = ACC_N0; INT_N = INT_N0; ACC_W = ACC_W0; GYR_N = GYR_N0; GYR_W = GYR_W0;
                for (int i = 0; i < (int)pre_integrations.size(); i++) {
                    if (pre_integrations[i]) {
                        pre_integrations[i]->noise = Eigen::Matrix<double, 18, 18>::Zero();
                        pre_integrations[i]->noise.block<3, 3>(0, 0) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
                        pre_integrations[i]->noise.block<3, 3>(3, 3) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
                        pre_integrations[i]->noise.block<3, 3>(6, 6) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
                        pre_integrations[i]->noise.block<3, 3>(9, 9) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
                        pre_integrations[i]->noise.block<3, 3>(12, 12) =  (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
                        pre_integrations[i]->noise.block<3, 3>(15, 15) =  (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();
                        pre_integrations[i]->repropagate(Bas[i], Bgs[i], acc_scale, gyr_scale);
                    }

                }
                if (visual_inertial_bases_global[0]) visual_inertial_bases_global[0]->ResetInit();
            }
        }
    }
    vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;

    GetImuInterval(prev_time, cur_time, accVector, gyrVector);

    ASSERT(accVector.size() >= 1);
    ASSERT(prev_time == prev_time2);
    ASSERT(cur_time >= prev_time);

    ASSERT(accVector.size() > 2);
    double last_time_stamp = accVector[accVector.size() - 1].first;
    double second_last_time_stamp = accVector[accVector.size() - 2].first;
    int acc_size = accVector.size();
    if (abs(last_time_stamp - cur_time) > abs(second_last_time_stamp - cur_time))
        acc_size -= 1;
    Ps[image_count - 1] += Vs[image_count - 1] * old_time_shift;
    Rs[image_count - 1] *= Utility::deltaQ((gyr_0 - Bgs[image_count - 1]) * old_time_shift).toRotationMatrix();
    static double old_imu_time = -1;
    for (int i = 0; i < acc_size; i++) {
        double t = accVector[i].first;
        if (old_imu_time < 0) old_imu_time = t;
        double dt = t - old_imu_time;
        ASSERT(dt >= 0);
        old_imu_time = t;
        headers[image_count - 1] = t;
        if (dt > 0)IMUProcess(dt, accVector[i].second, gyrVector[i].second);
    }
    time_shifts[image_count - 1] = old_imu_time - cur_time;
    if (time_shifts[image_count - 1] == 0)time_shifts[image_count - 1] = 1e-10;
    ASSERT(time_shifts[image_count - 1] != 0);
    old_time_shift = time_shifts[image_count - 1];
    Ps[image_count - 1] -= Vs[image_count - 1] * old_time_shift;
    Rs[image_count - 1] *= Utility::deltaQ(-(gyr_0 - Bgs[image_count - 1]) * old_time_shift).toRotationMatrix();

    prev_time = cur_time;

}

void SWFOptimization::IMUProcess( double dt, const Vector3d& linear_acceleration, const Vector3d& angular_velocity) {
    static bool first_imu = true;
    if (first_imu) {
        first_imu = false;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }
    if (!pre_integrations[image_count - 1]) {
        pre_integrations[image_count - 1] =
            new IntegrationBase{acc_0, gyr_0, Bas[image_count  - 1], Bgs[image_count - 1], acc_scale, gyr_scale};
    }
    if (image_count - 1 != 0) {
        pre_integrations[image_count - 1]->push_back(dt, linear_acceleration, angular_velocity);
        dt_buf[image_count - 1].push_back(dt);
        linear_acceleration_buf[image_count - 1].push_back(linear_acceleration);
        angular_velocity_buf[image_count - 1].push_back(angular_velocity);
        int j = image_count - 1;
        Vector3d un_acc_0 = Rs[j] * ((acc_0.array() * acc_scale.array()).matrix() - Bas[j]) -  G;
        Vector3d un_gyr = 0.5 * ((gyr_0 + angular_velocity).array() * gyr_scale.array()).matrix() - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * ((linear_acceleration.array() * acc_scale.array()).matrix() - Bas[j]) - G;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;

    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;

}






void SWFOptimization::InitializeSqrtInfo() {


    Vector2Double();
    MarginalizationInfo* marginalization_info = new MarginalizationInfo();

    {
        Eigen::Matrix<double, 6, 6>sqrt_info_pose;
        sqrt_info_pose.setZero();
        sqrt_info_pose.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Identity() * 1e3;
        sqrt_info_pose.block<3, 3>(3, 3) = Eigen::Matrix<double, 3, 3>::Identity() * 1e3;
        ResidualBlockInfo* residual_block_info = new ResidualBlockInfo(
            new InitialPoseFactor(Ps[0] + Rs[0]*TIC[0], Quaterniond(Rs[0]*RIC[0]), sqrt_info_pose),
            0, std::vector<double*> {para_pose[0]}, std::vector<int> {}, std::vector<int> {});
        marginalization_info->addResidualBlockInfo(residual_block_info);
    }

    {
        Eigen::Matrix<double, 9, 9>sqrt_info_bias;
        sqrt_info_bias.setZero();
        sqrt_info_bias.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Identity() * 1e1;
        sqrt_info_bias.block<3, 3>(3, 3) = Eigen::Matrix<double, 3, 3>::Identity() * 1;
        sqrt_info_bias.block<3, 3>(6, 6) = Eigen::Matrix<double, 3, 3>::Identity() * 1e1;
        ResidualBlockInfo* residual_block_info = new ResidualBlockInfo(
            new InitialBiasFactor(Vs[0], Bas[0], Bgs[0], sqrt_info_bias),
            0, std::vector<double*> {para_speed_bias[0]}, std::vector<int> {}, std::vector<int> {});
        marginalization_info->addResidualBlockInfo(residual_block_info);
    }
    if (ESTIMATE_EXTRINSIC) {
        Eigen::Matrix<double, 6, 6>sqrt_info_pose;
        sqrt_info_pose.setZero();
        sqrt_info_pose.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Identity() * 100;
        sqrt_info_pose.block<3, 3>(3, 3) = Eigen::Matrix<double, 3, 3>::Identity() * 100;
        ResidualBlockInfo* residual_block_info = new ResidualBlockInfo(
            new InitialPoseFactor(TIC[0], QIC[0], sqrt_info_pose),
            0, std::vector<double*> {para_extrinsic}, std::vector<int> {}, std::vector<int> {});
        marginalization_info->addResidualBlockInfo(residual_block_info);
    }
    if (ESTIMATE_ACC_SCALE) {
        Eigen::Matrix<double, 3, 3>sqrt_info_imu_scale;
        sqrt_info_imu_scale.setZero();
        sqrt_info_imu_scale.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Identity() * 200;
        ResidualBlockInfo* residual_block_info = new ResidualBlockInfo(
            new InitialFactor33(Eigen::Vector3d({1, 1, 1}), sqrt_info_imu_scale),
            0, std::vector<double*> {acc_scale.data()}, std::vector<int> {}, std::vector<int> {});
        marginalization_info->addResidualBlockInfo(residual_block_info);
    }
    if (ESTIMATE_GYR_SCALE) {
        Eigen::Matrix<double, 3, 3>sqrt_info_imu_scale;
        sqrt_info_imu_scale.setZero();
        sqrt_info_imu_scale.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Identity() * 200;
        ResidualBlockInfo* residual_block_info = new ResidualBlockInfo(
            new InitialFactor33(Eigen::Vector3d({1, 1, 1}), sqrt_info_imu_scale),
            0, std::vector<double*> {gyr_scale.data()}, std::vector<int> {}, std::vector<int> {});
        marginalization_info->addResidualBlockInfo(residual_block_info);
    }


    marginalization_info->marginalize(true);
    marginalization_info->getParameterBlocks();
    if (last_marg_info) delete last_marg_info;
    last_marg_info = marginalization_info;

}
