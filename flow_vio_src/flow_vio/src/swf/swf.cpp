

#include "swf.h"
#include "../utility/visualization.h"
#include <thread>
#include <queue>



SWFOptimization::SWFOptimization() {
    printf("init begins");
    ClearState();
    prev_time = -1;
    prev_time2 = -1;
    cur_time = 0;
    R_WI_WC.setIdentity();

}


void SWFOptimization::SetParameter() {

    LOG_OUT << "set G " << G.transpose() << endl;
    feature_tracker.readIntrinsicParameter(CAM_NAMES);

}



//need to fix for reseting the system.
void SWFOptimization::ClearState() {
    visual_inertial_bases_global.resize(SWF_SIZE_OUT + 1, 0);
    idepths_all.resize(SWF_SIZE_OUT + 1);
    Ps.resize(SWF_WINDOW_SIZE + 1);
    Vs.resize(SWF_WINDOW_SIZE + 1);
    Rs.resize(SWF_WINDOW_SIZE + 1);
    Bas.resize(SWF_WINDOW_SIZE + 1);
    Bgs.resize(SWF_WINDOW_SIZE + 1);
    headers.resize(SWF_WINDOW_SIZE + 1);
    para_pose.resize(SWF_WINDOW_SIZE + 1, 0);
    para_speed_bias.resize(SWF_WINDOW_SIZE + 1, 0);
    dt_buf.resize(SWF_WINDOW_SIZE + 1);
    linear_acceleration_buf.resize(SWF_WINDOW_SIZE + 1);
    angular_velocity_buf.resize(SWF_WINDOW_SIZE + 1);
    pre_integrations.resize(SWF_WINDOW_SIZE + 1, 0);
    time_shifts.resize(SWF_WINDOW_SIZE + 1, 0);
    old_time_shift = 0;

    for (int i = 0; i < SWF_WINDOW_SIZE + 1; i++) {

        if (para_pose[i]) delete para_pose[i];
        if (para_speed_bias[i]) delete para_speed_bias[i];
        if (pre_integrations[i] != nullptr) delete pre_integrations[i];

        para_pose[i] = new double[SIZE_POSE];
        para_speed_bias[i] = new double[SIZE_SPEEDBIAS];

        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < SWF_SIZE_OUT + 1; i++) {
        if (visual_inertial_bases_global[i] != nullptr)
            delete visual_inertial_bases_global[i];
        visual_inertial_bases_global[i] = nullptr;
    }


    solver_flag = Initial;
    last_marg_info = nullptr;
    image_count = 0;
    acc_count = 0;

    f_manager.ClearState();
    acc_mean.setZero();

    acc_scale = Eigen::Vector3d({1, 1, 1});
    gyr_scale = Eigen::Vector3d({1, 1, 1});

    if (last_marg_info != nullptr)delete last_marg_info;
}


//getting the pointer of the states.
void SWFOptimization::Vector2Double() {

    Quaterniond q {R_WI_WC};
    para_global_pos[3] = q.x();
    para_global_pos[4] = q.y();
    para_global_pos[5] = q.z();
    para_global_pos[6] = q.w();


    for (int i = 0; i < image_count; i++) {
        Eigen::Vector3d Pc = Ps[i] + Rs[i] * TIC[0];
        Eigen::Matrix3d Rc = Rs[i] * RIC[0];
        Pc = R_WI_WC.transpose() * Pc / scale_factor;
        Rc = R_WI_WC.transpose() * Rc;

        para_pose[i][0] = Pc.x();
        para_pose[i][1] = Pc.y();
        para_pose[i][2] = Pc.z();
        Quaterniond q{Rc};
        para_pose[i][3] = q.x();
        para_pose[i][4] = q.y();
        para_pose[i][5] = q.z();
        para_pose[i][6] = q.w();

        para_speed_bias[i][0] = Vs[i].x();
        para_speed_bias[i][1] = Vs[i].y();
        para_speed_bias[i][2] = Vs[i].z();

        para_speed_bias[i][3] = Bas[i].x();
        para_speed_bias[i][4] = Bas[i].y();
        para_speed_bias[i][5] = Bas[i].z();

        para_speed_bias[i][6] = Bgs[i].x();
        para_speed_bias[i][7] = Bgs[i].y();
        para_speed_bias[i][8] = Bgs[i].z();

    }

    if (ESTIMATE_EXTRINSIC) {
        para_extrinsic[0] = TIC[0].x();
        para_extrinsic[1] = TIC[0].y();
        para_extrinsic[2] = TIC[0].z();
        para_extrinsic[3] = QIC[0].x();
        para_extrinsic[4] = QIC[0].y();
        para_extrinsic[5] = QIC[0].z();
        para_extrinsic[6] = QIC[0].w();
    }

}


//saving the states from pointer.
void SWFOptimization::Double2Vector() {

    R_WI_WC = Quaterniond(para_global_pos[6], para_global_pos[3], para_global_pos[4], para_global_pos[5]).normalized().toRotationMatrix();

    if (ESTIMATE_EXTRINSIC) {
        TIC[0] = Vector3d(para_extrinsic[0], para_extrinsic[1], para_extrinsic[2] ) ;
        QIC[0] = Quaterniond(para_extrinsic[6], para_extrinsic[3], para_extrinsic[4], para_extrinsic[5]).normalized();
        RIC[0] = QIC[0].toRotationMatrix();
    }

    for (int i = 0; i < image_count; i++) {

        Eigen::Vector3d Pc;
        Eigen::Matrix3d Rc;

        Rc = Quaterniond(para_pose[i][6], para_pose[i][3], para_pose[i][4], para_pose[i][5]).normalized().toRotationMatrix();
        Pc = Vector3d(para_pose[i][0], para_pose[i][1], para_pose[i][2] ) ;

        Pc = R_WI_WC * Pc * scale_factor;
        Rc = R_WI_WC * Rc;

        Rs[i] = Rc * RIC[0].transpose();
        Ps[i] = Pc - Rs[i] * TIC[0];
        Vs[i] = Vector3d(para_speed_bias[i][0], para_speed_bias[i][1], para_speed_bias[i][2]);
        Bas[i] = Vector3d(para_speed_bias[i][3], para_speed_bias[i][4], para_speed_bias[i][5]);
        Bgs[i] = Vector3d(para_speed_bias[i][6], para_speed_bias[i][7], para_speed_bias[i][8]);

    }

    #pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic) if (NUM_THREADS > 1)
    for (int j = 1; j < image_count; j++) {
        if (pre_integrations[j]->dt_buf.size() > 5000)continue;
        if ((Bas[j - 1] - pre_integrations[j]->linearized_ba).norm() > 0.01 || (Bgs[j - 1] - pre_integrations[j]->linearized_bg).norm() > 0.001 || (acc_scale - pre_integrations[j]->acc_scale).norm() > 0.001 || (gyr_scale - pre_integrations[j]->gyr_scale).norm() > 0.001)
            pre_integrations[j]->repropagate(Bas[j - 1], Bgs[j - 1], acc_scale, gyr_scale
                                            );
    }

    for (auto& it_per_id : f_manager.feature) {
        if (!it_per_id.valid)continue;
        it_per_id.solve_flag = 1;
        for (int i = 0; i < SWF_SIZE_OUT; i++) {
            if (idepths_all[i].find(it_per_id.feature_id) != idepths_all[i].end() && idepths_all[i][it_per_id.feature_id] < 1 / 500.0)
                it_per_id.solve_flag = 2;
        }
    }

}


//
void SWFOptimization::SlideWindowFrame(int frameindex, int windowsize, bool updateIMU) {

    if (frameindex != 0) {
        for (unsigned int i = 0; i < dt_buf[frameindex + 1].size(); i++) {
            pre_integrations[frameindex]->push_back(dt_buf[frameindex + 1][i], linear_acceleration_buf[frameindex + 1][i], angular_velocity_buf[frameindex + 1][i]);
            dt_buf[frameindex].push_back(dt_buf[frameindex + 1][i]);
            linear_acceleration_buf[frameindex].push_back(linear_acceleration_buf[frameindex + 1][i]);
            angular_velocity_buf[frameindex].push_back(angular_velocity_buf[frameindex + 1][i]);
        }
        std::swap(pre_integrations[frameindex], pre_integrations[frameindex + 1]);
        dt_buf[frameindex].swap(dt_buf[frameindex + 1]);
        linear_acceleration_buf[frameindex].swap(linear_acceleration_buf[frameindex + 1]);
        angular_velocity_buf[frameindex].swap(angular_velocity_buf[frameindex + 1]);
    }

    for (int i = frameindex; i < windowsize - 1; i++) {
        headers[i] = headers[i + 1];
        Rs[i] = Rs[i + 1];
        Ps[i] = Ps[i + 1];
        Vs[i] = Vs[i + 1];
        Bas[i] = Bas[i + 1];
        Bgs[i] = Bgs[i + 1];

        time_shifts[i] = time_shifts[i + 1];

        std::swap(para_pose[i], para_pose[i + 1]);
        std::swap(para_speed_bias[i], para_speed_bias[i + 1]);
        std::swap(pre_integrations[i], pre_integrations[i + 1]);

        dt_buf[i].swap(dt_buf[i + 1]);
        linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
        angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);
    }
    delete para_pose[windowsize - 1];
    delete para_speed_bias[windowsize - 1];
    delete pre_integrations[windowsize - 1];

    para_pose[windowsize - 1] = new double[SIZE_POSE];
    para_speed_bias[windowsize - 1] = new double[SIZE_SPEEDBIAS];
    pre_integrations[windowsize - 1] = 0;

    dt_buf[windowsize - 1].clear();
    linear_acceleration_buf[windowsize - 1].clear();
    angular_velocity_buf[windowsize - 1].clear();
}


//marginalizing the select frames.
//param margeindex is the set of frame indexes that are selected to be marginalized.
void SWFOptimization::MargFrames() {

    if (marg_flag == MargImagOld) {
        OptimizationOrMarginalization( MargeMode);
        have_hist = 1;
    }
}




void SWFOptimization::SlideWindow() {
    TicToc t_marg;
    if (solver_flag != Initial) {
        if (imag_marg_index != 0) marg_flag = MargImagSecondNew;
        else marg_flag = MargImagOld;
    } else
        return;


    if (marg_flag == MargImagOld && image_count <= SWF_WINDOW_SIZE)
        return;

    if (marg_flag == MargImagOld) {
        Eigen::Vector3d P0 = Ps[0];
        Eigen::Matrix3d R0 = Rs[0];
        Eigen::Vector3d P1 = Ps[1];
        Eigen::Matrix3d R1 = Rs[1];

        MargFrames();

        if (visual_inertial_bases_global[0])delete visual_inertial_bases_global[0];
        visual_inertial_bases_global[0] = 0;
        for (int i = 0; i < (int)visual_inertial_bases_global.size() - 1; i++)
            visual_inertial_bases_global[i] = visual_inertial_bases_global[i + 1];

        for (int i = 0; i < SWF_SIZE_IN; i++)
            SaveKefPos(i);

        int frame_counts = image_count;
        for (int i = 0; i < SWF_SIZE_IN; i++) {
            SlideWindowFrame( 0, frame_counts, 1);
            frame_counts--;
        }
        SlideWindowOld( P0, R0, P1, R1, TIC[0], RIC[0]);
        image_count -= SWF_SIZE_IN;

#if USE_ASSERT
        for (int i = 0; i < (int)last_marg_info->keep_block_size.size(); i++) {
            if (last_marg_info->keep_block_size[i] == 1)
                ASSERT(fabs(last_marg_info->keep_block_addr[i][0] - last_marg_info->keep_block_data[i][0]) < 1e4);
        }
#endif

    } else if (marg_flag == MargImagSecondNew) {

        SaveLocalPos(image_count - 2, image_count - 3);

        Eigen::Vector3d P0 = Ps[(image_count - 2)];
        Eigen::Matrix3d R0 = Rs[(image_count - 2)];
        Eigen::Vector3d P1 = Ps[(image_count - 1)];
        Eigen::Matrix3d R1 = Rs[(image_count - 1)];
        SlideWindowFrame((image_count - 2), image_count, 1);
        SlideWindowNew(P0, R0, P1, R1, TIC[0], RIC[0]);
        image_count--;
    }
    LOG_OUT << "marge time:" << t_marg.toc() << std::endl;

#if USE_ASSERT
    for (int i = 0; i < (int)last_marg_info->keep_block_addr.size(); i++) {
        double* pointer = last_marg_info->keep_block_addr[i];
        if (last_marg_info->keep_block_size[i] == 1) {
            bool found = false;
            for (auto& it_per_id : f_manager.feature) {
                if (!it_per_id.valid)continue;
                bool condition = (idepths_all[0].find(it_per_id.feature_id) != idepths_all[0].end() && pointer == &idepths_all[0][it_per_id.feature_id]) || pointer == &scale_factor;
                if (ESTIMATE_TD)
                    condition = condition || (pointer == &td);
                if (condition)
                    found = true;
            }
            ASSERT(found);
        }
    }
    for (int i = 0; i < (int)idepths_all.size(); i++) {
        for (auto it = idepths_all[i].begin(); it != idepths_all[i].end(); it++)
            ASSERT(it->second != 0);
    }
#endif

}




void SWFOptimization::SaveLocalPos(int m_index, int r_index) {
    if (!enable_output)return;
    PosInfo pos_info;

    Eigen::Vector3d Pc_m = Ps[m_index] + Rs[m_index] * TIC[0];
    Eigen::Matrix3d Rc_m = Rs[m_index] * RIC[0];
    Pc_m = R_WI_WC.transpose() * Pc_m / scale_factor;
    Rc_m = R_WI_WC.transpose() * Rc_m;

    Eigen::Vector3d Pc_r = Ps[r_index] + Rs[r_index] * TIC[0];
    Eigen::Matrix3d Rc_r = Rs[r_index] * RIC[0];
    Pc_r = R_WI_WC.transpose() * Pc_r / scale_factor;
    Rc_r = R_WI_WC.transpose() * Rc_r;

    pos_info.t = Rc_r.transpose() * (Pc_m - Pc_r);
    pos_info.R = Rc_r.transpose() * Rc_m;
    // pos_info.scale_factor = scale_factor;
    pos_info.time_stamp = headers[m_index];
    pos_save[para_pose[r_index]].push_back(pos_info);
}

void SWFOptimization::SaveKefPos(int r_index) {
    if (!enable_output)return;
    PosInfo pos_info;
    Eigen::Vector3d Pc_r = Ps[r_index] + Rs[r_index] * TIC[0];
    Eigen::Matrix3d Rc_r = Rs[r_index] * RIC[0];
    Pc_r = R_WI_WC.transpose() * Pc_r / scale_factor;
    Rc_r = R_WI_WC.transpose() * Rc_r;
    pos_info.R = Rc_r;
    pos_info.t = Pc_r;
    pos_info.time_stamp = headers[r_index];
    if (pos_save_all.size())
        ASSERT(pos_save_all[pos_save_all.size() - 1].time_stamp < pos_info.time_stamp);
    pos_save_all.push_back(pos_info);
    if (pos_save.find(para_pose[r_index]) != pos_save.end()) {
        auto& tmp = pos_save[para_pose[r_index]];
        for (int j = 0; j < (int)tmp.size(); j++) {

            PosInfo pos_info;
            pos_info.time_stamp = tmp[j].time_stamp;
            pos_info.R = Rc_r * tmp[j].R;
            pos_info.t = Rc_r * tmp[j].t + Pc_r;
            if (pos_save_all.size())
                ASSERT(pos_save_all[pos_save_all.size() - 1].time_stamp < pos_info.time_stamp);
            pos_save_all.push_back(pos_info);
        }
        pos_save.erase(para_pose[r_index]);
    }
}
std::vector<PosInfo> SWFOptimization::RetriveAllPose() {

    for (int i = 0; i < image_count; i++)
        SaveKefPos(i);
    return pos_save_all;
}


//main process
void SWFOptimization::MeasurementProcess() {

    while (!feature_buf.empty() && !acc_buf.empty()) {

        TicToc t_process;
        pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>> feature;

        feature = feature_buf.front();
        cur_time = feature.first;
        ASSERT(!(cur_time <= prev_time2 || cur_time - prev_time2 <= 0.005));

        if (!ImuAvailable(cur_time))
            return;

        feature_buf.pop();
        image_count++;
        ImuIntegrate();


        prev_time2 = cur_time;
        headers[image_count - 1] = cur_time;
        ASSERT(fabs(headers[image_count - 1] - cur_time) < 1e-4);
        ASSERT(image_count <= SWF_WINDOW_SIZE + 1);
        ImagePreprocess(feature.second);

        if (solver_flag != Initial )
            MyOptimization();

        ImagePostprocess();

        SlideWindow();

#if USE_ASSERT
        if (last_marg_info)
            for (int i = 0; i < (int)last_marg_info->keep_block_addr.size(); i++) {
                double* pointer = last_marg_info->keep_block_addr[i];
                if (last_marg_info->keep_block_size[i] == 1) {
                    bool found = false;
                    for (auto& it_per_id : f_manager.feature) {
                        if (!it_per_id.valid)continue;
                        bool condition = (idepths_all[0].find(it_per_id.feature_id) != idepths_all[0].end() && pointer == &idepths_all[0][it_per_id.feature_id]) || pointer == &scale_factor;
                        if (ESTIMATE_TD)
                            condition = condition || (pointer == &td);
                        if (condition)
                            found = true;
                    }
                    ASSERT(found);
                }
            }

#endif

        Ps[image_count] = Ps[image_count - 1];
        Rs[image_count] = Rs[image_count - 1];
        Vs[image_count] = Vs[image_count - 1];
        Bas[image_count] = Bas[image_count - 1];
        Bgs[image_count] = Bgs[image_count - 1];
        headers[image_count] = headers[image_count - 1];

        static Eigen::Vector3d old_P = Ps[image_count - 1];
        if (solver_flag != Initial) {
            static double travel_distance = 0;
            travel_distance += (Ps[image_count - 1] - old_P).norm();
            if (!has_excitation && travel_distance > 1 && Vs[image_count - 1].norm() > 0.1 && image_count > 11) {
                has_excitation = true;
                enable_output = true;
                if (!have_hist) {
                    fix_scale = true;
                    if (visual_inertial_bases_global[0]) visual_inertial_bases_global[0]->ResetInit();
                }
            }
        }
        old_P = Ps[image_count - 1];


        if (solver_flag != Initial)
            PubData();

        {
            static double t_process2 = 0;
            static int t_count = 0;
            double ts = t_process.toc();
            t_process2 += ts;
            t_count += 1;
            printf("process measurement time: %f   ,%f   ,%f   \n", headers[image_count], ts, t_process2 / t_count);
            LOG_OUT << "process measurement time: " << ts << "," << t_process2 / t_count << std::endl << std::endl;
            LOG_OUT.flush();
        }

        if (last_marg_info) {
            int count = 0;
            for (int i = 0; i < (int)last_marg_info->keep_block_addr.size(); i++) {
                ASSERT(!(last_marg_info->keep_block_size[i] == 9 && last_marg_info->keep_block_addr[i] != para_speed_bias[0]));
                if (last_marg_info->keep_block_size[i] == 1)
                    count++;
            }
            ASSERT((int)last_marg_info->keep_block_size.size() - count <= 3 + SWF_SIZE_IN);
            LOG_OUT << "prior:" << count << std::endl;
            for (int i = 0; i < (int)idepths_all.size(); i++) {
                for (auto it = idepths_all[i].begin(); it != idepths_all[i].end(); it++)
                    ASSERT(it->second != 0);
            }
        }
    }

}




//publicating and saving results.
void SWFOptimization::PubData() {


    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp = ros::Time(headers[image_count - 1]);
    if (!pub_init) {
        resetpot(*this, header);
        pub_init = true;
    }
    printStatistics(*this, 0);

    if (enable_output) {
        pubOdometry(*this, header);
        pubCameraPose(*this, header);
        pubPointCloud(*this, header);
    }

}

