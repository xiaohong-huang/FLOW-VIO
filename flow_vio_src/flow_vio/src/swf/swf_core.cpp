
#include "swf.h"
#include "../solver/solver.h"
#include "../solver/solver_residualblock.h"
#include "../factor/initial_factor.h"




inline void SWFOptimization::AddFeatures(FeaturePerId& it_per_id, std::vector<VisualInertialBase*>& visual_inertial_bases, std::set<int>& factor_index_set) {
    int imu_j = it_per_id.start_frame - 1;
    int imu_i = it_per_id.start_frame ;


    bool is_cross = (it_per_id.start_frame / SWF_SIZE_IN != (it_per_id.endFrame() - 1) / SWF_SIZE_IN)
                    || (idepths_all[0].find(it_per_id.feature_id) != idepths_all[0].end() && last_marg_info->keep_block_addr_set.find(&idepths_all[0][it_per_id.feature_id]) != last_marg_info->keep_block_addr_set.end());

    if ( is_cross && it_per_id.start_frame % SWF_SIZE_IN != 0) {
        imu_i = (imu_i / SWF_SIZE_IN + 1) * SWF_SIZE_IN;
        if (factor_index_set.find(imu_i / SWF_SIZE_IN - 1) != factor_index_set.end()) {

            visual_inertial_bases[imu_i / SWF_SIZE_IN - 1]->deliver_idepth_pointer[&idepths_all[imu_i / SWF_SIZE_IN][it_per_id.feature_id]] = 0;
        }
    }
    int factor_index = imu_i / SWF_SIZE_IN;

    ASSERT(idepths_all[it_per_id.start_frame / SWF_SIZE_IN][it_per_id.feature_id] != 0);

    if (imu_i != 0 && (is_cross && (it_per_id.endFrame() - imu_i) <= LEAK_NUM)) {
        is_cross = false;
        factor_index -= 1;
        if (factor_index_set.find(imu_i / SWF_SIZE_IN - 1) != factor_index_set.end())
            visual_inertial_bases[imu_i / SWF_SIZE_IN - 1]->deliver_idepth_pointer.erase(&idepths_all[imu_i / SWF_SIZE_IN][it_per_id.feature_id]);
    }

    for (auto& it_per_frame : it_per_id.feature_per_frame) {
        imu_j++;
        if (imu_i == imu_j)continue;

        if (imu_j - 1 == (imu_i / SWF_SIZE_IN + 1)*SWF_SIZE_IN) {
            ASSERT(is_cross);

            double* idepth_pointeri = &idepths_all[imu_i / SWF_SIZE_IN][it_per_id.feature_id];
            double* idepth_pointerj = &idepths_all[(imu_j - 1) / SWF_SIZE_IN][it_per_id.feature_id];

            if (idepth_pointerj[0] == 0) {
                Eigen::Vector3d ptsInW =
                    Rs[imu_i] * (
                        RIC[0] * (it_per_id.feature_per_frame[imu_i - it_per_id.start_frame].point /
                                  idepths_all[imu_i / SWF_SIZE_IN][it_per_id.feature_id]) + TIC[0]
                    ) + Ps[imu_i];
                Vector3d pts_cj = RIC[0].transpose() * ( Rs[imu_j - 1].transpose() * (ptsInW - Ps[imu_j - 1]) - TIC[0]);
                idepth_pointerj[0] = 1.0 / pts_cj.z();
            }
            if (factor_index_set.find(imu_i / SWF_SIZE_IN) != factor_index_set.end())
                visual_inertial_bases[imu_i / SWF_SIZE_IN]->deliver_idepth_pointer[idepth_pointeri] = idepth_pointerj;
            imu_i = imu_j - 1;
        }


        double* idepth_pointer = &idepths_all[imu_i / SWF_SIZE_IN][it_per_id.feature_id];

        ASSERT(imu_i <= it_per_id.endFrame());
        ASSERT(imu_j <= it_per_id.endFrame());
        Vector3d pts_i = it_per_id.feature_per_frame[imu_i - it_per_id.start_frame].point;

        if (idepth_pointer[0] == 0) {
            Eigen::Vector3d ptsInW =
                Rs[it_per_id.start_frame] * (RIC[0] * (it_per_id.feature_per_frame[0].point / idepths_all[it_per_id.start_frame / SWF_SIZE_IN][it_per_id.feature_id]) + TIC[0]) + Ps[it_per_id.start_frame];
            Vector3d pts_cj = RIC[0].transpose() * ( Rs[imu_i].transpose() * (ptsInW - Ps[imu_i]) - TIC[0]);
            idepth_pointer[0] = 1.0 / pts_cj.z();
            ASSERT(idepths_all[it_per_id.start_frame / SWF_SIZE_IN][it_per_id.feature_id] != 0);
        }
        ASSERT(idepth_pointer[0] != 0);


        if (!is_cross) {
            if (factor_index_set.find(factor_index) == factor_index_set.end())continue;
            std::vector<double*>parameters{para_pose[imu_i], para_pose[imu_j], idepth_pointer};

            uint8_t pindexi, pindexj;
            std::unordered_map<double*, int> pos_pointer2indexs = visual_inertial_bases[factor_index]->pos_pointer2indexs;
            if (pos_pointer2indexs.find(parameters[0]) != pos_pointer2indexs.end()) pindexi = pos_pointer2indexs[parameters[0]];
            else pindexi = pos_pointer2indexs.size();

            if (pos_pointer2indexs.find(parameters[1]) != pos_pointer2indexs.end()) pindexj = pos_pointer2indexs[parameters[1]];
            else pindexj = pos_pointer2indexs.size();

            ASSERT(pindexi == imu_i - factor_index * SWF_SIZE_IN);
            ASSERT(pindexj == imu_j - factor_index * SWF_SIZE_IN);

            visual_inertial_bases[factor_index]->AddVisualFactorShort( it_per_id.feature_id,
                                                                       parameters[0], parameters[1], parameters[2],
                                                                       pindexi, pindexj,
                                                                       pts_i, it_per_frame.point
                                                                     );

        } else {

            int factor_index = (imu_j - 1) / SWF_SIZE_IN;
            if (factor_index_set.find(factor_index) == factor_index_set.end())continue;

            std::vector<double*>parameters{para_pose[imu_i], para_pose[imu_j], idepth_pointer};

            std::unordered_map<double*, int> pos_pointer2indexs = visual_inertial_bases[factor_index]->pos_pointer2indexs;
            uint8_t pindexi;
            ASSERT(pos_pointer2indexs.find(parameters[1]) != pos_pointer2indexs.end());
            if (pos_pointer2indexs.find(parameters[0]) != pos_pointer2indexs.end()) pindexi = pos_pointer2indexs[parameters[0]];
            else pindexi = pos_pointer2indexs.size();
            uint8_t pindexj = imu_j - factor_index * SWF_SIZE_IN;

            visual_inertial_bases[factor_index]->AddVisualFactorLong( it_per_id.feature_id,
                                                                      parameters[0], parameters[1], parameters[2],
                                                                      pindexi, pindexj,
                                                                      pts_i, it_per_frame.point
                                                                    );
        }
    }
}

void SWFOptimization::AddFactors(std::vector<VisualInertialBase*>& visual_inertial_bases, std::set<int>factor_index_set) {


    for (int j = 1; j < image_count; j++) {
        int factor_index = (j - 1) / SWF_SIZE_IN;
        if (factor_index_set.find(factor_index) == factor_index_set.end())continue;

        IMUFactor* factor = new IMUFactor(pre_integrations[j]);
        factor->dt_i = time_shifts[j - 1];
        factor->dt_j = time_shifts[j];
        ASSERT(factor->dt_i != 0);
        ASSERT(factor->dt_j != 0);
        IMUPreFactor* imu_pre_factor = new IMUPreFactor();
        std::vector<double*>parameters{para_pose[j - 1], para_speed_bias[j - 1],
                                       para_pose[j], para_speed_bias[j],
                                       &scale_factor,
                                       para_global_pos};
        if (ESTIMATE_TD) parameters.push_back(&td);
        if (ESTIMATE_EXTRINSIC) parameters.push_back(para_extrinsic);
        if (ESTIMATE_ACC_SCALE) parameters.push_back(acc_scale.data());
        if (ESTIMATE_GYR_SCALE) parameters.push_back(gyr_scale.data());

        imu_pre_factor->pindexi = (j - 1) - factor_index * SWF_SIZE_IN;
        imu_pre_factor->pindexj = j - factor_index * SWF_SIZE_IN;
        imu_pre_factor->factor = factor;
        imu_pre_factor->parameters = parameters;
        visual_inertial_bases[factor_index]->AddIMUFactor(imu_pre_factor);
    }

    for (auto& it_per_id : f_manager.feature) {
        if (!it_per_id.valid)continue;
        AddFeatures(it_per_id, visual_inertial_bases, factor_index_set);
    }

}

void SWFOptimization::UpdataGlobalFactors() {

    int max_factor_num = (image_count - 2) / SWF_SIZE_IN - 2;
    int old_max_factor_num = -1;
    for (int i = 0; i < max_factor_num; i++) {
        if (visual_inertial_bases_global[i] == 0) {
            old_max_factor_num = i;
            break;
        }
    }

    if (old_max_factor_num < 0)return;
    ASSERT(old_max_factor_num != max_factor_num);
    for (int i = old_max_factor_num; i < max_factor_num; i++)
        visual_inertial_bases_global[i] = new VisualInertialBase();

    std::set<int>factor_index_set;
    for (int i = old_max_factor_num; i < max_factor_num; i++)
        factor_index_set.insert(i);
    AddFactors(visual_inertial_bases_global, factor_index_set);
    if (old_max_factor_num == 0)visual_inertial_bases_global[0]->AddLastMargeInfo(last_marg_info);
    LOG_OUT << "add new\r\n";
}

void SWFOptimization::OptimizationOrMarginalization( int mode) {

    UpdataGlobalFactors();
    std::vector<VisualInertialBase*> visual_inertial_bases2;
    std::vector<VisualInertialBase*> visual_inertial_bases;
    visual_inertial_bases2.resize(image_count / SWF_SIZE_IN + 1, 0);
    idepth_pointers.clear();
    for (int i = 0; i < image_count / SWF_SIZE_IN + 1; i++)
        visual_inertial_bases2[i] = new VisualInertialBase();

    int old_leak_num = LEAK_NUM;
    std::set<int>factor_index_set;
    if (mode == MargeMode) {
        LEAK_NUM = 0;
        factor_index_set.insert(0);
        visual_inertial_bases.push_back(visual_inertial_bases2[0]);
    } else {
        for (int i = 0; i < image_count / SWF_SIZE_IN + 1; i++) {
            if (visual_inertial_bases_global[i] == 0) {
                factor_index_set.insert(i);
                visual_inertial_bases.push_back(visual_inertial_bases2[i]);
            } else
                visual_inertial_bases.push_back(visual_inertial_bases_global[i]);
        }
    }
    if (factor_index_set.size())AddFactors(visual_inertial_bases2, factor_index_set);

    LEAK_NUM = old_leak_num;
    for (int factor_index = 0; factor_index < (int)visual_inertial_bases.size(); factor_index++) {
        if (visual_inertial_bases[factor_index]->imu_map_factors.size() == 0) continue;
        for (auto it = visual_inertial_bases[factor_index]->idepth_map_factors_short.begin(); it != visual_inertial_bases[factor_index]->idepth_map_factors_short.end(); it++)
            idepth_pointers.insert(it->second->p_idepth);
        for (auto it = visual_inertial_bases[factor_index]->idepth_map_factors_long.begin(); it != visual_inertial_bases[factor_index]->idepth_map_factors_long.end(); it++)
            idepth_pointers.insert(it->second->p_idepth);

        for (auto it = visual_inertial_bases[factor_index]->deliver_idepth_pointer.begin(); it != visual_inertial_bases[factor_index]->deliver_idepth_pointer.end(); it++) {
            if (it->second)
                idepth_pointers.insert(it->second);
        }
    }

    for (auto it = idepth_pointers.begin(); it != idepth_pointers.end(); it++)
        (*it)[0] *= scale_factor;


    SolverInfo solver_info;
    VI_info_pointer_head = visual_inertial_bases[0];
    if (visual_inertial_bases[0]->last_marg_info == 0)visual_inertial_bases[0]->AddLastMargeInfo(last_marg_info);

    for (int factor_index = 0; factor_index <  (int)visual_inertial_bases.size(); factor_index++) {
        if (visual_inertial_bases[factor_index]->imu_map_factors.size() == 0) continue;
        if (factor_index != 0 && mode == MargeMode)continue;

        VI_info_pointer_tail = visual_inertial_bases[factor_index];

        if (factor_index != 0)
            visual_inertial_bases[factor_index]->prev_vibase = visual_inertial_bases[factor_index - 1];
        else visual_inertial_bases[factor_index]->prev_vibase = 0;

        if (factor_index != image_count / SWF_SIZE_IN && visual_inertial_bases[factor_index + 1] && visual_inertial_bases[factor_index + 1]->imu_map_factors.size())
            visual_inertial_bases[factor_index]->next_vibase = visual_inertial_bases[factor_index + 1];
        else visual_inertial_bases[factor_index]->next_vibase = 0;

        visual_inertial_bases[factor_index]->FactorUpdates();

        solver_info.addResidualBlockInfo(new SolverResidualBlockInfo(
                                             new VisualInertialFactor(visual_inertial_bases[factor_index])
                                             , 0, visual_inertial_bases[factor_index]->outside_pointer_raw));

    }
    if (mode == MargeMode) {
        visual_inertial_bases[0]->next_vibase = 0;
        ASSERT(visual_inertial_bases[0]->next_vibase == 0);
        ASSERT(visual_inertial_bases[0]->prev_vibase == 0);
    }

    for (int factor_index = 0; factor_index <  (int)visual_inertial_bases.size(); factor_index++) {
        if (visual_inertial_bases_global[factor_index]) visual_inertial_bases_global[factor_index]->FactorUpdates();
    }



    TicToc t_marg;

    solver_info.init_solver();
    if (mode == NormalMode) {
        solver_info.solve(8);
        static double t_sum = 0;
        static int t_count = 0;
        t_sum += t_marg.toc();
        t_count++;
        LOG_OUT << t_marg.toc() << "," << t_sum / t_count << std::endl;

    } else {
        solver_info.marginalization_process();
        std::vector<double*>new_parameters;
        std::vector<int>new_sizes;
        flow_vio::MatrixXd lhs;
        Eigen::VectorXd rhs;

        visual_inertial_bases[0]->GetMatrixNext(lhs,  rhs, new_parameters, new_sizes);

        MarginalizationInfo* marginalization_info = new MarginalizationInfo();
        marginalization_info->setmarginalizeinfo(new_parameters, new_sizes, lhs, rhs, true);
        marginalization_info->getParameterBlocks();

        if (last_marg_info)
            delete last_marg_info;
        last_marg_info = marginalization_info;

        for (int i = 0; i < (int)visual_inertial_bases_global.size(); i++) {
            if (visual_inertial_bases_global[i]) {
                delete visual_inertial_bases_global[i];
                visual_inertial_bases_global[i] = 0;
            }
        }
    }

    for (int i = 0; i < (int)visual_inertial_bases_global.size() - 1; i++) {
        if (visual_inertial_bases_global[i] && !visual_inertial_bases_global[i + 1])visual_inertial_bases_global[i]->next_vibase = 0;
    }

    for (auto it = idepth_pointers.begin(); it != idepth_pointers.end(); it++)
        (*it)[0] /= scale_factor;

    for (int i = 0; i < image_count / SWF_SIZE_IN + 1; i++) {
        if (visual_inertial_bases2[i]) delete visual_inertial_bases2[i];
    }

    LOG_OUT << "scale_factor:" << scale_factor << std::endl;

}
