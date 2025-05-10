#include "visual_inerial_base.h"
#include "../utility/utility.h"
#include "operation.h"

#include "../utility/tic_toc.h"

#define VRNUM (6+6+1)

#define LEFT 1
#define RIGHT 0


void VisualInertialBase::BuildForwardMatrixInfo() {

    flow_vio::Matrix1_3d reduce;


    setRPwc();
    int shift_cur = order2p_local[O_BIAS0];
    int shift2 = order2p_local[O_IDEPTHR];
    matrix_info.clear();

    matrix_info.push_back(MatirxInfo(order2p_local[O_BIAS0] - shift_cur, order2p_local[O_BIAS0] - shift_cur,  true,
                                     order2p_local[O_IDEPTHR] - order2p_local[O_BIAS0], order2p_local[O_IDEPTHR] - order2p_local[O_BIAS0]));//diagonal matrix,A

    flow_vio::MatrixXd idepth_pos1 = flow_vio::MatrixXd::Zero(deliver_idepth_pointer.size(), 12);
    flow_vio::MatrixXd idepth_pos0 = flow_vio::MatrixXd::Zero(deliver_idepth_pointer.size(), 6);

    for (auto& it : long_feature_factors) {
        double* p_idepthi = it->p_idepth;
        auto& pindexs = it->pindexs;
        if (deliver_idepth_pointer.find(p_idepthi) != deliver_idepth_pointer.end()) {
            int idepthi_p_local = pointer2p_local[p_idepthi] - shift_cur;
            if (pindexs[0] == 0) {

                Eigen::Vector3d pts_i = it->ptss[0];
                Eigen::Vector3d pts_camera_j;
                pts_camera_j = Rwcs[SWF_SIZE_IN].transpose() * (Rwcs[0] * pts_i / p_idepthi[0] + Pwcs[0] - Pwcs[SWF_SIZE_IN]);
                double dep_j = pts_camera_j.z();
#if USE_ASSERT
                ASSERT((int)pindexs.size() >= SWF_SIZE_IN + 1);
                ASSERT(pindexs[pindexs.size() - 1] == SWF_SIZE_IN);
                ASSERT((int)it->ptss.size() >= SWF_SIZE_IN + 1);
                ASSERT(deliver_idepth_pointer[p_idepthi]);
                ASSERT( fabs(deliver_idepth_pointer[p_idepthi][0] - 1. / dep_j) < 1e-6);
                ASSERT(idepth_pointers.find(deliver_idepth_pointer[p_idepthi]) != idepth_pointers.end());
                ASSERT(idepth_pointers.find(p_idepthi) != idepth_pointers.end());
#endif

                reduce << 0, 0, -1 / (dep_j * dep_j);
                double idepth_tmp;
                idepth_tmp = 1 / (reduce * Rwcs[SWF_SIZE_IN].transpose() * Rwcs[0] * pts_i * -1.0 / (p_idepthi[0] * p_idepthi[0]))(0); //D^-1

                flow_vio::Matrix1_6d tmp;

                tmp.block<1, 3>(0, 0) = reduce * Rwcs[SWF_SIZE_IN].transpose();

                tmp.block<1, 3>(0, 3) = -reduce * Rwcs[SWF_SIZE_IN].transpose() * Rwcs[0] * Utility::skewSymmetric(pts_i / p_idepthi[0]);

                idepth_pos1.block<1, 6>(pointer2p_local[p_idepthi] - shift2, 0) = -idepth_tmp * tmp; //-D^-1xC

                tmp.block<1, 3>(0, 0) = -reduce * Rwcs[SWF_SIZE_IN].transpose();
                tmp.block<1, 3>(0, 3) = reduce * Utility::skewSymmetric(pts_camera_j);
                idepth_pos1.block<1, 6>(pointer2p_local[p_idepthi] - shift2, 6) = -idepth_tmp * tmp; //-D^-1xC

                matrix_info.push_back(MatirxInfo(idepthi_p_local, idepthi_p_local, flow_vio::Vector1d(&idepth_tmp), false, 1, 1)); //D^-1

            } else {
#if USE_ASSERT
                ASSERT(pindexs[0] == SWF_SIZE_IN);
                ASSERT(deliver_idepth_pointer[p_idepthi] == 0);
#endif
                matrix_info.push_back(MatirxInfo(idepthi_p_local, idepthi_p_local,  true, 1, 1));
            }
        }
    }

    matrix_info.push_back(MatirxInfo(order2p_local[O_IDEPTHR] - shift_cur, order2p_local[O_POS0] - shift_cur, idepth_pos1, false, idepth_pos1.rows(), idepth_pos1.cols()));

}


void VisualInertialBase::BuildCurPrevIdx() {

    cur_idxs.clear();
    prev_idxs.clear();
    cur_sizes.clear();

    if (prev_vibase) {
        int shift_cur = order2p_local[O_POSR];
        int shift_prev = prev_vibase->order2p_local[prev_vibase->O_POSR];

        //add previous matrix
        for (int order = prev_vibase->O_POSK; order < prev_vibase->O_IDEPTHR + (int)prev_vibase->deliver_idepth_pointer.size(); ++order) {
            double* prev_pointeri = prev_vibase->order2pointers[order];
            double* cur_pointeri = prev_pointeri;
            if (order >= prev_vibase->O_IDEPTHR) {
                ASSERT(prev_vibase->deliver_idepth_pointer.find(prev_pointeri) != prev_vibase->deliver_idepth_pointer.end());
                if (prev_vibase->deliver_idepth_pointer[prev_pointeri] != 0)
                    cur_pointeri = prev_vibase->deliver_idepth_pointer[prev_pointeri];
            }
            ASSERT(pointer2p_local.find(cur_pointeri) != pointer2p_local.end());

            int prev_idxi = prev_vibase->pointer2p_local[prev_pointeri] - shift_prev;
            int cur_idxi = pointer2p_local[cur_pointeri] - shift_cur;
            int sizei = order2size[pointer2orders[cur_pointeri]];
            ASSERT(cur_idxi >= 0);
            ASSERT(prev_idxi >= 0);

            cur_idxs.push_back(cur_idxi);
            prev_idxs.push_back(prev_idxi);
            cur_sizes.push_back(sizei);
        }

    }
}

void VisualInertialBase::UpdateLhsRhsGradientNew() {

    if (nonlinear_quantity_deliver_accum > NLQ_THRESHOLD) {
        ASSERT((int)long_feature_factors.size() - (int)deliver_idepth_pointer.size() == num_alone);

        if (prev_vibase) {
            ASSERT(cur_idxs.size());
            for (int i = 0; i < (int)cur_idxs.size(); ++i) {
                int sizei = cur_sizes[i];
                int cur_idxi = cur_idxs[i];
                int prev_idxi = prev_idxs[i];
                rhs_outside.segment(cur_idxi, sizei) += prev_vibase->rhs_outside.segment(prev_idxi, sizei);
                gradient_outside.segment(cur_idxi, sizei) += prev_vibase->gradient_outside.segment(prev_idxi, sizei);
                for (int j = 0; j < (int)cur_idxs.size(); ++j) {
                    int sizej = cur_sizes[j];
                    int cur_idxj = cur_idxs[j];
                    int prev_idxj = prev_idxs[j];
                    if (cur_idxi <= cur_idxj)
                        lhs_outside.block(cur_idxi, cur_idxj, sizei, sizej) += prev_vibase->lhs_outside.block(prev_idxi, prev_idxj, sizei, sizej);
                }
            }
        }



        if (O_BIAS0 != O_POSR) {
            //marginalize O_POSR
            int m = order2p_local[O_BIAS0] - order2p_local[O_POSR];
            int n = lhs_outside.cols() - m;

            flow_vio::MatrixXd Amm = lhs_outside.block(0, 0, m, m);
            flow_vio::MatrixXd Amn = lhs_outside.block(0, m, m, n);
            flow_vio::MatrixXd Anm_Amm_inverse(n, m);
            Amm.diagonal().array() += mu;
            Amm = InvertPSDMatrix(Amm );
            lhs_outside.block(0, 0, m, m) = Amm;

            MatrixTransposeMatrixMultiply<Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, 0>(
                Amn.data(), m, n,
                Amm.data(), m, m,
                Anm_Amm_inverse.data(), 0, 0, n, m);
            MatrixMatrixMultiply < Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, -1 > (
                Anm_Amm_inverse.data(),  n, m,
                Amn.data(), m, n,
                lhs_outside.data(), m, m, lhs_outside.rows(), lhs_outside.cols(), true);

            MatrixVectorMultiply < Eigen::Dynamic, Eigen::Dynamic, -1 > (
                Anm_Amm_inverse.data(),  n, m,
                rhs_outside.data() + 0,
                rhs_outside.data() + m);
            lhs_outside.block(0, m, m, n) = Anm_Amm_inverse.transpose();
        }


        if (num_alone) {
            //marginalize idepth alone
            int k = order2p_local[O_BIAS0] - order2p_local[O_POSR];
            int m = num_alone;
            int n = lhs_outside.cols() - m - k;
            // k,n,m

            flow_vio::MatrixXd Amm = lhs_outside.block(n + k, n + k, m, m);
            flow_vio::MatrixXd Amn = lhs_outside.block(k, n + k, n, m).transpose();
            flow_vio::MatrixXd Anm_Amm_inverse(n, m);
            Amm.diagonal().array() += IDEPTH_REGION;//mu;
            Amm = InvertPSDMatrix(Amm );
            lhs_outside.block(n + k, n + k, m, m) = Amm;

            MatrixTransposeMatrixMultiply<Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, 0>(
                Amn.data(), m, n,
                Amm.data(), m, m,
                Anm_Amm_inverse.data(), 0, 0, n, m);

            MatrixMatrixMultiply < Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, -1 > (
                Anm_Amm_inverse.data(),  n, m,
                Amn.data(), m, n,
                lhs_outside.data(), k, k, lhs_outside.rows(), lhs_outside.cols(), true);

            MatrixVectorMultiply < Eigen::Dynamic, Eigen::Dynamic, -1 > (
                Anm_Amm_inverse.data(),  n, m,
                rhs_outside.data() + n + k,
                rhs_outside.data() + k);
            lhs_outside.block(k, n + k, n, m) = Anm_Amm_inverse;

        }


        if (deliver_idepth_pointer.size()) {

            lhs_outside = lhs_outside.selfadjointView<Eigen::Upper>();
            BuildForwardMatrixInfo();


            int size = lhs_outside.cols() - num_alone - (order2p_local[O_BIAS0] - order2p_local[O_POSR]);
            int idx = order2p_local[O_BIAS0] - order2p_local[O_POSR];

            flow_vio::MatrixXd A = lhs_outside.block(idx, idx, size, size);
            flow_vio::MatrixXd D = A;
            int count = 0;
            int size2 = 0;
            ASSERT(matrix_info.size() == deliver_idepth_pointer.size() + 2);
            for (int i = 0; i < (int)matrix_info.size(); ++i) {
                auto& info = matrix_info[i];
                if (info.idx1 == info.idx2 && !info.is_identity) {
                    ASSERT(info.b_col == 1 && info.b_row == 1);
                    D.col(info.idx1) *= info.m(0);
                }
            }
            for (int i = 0; i < (int)matrix_info.size(); ++i) {
                auto& info = matrix_info[i];
                if (info.idx1 != info.idx2) {
                    matrix_update(A, info.m, D, info.idx1, info.idx2,  LEFT, info.is_identity, info.b_row, info.b_col);
                    count++;
                } else
                    size2 += info.b_col;
            }
            ASSERT(size2 == A.cols());
            ASSERT(count == 1);

            A = D;
            for (int i = 0; i < (int)matrix_info.size(); ++i) {
                auto& info = matrix_info[i];
                if (info.idx1 == info.idx2 && !info.is_identity) {
                    ASSERT(info.b_col == 1 && info.b_row == 1);
                    A.row(info.idx1) *= info.m(0);
                }
            }
            for (int i = 0; i < (int)matrix_info.size(); ++i) {
                auto& info = matrix_info[i];
                if (info.idx1 != info.idx2) {
                    matrix_update(D, info.m, A, info.idx1, info.idx2,  RIGHT, info.is_identity, info.b_row, info.b_col);
                    count++;
                }
            }
            ASSERT(count == 2);

            lhs_outside.block(idx, idx, size, size) = A;



            Eigen::VectorXd b = rhs_outside.segment(idx, size);
            Eigen::VectorXd b2 = Eigen::VectorXd::Zero(size);
            for (int i = 0; i < (int)matrix_info.size(); ++i) {
                auto& info = matrix_info[i];
                vector_update(b, info.m, b2, info.idx1, info.idx2,  RIGHT, info.is_identity, info.b_row, info.b_col);
            }

            rhs_outside.segment(idx, size) = b2;

            Eigen::VectorXd c = gradient_outside.segment(idx, size);
            Eigen::VectorXd c2 = Eigen::VectorXd::Zero(size);
            for (int i = 0; i < (int)matrix_info.size(); ++i) {
                auto& info = matrix_info[i];
                vector_update(c, info.m, c2, info.idx1, info.idx2,  RIGHT, info.is_identity, info.b_row, info.b_col);
            }

            gradient_outside.segment(idx, size) = c2;

        }


        {
            //marginalize pos0 bias0
            int k = order2p_local[O_BIAS0] - order2p_local[O_POSR];
            int m = 15;
            int n = lhs_outside.cols() - num_alone - 15 - k;


            flow_vio::MatrixXd Amm = lhs_outside.block(k, k, m, m);
            flow_vio::MatrixXd Amn = lhs_outside.block(k, k + m, m, n);
            flow_vio::MatrixXd Anm_Amm_inverse(n, m);


            Amm.diagonal().array() += mu;

            Amm = InvertPSDMatrix(Amm );
            lhs_outside.block(k, k, m, m) = Amm;

            MatrixTransposeMatrixMultiply<Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, 0>(
                Amn.data(), m, n,
                Amm.data(), m, m,
                Anm_Amm_inverse.data(), 0, 0, n, m);
            MatrixMatrixMultiply < Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, -1 > (
                Anm_Amm_inverse.data(),  n, m,
                Amn.data(), m, n,
                lhs_outside.data(), k + m, k + m, lhs_outside.rows(), lhs_outside.cols(), true);

            MatrixVectorMultiply < Eigen::Dynamic, Eigen::Dynamic, -1 > (
                Anm_Amm_inverse.data(),  n, m,
                rhs_outside.data() + k,
                rhs_outside.data() + k + m);
            lhs_outside.block(k, k + m, m, n) = Anm_Amm_inverse.transpose();
        }
        lhs_outside = lhs_outside.selfadjointView<Eigen::Upper>();

    } else
        MargRhsNew();





}

void VisualInertialBase::UpdateGaussAndGradientOutside() {

    gauss_newton_step_inside_square_norm = 0;
    gradient_dot_gauss_newton_inside = 0;
    gradient_squared_norm_inside = 0;

    gauss_outside.setZero();
    if (!next_vibase) {

        int k = order2p_local[O_BIAS0] - order2p_local[O_POSR];
        int m = 15;
        int n = lhs_outside.cols() - num_alone - 15 - k;

        flow_vio::MatrixXd A =  lhs_outside.block(k + m, k + m, n, n);

        int j_idx = 22;
        if (ESTIMATE_TD)
            j_idx += 1;
        if (ESTIMATE_EXTRINSIC)
            j_idx += 6;
        if (ESTIMATE_ACC_SCALE)
            j_idx += 3;
        if (ESTIMATE_GYR_SCALE)
            j_idx += 3;
        ASSERT(A.cols() == j_idx);

        A.diagonal().array() += mu;


        A = InvertPSDMatrix(A );

        gauss_outside.segment(k + m, n) = A * rhs_outside.segment(k + m, n);
        gradient_squared_norm_inside += gradient_outside.segment(k + m, n).squaredNorm();
        gauss_newton_step_inside_square_norm += gauss_outside.segment(k + m, n).squaredNorm();
        gradient_dot_gauss_newton_inside += gauss_outside.segment(k + m, n).dot(gradient_outside.segment(k + m, n));


    } else {
        int shift_cur = order2p_local[O_POSR];
        int shift_next = next_vibase->order2p_local[next_vibase->O_POSR];

        for (int order = O_POSK; order < O_IDEPTHR + (int)deliver_idepth_pointer.size(); ++order) {
            double* cur_pointeri = order2pointers[order];
            double* next_pointeri = cur_pointeri;
            if (order >= O_IDEPTHR) {
                ASSERT(deliver_idepth_pointer.find(cur_pointeri) != deliver_idepth_pointer.end());
                if (deliver_idepth_pointer[cur_pointeri] != 0)
                    next_pointeri = deliver_idepth_pointer[cur_pointeri];
            }
            ASSERT(next_vibase->pointer2p_local.find(next_pointeri) != next_vibase->pointer2p_local.end());
            int next_idxi = next_vibase->pointer2p_local[next_pointeri] - shift_next;
            int cur_idxi = pointer2p_local[cur_pointeri] - shift_cur;
            int sizei = order2size[pointer2orders[cur_pointeri]];
            gauss_outside.segment(cur_idxi, sizei) = next_vibase->gauss_outside.segment(next_idxi, sizei);
            gradient_outside.segment(cur_idxi, sizei) = next_vibase->gradient_outside.segment(next_idxi, sizei);
        }
    }


    {
        //recover pos0 bias0
        int k = order2p_local[O_BIAS0] - order2p_local[O_POSR];
        int m = 15;
        int n = lhs_outside.cols() - num_alone - 15 - k;

        gauss_outside.segment(k, m) = lhs_outside.block(k, k, m, m) * rhs_outside.segment(k, m) - lhs_outside.block(k, k + m, m, n) * gauss_outside.segment(k + m, n);

        gradient_squared_norm_inside += gradient_outside.segment(k, m).squaredNorm();
        gauss_newton_step_inside_square_norm += gauss_outside.segment(k, m).squaredNorm();
        gradient_dot_gauss_newton_inside += gauss_outside.segment(k, m).dot(gradient_outside.segment(k, m));

    }
    if (deliver_idepth_pointer.size()) {
        //recover transformation
        int size = lhs_outside.cols() - num_alone - (order2p_local[O_BIAS0] - order2p_local[O_POSR]);
        int idx = order2p_local[O_BIAS0] - order2p_local[O_POSR];

        {
            Eigen::VectorXd c = gauss_outside.segment(idx, size);
            Eigen::VectorXd c2 = Eigen::VectorXd::Zero(size);
            for (int i = 0; i < (int)matrix_info.size(); ++i) {
                auto& info = matrix_info[i];
                vector_update(c, info.m, c2, info.idx1, info.idx2,  LEFT, info.is_identity, info.b_row, info.b_col);
            }

            gauss_outside.segment(idx, size) = c2;
        }
        {
            Eigen::VectorXd c = gradient_outside.segment(idx, size);
            Eigen::VectorXd c2 = Eigen::VectorXd::Zero(size);
            for (int i = 0; i < (int)matrix_info.size(); ++i) {
                auto& info = matrix_info[i];
                vector_update(c, info.m, c2, info.idx1, info.idx2,  LEFT, info.is_identity, info.b_row, info.b_col);
            }

            gradient_outside.segment(idx, size) = c2;
        }
    }




    if (num_alone) {
        //recover idepth alone
        int k = order2p_local[O_BIAS0] - order2p_local[O_POSR];
        int n = lhs_outside.cols() - num_alone - k;
        int m = num_alone;
        //k,n,m

        gauss_outside.segment(k + n, m) = lhs_outside.block(k + n, k + n, m, m) * rhs_outside.segment(k + n, m) - lhs_outside.block(k + 0, k + n, n, m).transpose() * gauss_outside.segment(k + 0, n);

        gradient_squared_norm_inside += gradient_outside.segment(k + n, m).squaredNorm();
        gauss_newton_step_inside_square_norm += gauss_outside.segment(k + n, m).squaredNorm();
        gradient_dot_gauss_newton_inside += gauss_outside.segment(k + n, m).dot(gradient_outside.segment(k + n, m));
    }

    if (O_BIAS0 != O_POSR) {

        int m = order2p_local[O_BIAS0] - order2p_local[O_POSR];
        int n = lhs_outside.cols() - m;

        gauss_outside.segment(0, m) = lhs_outside.block(0, 0, m, m) * rhs_outside.segment(0, m) - lhs_outside.block(0, 0 + m, m, n) * gauss_outside.segment(0 + m, n);

        gradient_squared_norm_inside += gradient_outside.segment(0, m).squaredNorm();
        gauss_newton_step_inside_square_norm += gauss_outside.segment(0, m).squaredNorm();
        gradient_dot_gauss_newton_inside += gauss_outside.segment(0, m).dot(gradient_outside.segment(0, m));
    }

}



void VisualInertialBase::UpdateStateNew(double gauss_scale, double gradient_scale) {

    Eigen::Map<flow_vio::VectorXd>inc_new(inc_outside_pointer, gradient_outside.size());
    inc_new = gauss_outside * gauss_scale - gradient_scale * gradient_outside;


    int shift1 = order2p_local[O_POSR];
    for (int order = O_POSR; order < O_IDEPTHR; ++order) {
        int local_size = order2size[order];
        int inc_index = order2p_local[order] - shift1;
        if (order < O_IDEPTHR)
            memcpy(inc_posbias_pointer + order2p_local[order], inc_outside_pointer + inc_index, local_size * sizeof(double));
    }

    std::vector<double*>update_pointers;
    if (!next_vibase) {
        //update O_POSK,O_BIASK
        for (int order = O_POSK; order < O_IDEPTHR; ++order)
            update_pointers.push_back(order2pointers[order]);
        ASSERT(deliver_idepth_pointer.size() == 0);
    }

    for (int order = O_POSR; order < O_POSK; ++order)
        update_pointers.push_back(order2pointers[order]);

    //update idepth alone
    for (auto& it : long_feature_factors) {
        double* p_idepth = it->p_idepth;
        if (it->pindexs[0] == 0)
            update_pointers.push_back(p_idepth);
    }



    int shift_cur = order2p_local[O_POSR];
    for (int i = 0; i < (int)update_pointers.size(); ++i) {
        double* addr = update_pointers[i];
        int size = globalSize(order2size[pointer2orders[addr]]);
        int index = pointer2p_local[addr] - shift_cur;
        ASSERT(index >= 0);
        double* inc_addr = inc_new.data() + index;
        ASSERT(size != 0);
        if (size != 7) {
            Eigen::Map<Eigen::VectorXd>addr0(addr, size);
            Eigen::Map<Eigen::VectorXd>inc0(inc_addr, size);
            addr0 -= inc0;
        } else {
            Eigen::Map<Eigen::Vector3d> p(addr);
            Eigen::Map<Eigen::Quaterniond> q(addr + 3);
            Eigen::Map<Eigen::Vector3d> dp(inc_addr);
            Eigen::Quaterniond dq = Utility::deltaQ(-Eigen::Map<Eigen::Vector3d>(inc_addr + 3));
            p = p - dp;
            q = (q * dq).normalized();
        }
    }
}

void VisualInertialBase::ForwardDeliverIdepthValues() {
    setRPwc();

    if (deliver_idepth_pointer.size()) {
        for (auto& it : long_feature_factors) {
            double* p_idepthi = it->p_idepth;
            auto& pindexs = it->pindexs;
            ASSERT(p_idepthi[0]);
            if (deliver_idepth_pointer.find(p_idepthi) != deliver_idepth_pointer.end()) {
                if (pindexs[0] == 0) {
                    double* p_idepthj = deliver_idepth_pointer[p_idepthi];
                    ASSERT((int)pindexs.size() >= SWF_SIZE_IN + 1);
                    ASSERT(pindexs[pindexs.size() - 1] == SWF_SIZE_IN);
                    ASSERT((int)it->ptss.size() >= SWF_SIZE_IN + 1);
                    ASSERT(p_idepthj);
                    ASSERT(p_idepthj[0]);
                    ASSERT(p_idepthi);
                    ASSERT(p_idepthi[0]);
                    Eigen::Vector3d pts_i = it->ptss[0];
                    Eigen::Vector3d pts_camera_j = Rwcs[SWF_SIZE_IN].transpose() * (Rwcs[0] * pts_i / p_idepthi[0] + Pwcs[0] - Pwcs[SWF_SIZE_IN]);
                    p_idepthj[0] = 1. / pts_camera_j.z();

                } else {
                    ASSERT(pindexs[0] == SWF_SIZE_IN);
                    ASSERT(deliver_idepth_pointer[p_idepthi] == 0);
                }
            }
        }
    }
}



void VisualInertialBase::GetMatrixNext(flow_vio::MatrixXd& lhs, Eigen::VectorXd& rhs, std::vector<double*>& new_parameters, std::vector<int>& new_sizes) {

    int shift1 = order2p_local[O_POSK] - order2p_local[O_POSR];

    int size = lhs_outside.cols() - num_alone - shift1;

    lhs = lhs_outside.block(shift1, shift1, size, size);
    rhs = rhs_outside.segment(shift1, size);
    lhs = lhs.selfadjointView<Eigen::Upper>();

    for (int order = O_POSK; order < O_IDEPTHR + (int)deliver_idepth_pointer.size(); order++) {
        double* old_pointer = order2pointers[order];
        double* new_pointer = old_pointer;
        if (order >= O_IDEPTHR && deliver_idepth_pointer.find(old_pointer) != deliver_idepth_pointer.end() && deliver_idepth_pointer[old_pointer] != 0)
            new_pointer = deliver_idepth_pointer[old_pointer];
        new_parameters.push_back(new_pointer); new_sizes.push_back(globalSize(order2size[order]));
    }

}
