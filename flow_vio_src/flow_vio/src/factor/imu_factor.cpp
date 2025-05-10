#include"imu_factor.h"
#include "../utility/utility.h"
#include "../parameter/parameters.h"



// ----------------------------------------------------------------------------------------------------------------------
bool IMUFactor::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
    if (parameters == 0) {
        residuals[0] = 0;
        return false;
    };
#define LEN 6
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]);
    Eigen::Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]);

    Eigen::Vector3d Pj(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond Qj(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    Eigen::Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
    Eigen::Vector3d Baj(parameters[3][3], parameters[3][4], parameters[3][5]);
    Eigen::Vector3d Bgj(parameters[3][6], parameters[3][7], parameters[3][8]);

    double scale_factor = parameters[4][0];

    Eigen::Vector3d P0(parameters[5][0], parameters[5][1], parameters[5][2]);
    Eigen::Quaterniond Q_WI_WC(parameters[5][6], parameters[5][3], parameters[5][4], parameters[5][5]);
    ASSERT(dt_i != 0 && dt_j != 0);

    int p_idx = 6;
    double td = 0;
    if (ESTIMATE_TD) {
        td = parameters[p_idx][0]; p_idx++;
    }



    Eigen::Vector3d tic;
    Eigen::Quaterniond qic;
    Eigen::Matrix3d ric;
    Eigen::Vector3d tci;

    if (ESTIMATE_EXTRINSIC) {
        tic = Eigen::Vector3d(parameters[p_idx][0], parameters[p_idx][1], parameters[p_idx][2]);
        qic = Eigen::Quaterniond(parameters[p_idx][6], parameters[p_idx][3], parameters[p_idx][4], parameters[p_idx][5]);
        ric = qic.toRotationMatrix();
        tci = -(ric.transpose() * tic);
        p_idx++;
    } else {
        tic = TIC[0];
        qic = QIC[0];
        ric = RIC[0];
        tci = -(RIC[0].transpose() * TIC[0]);
    }
    Eigen::Vector3d acc_scale = Eigen::Vector3d({1, 1, 1});
    Eigen::Vector3d gyr_scale = Eigen::Vector3d({1, 1, 1});

    if (ESTIMATE_ACC_SCALE) {
        acc_scale = Eigen::Vector3d(parameters[p_idx][0], parameters[p_idx][1], parameters[p_idx][2]);
        p_idx++;
    }
    if (ESTIMATE_GYR_SCALE) {
        gyr_scale = Eigen::Vector3d(parameters[p_idx][0], parameters[p_idx][1], parameters[p_idx][2]);
        p_idx++;
    }




    Eigen::Matrix3d dp_dba = pre_integration->jacobian.block<3, 3>(O_P, O_BA);
    Eigen::Matrix3d dp_dbg = pre_integration->jacobian.block<3, 3>(O_P, O_BG);
    Eigen::Matrix3d dq_dbg = pre_integration->jacobian.block<3, 3>(O_R, O_BG);
    Eigen::Matrix3d dv_dba = pre_integration->jacobian.block<3, 3>(O_V, O_BA);
    Eigen::Matrix3d dv_dbg = pre_integration->jacobian.block<3, 3>(O_V, O_BG);
    Eigen::Matrix<double, 15, 15> sqrt_info = pre_integration->get_sqrtinfo();
    double sum_dt = pre_integration->sum_dt;


    Eigen::Matrix3d Ri = Qi.toRotationMatrix();
    Eigen::Matrix3d Rj = Qj.toRotationMatrix();
    Eigen::Matrix3d RIC_Ri_inv = ric * Ri.transpose();
    Eigen::Quaterniond Qij = Qi.inverse() * Qj;

    Eigen::Matrix3d R_WI_WC = Q_WI_WC.toRotationMatrix();
    Eigen::Vector3d newG =  R_WI_WC.transpose() * G;
    Eigen::Vector3d newVi =  R_WI_WC.transpose() * Vi;
    Eigen::Vector3d newVj =  R_WI_WC.transpose() * Vj;

    Eigen::Vector3d un_gyri = (pre_integration->gyri.array() * gyr_scale.array()).matrix() - Bgi;
    Eigen::Vector3d un_gyrj = (pre_integration->gyrj.array() * gyr_scale.array()).matrix() - Bgj;


    double new_dt_i;
    double new_dt_j;

    if (ESTIMATE_TD) {
        new_dt_i = dt_i + td;
        new_dt_j = dt_j + td;
    } else {
        new_dt_i = dt_i;
        new_dt_j = dt_j;
    }



    Eigen::Quaterniond dQti(1, un_gyri(0) * new_dt_i / 2, un_gyri(1) *new_dt_i / 2, un_gyri(2) * new_dt_i / 2);
    Eigen::Quaterniond dQtj(1, un_gyrj(0) * new_dt_j / 2, un_gyrj(1) *new_dt_j / 2, un_gyrj(2) * new_dt_j / 2);
    Eigen::Matrix3d dRti = dQti.toRotationMatrix();


    Eigen::Vector3d dP = 0.5 * newG * sum_dt * sum_dt +  scale_factor * (Pj - Pi) + newVj * new_dt_j - newVi * new_dt_i - newVi * sum_dt + Rj * tci;
    Eigen::Vector3d dV = newG * sum_dt + newVj  - newVi;

    Eigen::Vector3d dba = Bai - pre_integration->linearized_ba;
    Eigen::Vector3d dbg = Bgi - pre_integration->linearized_bg;

    Eigen::Vector3d dacc_scale = acc_scale - pre_integration->acc_scale;
    Eigen::Vector3d dgyr_scale = gyr_scale - pre_integration->gyr_scale;

    Eigen::Matrix3d dp_dacc_scale = pre_integration->jacobian_s.block<3, 3>(O_P, 0);
    Eigen::Matrix3d dp_dgyr_scale = pre_integration->jacobian_s.block<3, 3>(O_P, 3);
    Eigen::Matrix3d dq_dgyr_scale = pre_integration->jacobian_s.block<3, 3>(O_R, 3);
    Eigen::Matrix3d dv_dacc_scale = pre_integration->jacobian_s.block<3, 3>(O_V, 0);
    Eigen::Matrix3d dv_dgyr_scale = pre_integration->jacobian_s.block<3, 3>(O_V, 3);

    Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * dbg + dq_dgyr_scale * dgyr_scale);
    Eigen::Vector3d corrected_delta_v = pre_integration->delta_v + dv_dba * dba + dv_dbg * dbg + dv_dacc_scale * dacc_scale + dv_dgyr_scale * dgyr_scale;
    Eigen::Vector3d corrected_delta_p = pre_integration->delta_p + dp_dba * dba + dp_dbg * dbg + dp_dacc_scale * dacc_scale + dp_dgyr_scale * dgyr_scale;


    if (residuals) {

        Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
        residual.block<3, 1>(O_P, 0) = dRti.transpose() * RIC_Ri_inv * dP + dRti.transpose() * tic - corrected_delta_p;
        residual.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * dQti.inverse() * qic * Qij * qic.inverse() * dQtj).vec();
        residual.block<3, 1>(O_V, 0) = dRti.transpose() * RIC_Ri_inv * dV - corrected_delta_v;
        residual.block<3, 1>(O_BA, 0) = Baj - Bai;
        residual.block<3, 1>(O_BG, 0) = Bgj - Bgi;
        residual = sqrt_info * residual;
    }


    if (jacobians) {

        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 15, LEN, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
            jacobian_pose_i.setZero();
            jacobian_pose_i.block<3, 3>(O_P, O_P) = -(dRti.transpose() * RIC_Ri_inv * scale_factor);
            jacobian_pose_i.block<3, 3>(O_P, O_R) = dRti.transpose() * ric * Utility::skewSymmetric(Ri.transpose() * dP);
            jacobian_pose_i.block<3, 3>(O_R, O_R) =
                -(Utility::Qleft(dQtj.inverse() * qic * Qij.inverse()) * Utility::Qright(qic.inverse() * dQti * corrected_delta_q)).bottomRightCorner<3, 3>();
            jacobian_pose_i.block<3, 3>(O_V, O_R) = dRti.transpose() * ric * Utility::skewSymmetric(Ri.transpose() * dV);
            jacobian_pose_i = sqrt_info * jacobian_pose_i;
        }
        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_i(jacobians[1]);
            jacobian_speedbias_i.setZero();
            jacobian_speedbias_i.block<3, 3>(O_P, O_V - O_V) = -(dRti.transpose() * RIC_Ri_inv * R_WI_WC.transpose() * (sum_dt + new_dt_i));
            jacobian_speedbias_i.block<3, 3>(O_P, O_BA - O_V) = -dp_dba;
            jacobian_speedbias_i.block<3, 3>(O_P, O_BG - O_V) = -dp_dbg - Utility::skewSymmetric(dRti.transpose() * RIC_Ri_inv * dP * new_dt_i) - Utility::skewSymmetric(dRti.transpose() * tic * new_dt_i);
            jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) =
                -(Utility::Qleft(dQtj.inverse() * qic * Qij.inverse() * qic.inverse() * dQti * pre_integration->delta_q).bottomRightCorner<3, 3>() * dq_dbg)
                + (new_dt_i * Utility::Qleft(corrected_delta_q.inverse()) * Utility::Qright(dQti.inverse() * qic * Qij * qic.inverse() * dQtj)).bottomRightCorner<3, 3>();

            jacobian_speedbias_i.block<3, 3>(O_V, O_V - O_V) = -(dRti.transpose() * RIC_Ri_inv * R_WI_WC.transpose());
            jacobian_speedbias_i.block<3, 3>(O_V, O_BA - O_V) = -dv_dba;
            jacobian_speedbias_i.block<3, 3>(O_V, O_BG - O_V) = -dv_dbg - Utility::skewSymmetric(dRti.transpose() * RIC_Ri_inv * dV * new_dt_i);
            jacobian_speedbias_i.block<3, 3>(O_BA, O_BA - O_V) = -Eigen::Matrix3d::Identity();
            jacobian_speedbias_i.block<3, 3>(O_BG, O_BG - O_V) = -Eigen::Matrix3d::Identity();
            jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;
        }
        if (jacobians[2]) {
            Eigen::Map<Eigen::Matrix<double, 15, LEN, Eigen::RowMajor>> jacobian_pose_j(jacobians[2]);
            jacobian_pose_j.setZero();
            jacobian_pose_j.block<3, 3>(O_P, O_P) = dRti.transpose() * RIC_Ri_inv * scale_factor;
            jacobian_pose_j.block<3, 3>(O_P, O_R) = -(dRti.transpose() * RIC_Ri_inv * Rj * Utility::skewSymmetric(tci));
            jacobian_pose_j.block<3, 3>(O_R, O_R) = (Utility::Qright(qic.inverse() * dQtj) * Utility::Qleft(corrected_delta_q.inverse() * dQti.inverse() * qic * Qij)).bottomRightCorner<3, 3>();
            jacobian_pose_j = sqrt_info * jacobian_pose_j;
        }
        if (jacobians[3]) {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_j(jacobians[3]);
            jacobian_speedbias_j.setZero();
            jacobian_speedbias_j.block<3, 3>(O_P, O_V - O_V) = dRti.transpose() * RIC_Ri_inv * R_WI_WC.transpose() * new_dt_j;
            jacobian_speedbias_j.block<3, 3>(O_R, O_BG - O_V) = -(new_dt_j * Utility::Qleft(corrected_delta_q.inverse()  * dQti.inverse() * qic * Qij * qic.inverse() * dQtj)).bottomRightCorner<3, 3>(); //待定
            jacobian_speedbias_j.block<3, 3>(O_V, O_V - O_V) = dRti.transpose() * RIC_Ri_inv * R_WI_WC.transpose();
            jacobian_speedbias_j.block<3, 3>(O_BA, O_BA - O_V) = Eigen::Matrix3d::Identity();
            jacobian_speedbias_j.block<3, 3>(O_BG, O_BG - O_V) = Eigen::Matrix3d::Identity();
            jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;
        }
        if (jacobians[4]) {
            Eigen::Map<flow_vio::Vector15d > jacobian_scale(jacobians[4]);
            jacobian_scale.setZero();
            if (fix_scale) {
                jacobian_scale.segment(O_P, 3) = dRti.transpose() * RIC_Ri_inv * (Pj - Pi);
                jacobian_scale = sqrt_info * jacobian_scale;
            }
        }
        if (jacobians[5]) {
            Eigen::Map<Eigen::Matrix<double, 15, LEN, Eigen::RowMajor>> jacobian_pose_0(jacobians[5]);
            jacobian_pose_0.setZero();
            jacobian_pose_0.block<3, 3>(O_P, O_R) =
                dRti.transpose() * RIC_Ri_inv * Utility::skewSymmetric(newG * 0.5 * sum_dt * sum_dt - newVi * sum_dt + newVj * new_dt_j - newVi * new_dt_i);
            jacobian_pose_0.block<3, 3>(O_V, O_R) = dRti.transpose() * RIC_Ri_inv * Utility::skewSymmetric(dV);
            jacobian_pose_0 = sqrt_info * jacobian_pose_0;
        }
        int j_idx = 6;
        if (ESTIMATE_TD) {
            if (jacobians[j_idx]) {
                Eigen::Map<flow_vio::Vector15d > jacobian_dt(jacobians[j_idx]);
                jacobian_dt.setZero();
                jacobian_dt.segment(O_P, 3) = dRti.transpose() * RIC_Ri_inv * (newVj - newVi) + Utility::skewSymmetric(dRti.transpose() * RIC_Ri_inv * dP + dRti.transpose() * tic) * un_gyri;
                Eigen::Quaterniond a(0, -un_gyri(0), -un_gyri(1), -un_gyri(2) );
                Eigen::Quaterniond b(0, un_gyrj(0), un_gyrj(1), un_gyrj(2) );
                Eigen::Quaterniond c = dQti.inverse() * qic * Qij * qic.inverse() * dQtj;
                jacobian_dt.segment(O_R, 3) =  (corrected_delta_q.inverse() * a * c).vec() + (corrected_delta_q.inverse() * c * b).vec();
                jacobian_dt.segment(O_V, 3) = Utility::skewSymmetric(dRti.transpose() * RIC_Ri_inv * dV) * un_gyri;
                jacobian_dt = sqrt_info * jacobian_dt;
            }
            j_idx++;
        }
        if (ESTIMATE_EXTRINSIC) {
            if (jacobians[j_idx]) {
                Eigen::Map<Eigen::Matrix<double, 15, LEN, Eigen::RowMajor>> jacobian_extrinsic(jacobians[j_idx]);
                jacobian_extrinsic.setZero();
                jacobian_extrinsic.block<3, 3>(O_P, O_P) = -(dRti.transpose() * RIC_Ri_inv * Rj * ric.transpose()) + dRti.transpose();
                jacobian_extrinsic.block<3, 3>(O_P, O_R) = -(dRti.transpose() * ric * Utility::skewSymmetric(Ri.transpose() * (0.5 * newG * sum_dt * sum_dt +  scale_factor * (Pj - Pi) + newVj * new_dt_j - newVi * new_dt_i - newVi * sum_dt ))) +
                                                           dRti.transpose() * ric * Utility::skewSymmetric(Ri.transpose() * Rj * ric.transpose() * tic) - (dRti.transpose() * RIC_Ri_inv * Rj * Utility::skewSymmetric(ric.transpose() * tic));
                jacobian_extrinsic.block<3, 3>(O_R, O_R) = -2 * ((   Utility::Qleft(corrected_delta_q.inverse() * dQti.inverse() * qic) * Utility::Qright(qic.inverse() * dQtj)).bottomRightCorner<3, 3>() * Utility::skewSymmetric(Qij.vec()));
                jacobian_extrinsic.block<3, 3>(O_V, O_R) = dRti.transpose() * ric * Utility::skewSymmetric(-(Ri.transpose() * dV));
                jacobian_extrinsic = sqrt_info * jacobian_extrinsic;
            }
            j_idx++;
        }
        if (ESTIMATE_ACC_SCALE) {
            if (jacobians[j_idx]) {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_acc_scale(jacobians[j_idx]);
                jacobian_acc_scale.setZero();
                if ( has_excitation) {
                    ASSERT((pre_integration->jacobian_s.block<3, 3>(O_R, 0)).norm() == 0);
                    ASSERT((pre_integration->jacobian_s.block<3, 3>(O_BA, 0)).norm() == 0);
                    ASSERT((pre_integration->jacobian_s.block<3, 3>(O_BG, 0)).norm() == 0);
                    jacobian_acc_scale.block<3, 3>(O_P, 0) = -dp_dacc_scale;
                    jacobian_acc_scale.block<3, 3>(O_V, 0) = -dv_dacc_scale;
                    ASSERT((jacobian_acc_scale + (pre_integration->jacobian_s.block<15, 3>(0, 0))).norm() == 0);
                    jacobian_acc_scale = sqrt_info * jacobian_acc_scale;
                }
            }
            j_idx++;
        }
        if (ESTIMATE_GYR_SCALE) {
            if (jacobians[j_idx]) {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_gyr_scale(jacobians[j_idx]);
                jacobian_gyr_scale.setZero();
                if ( has_excitation) {
                    jacobian_gyr_scale.block<3, 3>(O_R, 0) =
                        -(Utility::Qleft(dQtj.inverse() * qic * Qij.inverse() * qic.inverse() * dQti * pre_integration->delta_q).bottomRightCorner<3, 3>() * dq_dgyr_scale)
                        - ((new_dt_i * Utility::Qleft(corrected_delta_q.inverse()) * Utility::Qright(dQti.inverse() * qic * Qij * qic.inverse() * dQtj)).bottomRightCorner<3, 3>() * (pre_integration->gyri).asDiagonal())
                        + (new_dt_j * Utility::Qleft(corrected_delta_q.inverse()  * dQti.inverse() * qic * Qij * qic.inverse() * dQtj)).bottomRightCorner<3, 3>() * (pre_integration->gyrj).asDiagonal(); //待定
                    jacobian_gyr_scale.block<3, 3>(O_P, 0) =
                        -dp_dgyr_scale
                        + (Utility::skewSymmetric(dRti.transpose() * RIC_Ri_inv * dP * new_dt_i) + Utility::skewSymmetric(dRti.transpose() * tic * new_dt_i)) * (pre_integration->gyri).asDiagonal();
                    jacobian_gyr_scale.block<3, 3>(O_V, 0) =
                        -dv_dgyr_scale
                        +  Utility::skewSymmetric(dRti.transpose() * RIC_Ri_inv * dV * new_dt_i) * (pre_integration->gyri).asDiagonal();

                    jacobian_gyr_scale = sqrt_info * jacobian_gyr_scale;
                }
            }
            j_idx++;
        }

    }
#undef LEN

    return true;


}

