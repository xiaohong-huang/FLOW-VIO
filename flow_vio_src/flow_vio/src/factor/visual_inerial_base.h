
#pragma once
#include "../parameter/parameters.h"
#include <eigen3/Eigen/Dense>
#include "integration_base.h"

#include"imu_factor.h"
#include "marginalization_factor.h"
#include "loss_function.h"
#include <omp.h>
#include "sparse_matrix.h"





class IMUPreFactor {
  public:
    ~IMUPreFactor() {
        delete factor;
    }
    std::vector<double*>parameters;
    uint8_t pindexi, pindexj;
    IMUFactor* factor = 0;
    std::vector<flow_vio::MatrixXd > imu_jacobians_in;
    flow_vio::Vector15d  imu_residual_in;
    flow_vio::Vector15d  imu_residual_in_save;
    flow_vio::Vector15d  model_residual_accum;
    flow_vio::Vector15d  model_residual_accum_save;
    flow_vio::Vector15d  imu_residual_in_linerized;

};


class VisualFactor {
  public:

    std::set<uint8_t>pindexs_set;
    std::vector<uint8_t> pindexs;
    std::vector<Eigen::Vector3d> ptss;
    double* lhs_pos_idepth;
    double rhs_idepth[1];
    double lhs_idepth[1];
    double rhs_idepth_save[1];
    double* p_idepth;
    uint8_t min_pindex;
};

class VisualInertialBase  {
  public:
    VisualInertialBase() {
        cauchy_loss_function = new CauchyLoss(1.0);
    };
    ~VisualInertialBase();
    void AddVisualFactorShort(int feature_id, double*, double*, double*, uint8_t, uint8_t, Eigen::Vector3d, Eigen::Vector3d);
    void AddVisualFactorLong(int feature_id, double*, double*, double*, uint8_t, uint8_t, Eigen::Vector3d, Eigen::Vector3d);
    void AddIMUFactor(IMUPreFactor* imu_pre_factor);

    void ConstructBiasConnections();
    void ConstructOrderingMap();
    void ResetMem();
    void PrepareMatrix();
    void FactorUpdates();

    void VisualJacobianResidualUpdatelhsRhs(uint8_t pindexi, uint8_t pindexj,
                                            Eigen::Vector3d pts_i, Eigen::Vector3d pts_j,
                                            double* lhs_pos_idepth,
                                            double* rhs_idepth,
                                            double* lhs_idepth,
                                            LossFunction* loss_function0,
                                            double* jacobian_pointer,
                                            double* visual_residual_raw_in, double visual_cost,
                                            uint8_t min_pindex,
                                            double* p_idepth
                                           );

    void EvaluateIdepthsShort();
    void EvaluateIdepthsLong();

    void IMUJacobianResidualUpdatelhsRhs();

    void MargIdepth(const std::set<uint8_t>& pindexs, double* lhs_pos_idepth, double* rhs_idepth, double* lhs_idepth, uint8_t);
    void MargeIdepthsShort();
    void MargInsideBias();
    void MargInsidePos();

    void UpdateMargposUseGradientAndGauss(double gradient_scale, double gauss_scale);
    void UpdateMargBiasUseGradientAndGauss(double gradient_scale, double gauss_scale);
    void UpdateMargFeatureUseGradientAndGauss(double gradient_scale, double gauss_scale);
    void UpdateInsideStateUseGradientAndGauss(double gradient_scale, double gauss_scale);

    void UpdateInsideGaussStep();
    void UpdateMargposGaussStep();
    void UpdateMargBiasGaussStep();
    void UpdateMargFeatureGaussStep();

    void SaveOrRestoreHidenStates(bool );

    void EvaluateCost(double* cost, double* model_cost_change);
    void EvaluateAlpha(double* alpha1);

    void SaveGradientOutside();
    void SaveGaussStepOutside();
    void SaveGaussStep();

    int GetBIndex(int);
    int GetPIndex(int);

    bool EvaluateLhsRhs(double const* const* parameters, double* rhs, double* lhs, double* gradient, double mu);

    void SaveOutsidePointerRaw();
    void setRPwc();

    void VisualEvaluate(int pindexi, int pindexj,
                        double* parameters, double* residuals, double** jacobians,
                        const Eigen::Vector3d pts_i, const Eigen::Vector3d pts_j);
    void EvaluateLastMargInfo();
    void AddLastMargeInfo(MarginalizationInfo* last_marg_info_);
    void ConstructPriorIdx();
    void EvaluatePriorAlpha(double* alpha1);
    void EvaluatePriorModelCostChange(double* model_cost_change);
    void EvaluatePriorCost(double* cost);

    void UpdateGaussAndGradientOutside();
    void UpdateLhsRhsGradientNew();
    void ForwardDeliverIdepthValues();
    void UpdateStateNew(double gauss_scale, double gradient_scale);

    void RemoveFeature(int feature_id);
    void SaveLoadCandidateResidual(bool is_save);
    void ComputeRhs();
    void VisualEvaluateRhs(uint8_t pindexi, uint8_t pindexj,
                           double* rhs_idepth,
                           double* jacobian_pointer,
                           double* visual_residual_raw_in);
    void MargeRhs();
    void EvaluateNonlinearQuantity();
    void MargRhsNew();

    void BuildForwardMatrixInfo();
    void BuildCurPrevIdx();
    void GetMatrixNext(flow_vio::MatrixXd& lhs, Eigen::VectorXd& rhs, std::vector<double*>& new_parameters, std::vector<int>& new_sizes);
    void Reset() {
        if (memory) {
            delete memory; memory = 0;
        }
    }
    void ResetInit() {
        init = false;
        if (next_vibase)next_vibase->ResetInit();
    }

    int visual_residual_short_num;
    int visual_residual_long_num;
    int visual_lhs_count;

    flow_vio::VectorXd prior_residual;
    flow_vio::VectorXd prior_residual_save;

    std::map<int, VisualFactor*, less<int>>idepth_map_factors_short;
    std::map<int, VisualFactor*, less<int>>idepth_map_factors_long;
    std::map<int, IMUPreFactor*, less<int>>imu_map_factors;
    std::vector<VisualFactor*>long_feature_factors;

    std::vector<flow_vio::MatrixXd > imu_jacobians_in;
    double* imu_jacobians_raw_in[10] = {0};
    flow_vio::Vector15d  imu_residual_in;
    double* imu_residual_raw_in;

    LossFunction* cauchy_loss_function = 0;

    double* memory = 0;

    double* lhs_pos_idepth_global;
    double* old_estimations_pointer;

    double* visual_jacobians;
    double* visual_residuals;
    double* visual_costs;
    double* visual_residuals_save;

    double* visual_model_residual_accum;
    double* visual_model_residual_accum_save;
    double* visual_residuals_linerized;

    std::vector<double*> lhs_posbias;
    double* rhs_posbias_pointer;

    double* rhs_posbias_save_pointer;

    double* lhs_margpos_outside_pointer;
    double* lhs_margpos_pointer;
    double* rhs_margpos_pointer;

    double* inc_posbias_pointer;
    double* inc_idepth_short_pointer;
    double* inc_outside_pointer;
    double* gauss_posbias_pointer;
    double* gauss_idepth_short_pointer;

    double* idepth_short_save_pointer;
    double* para_posebias_inside_save_pointer;

    std::vector<double*>outside_pointer_raw;

    std::unordered_map<int, double*>pos_index2pointers;
    std::unordered_map<double*, int>pointer2orders;
    std::unordered_map<int, double*>order2pointers;
    std::unordered_map<double*, int>pos_pointer2indexs;
    std::unordered_map<int, double*>bias_index2pointers;
    std::vector<Eigen::Matrix3d>Rwcs;
    std::vector<Eigen::Vector3d>Pwcs;


    std::vector<uint8_t> pindex2order;
    std::vector<uint8_t> bindex2order;
    std::vector<uint8_t> order2size;
    std::vector<int> order2p_local;
    std::vector<int> order2p_global;
    std::vector<std::vector<int>> connections_bias;

    double gradient_squared_norm_inside;
    double gradient_dot_gauss_newton_inside;
    double gauss_newton_step_inside_square_norm;


    int marg_count;

    int bias_para_count;
    int pos_para_count;
    int para_count;

    int idepth_long_count;

    double mu = 0;


    int O_MARG_BIAS;
    int O_MARG_POS;
    int O_POSR;
    int O_BIAS0;
    int O_POS0;
    int O_POSK;
    int O_BIASK;
    int O_POSR_NEXT;
    int O_SCALE;
    int O_POSGLOBAL;
    int O_TD;
    int O_EXTRINSIC;
    int O_ACC_S;
    int O_GYR_S;
    int O_IDEPTHR;
    int O_FULL;
    int E_O_MARG_BIAS;
    int E_O_MARG_POS;

    MarginalizationInfo* last_marg_info = 0;
    flow_vio::VectorXd prior_rhs;
    flow_vio::VectorXd prior_rhs_save;

    Eigen::Matrix<double, 1, 6, Eigen::RowMajor> yaw_constraint_jacobian;
    flow_vio::Vector1d yaw_constraint_residual;
    flow_vio::Vector1d yaw_constraint_residual_save;

    flow_vio::Vector1d yaw_constraint_model_residual_accum;
    flow_vio::Vector1d yaw_constraint_residual_linerized;
    flow_vio::Vector1d yaw_constraint_model_residual_accum_save;

    Eigen::Vector3d InitMag;

    std::vector<int>prior_idx;

    VisualInertialBase* prev_vibase = 0;
    VisualInertialBase* next_vibase = 0;
    std::unordered_map<double*, double*>deliver_idepth_pointer;//1
    std::unordered_map<double*, int>pointer2p_local;

    Eigen::VectorXd gauss_outside;
    flow_vio::MatrixXd lhs_outside;
    Eigen::VectorXd rhs_outside;
    Eigen::VectorXd gradient_outside;

    int num_alone = 0;
    std::vector<MatirxInfo>matrix_info;
    std::vector<int>cur_idxs, prev_idxs, cur_sizes;
    int outside_threshold;
    bool ready_for_sequential_forward = false;
    bool ready_for_sequential_backward = false;

    double* global_pos_pointer = 0;
    double* scale_pointer = 0;
    double distance0;
    Eigen::VectorXd init_distance_constraint_jacobian;
    double init_distance_constraint_residual;
    double init_distance_constraint_residual_save;

    double init_distance_constraint_model_residual_accum;
    double init_distance_constraint_residual_linerized;
    double init_distance_constraint_model_residual_accum_save;
    double last_cost=0;
    double* td_pointer = 0;
    double* extrinsic_pointer = 0;
    bool init = false;


    double nonlinear_quantity;
    bool history_flag;
    double nonlinear_quantity_deliver_accum = 1e8;
    double* acc_scale_pointer = 0;
    double* gyr_scale_pointer = 0;

};


extern VisualInertialBase* VI_info_pointer;
extern VisualInertialBase* VI_info_pointer_head;
extern VisualInertialBase* VI_info_pointer_tail;
