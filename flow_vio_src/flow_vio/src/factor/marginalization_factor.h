

#pragma once

#include <cstdlib>
#include <pthread.h>
#include <unordered_map>
#include "../parameter/parameters.h"
//very poor
#include "../solver/cost_function.h"
#include "loss_function.h"

struct ResidualBlockInfo {
    ResidualBlockInfo(CostFunction* _cost_function, LossFunction* _loss_function, std::vector<double*> _parameter_blocks, std::vector<int> _drop_set, std::vector<int> _sparse_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set), sparse_set(_sparse_set) {}

    void Evaluate();

    CostFunction* cost_function;
    LossFunction* loss_function;
    std::vector<double*> parameter_blocks;
    std::vector<int> drop_set;
    std::vector<int> sparse_set;

    double** raw_jacobians;
    std::vector<flow_vio::MatrixXd > jacobians;
    Eigen::VectorXd residuals;


};

struct ThreadsStruct {
    std::vector<ResidualBlockInfo*> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<double*, int> parameter_block_size; //global size
    std::unordered_map<double*, int> parameter_block_idx; //local size
};

class MarginalizationInfo {
  public:
    MarginalizationInfo() {};
    MarginalizationInfo(MarginalizationInfo* old) {
        keep_block_size = old->keep_block_size;
        keep_block_idx = old->keep_block_idx;
        keep_block_data = old->keep_block_data;
        keep_block_addr = old->keep_block_addr;
        A = old->A;
        b = old->b;
        n = old->n;
        m = old->m;
    }
    void setmarginalizeinfo(std::vector<double*>& parameter_block_addr,
                            std::vector<int>& parameter_block_global_size, Eigen::MatrixXd A_, Eigen::VectorXd& b_,
                            bool Sqrt);
    ~MarginalizationInfo();
    void addResidualBlockInfo(ResidualBlockInfo* residual_block_info);
    void marginalize(bool initialinformation);
    void getParameterBlocks();
    void resetLinerizationPoint();
    void GetJacobianAndResidual();
    void marginalize_pointers(std::set<double*>feature_marge_pointer);

    int m, n;
    double cost0;

    std::vector<ResidualBlockInfo*> factors;
    std::unordered_map<double*, int> parameter_block_size; //global size
    std::unordered_map<double*, int> parameter_block_idx; //local size
    std::unordered_map<double*, int> parameter_block_sparse_idx; //local size
    std::unordered_map<double*, int> parameter_block_drop_idx; //local size
    std::unordered_map<double*, double*> parameter_block_data;
    std::vector<int> keep_block_size; //global size
    std::vector<int> keep_block_idx;  //local size
    std::vector<double*> keep_block_data;
    std::vector<double*> keep_block_addr;
    std::unordered_map<double*, int>keep_block_addr_set;
    


    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;

    
};

