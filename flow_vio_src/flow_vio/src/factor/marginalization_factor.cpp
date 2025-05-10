

#include "marginalization_factor.h"
#include "../parameter/parameters.h"
#include "../utility/utility.h"
#include "../utility/tic_toc.h"

void ResidualBlockInfo::Evaluate() {
    residuals.resize(cost_function->num_residuals());

    std::vector<int> block_sizes = cost_function->parameter_block_sizes();
    raw_jacobians = new double *[block_sizes.size()];
    jacobians.resize(block_sizes.size());

    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++) {
        jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);
        raw_jacobians[i] = jacobians[i].data();
    }
    cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);


    if (loss_function) {
        double residual_scaling_, alpha_sq_norm_;

        double sq_norm, rho[3];

        sq_norm = residuals.squaredNorm();
        loss_function->Evaluate(sq_norm, rho);
        double sqrt_rho1_ = sqrt(rho[1]);
        if ((sq_norm == 0.0) || (rho[2] <= 0.0)) {
            residual_scaling_ = sqrt_rho1_;
            alpha_sq_norm_ = 0.0;
        } else {
            const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
            const double alpha = 1.0 - sqrt(D);
            residual_scaling_ = sqrt_rho1_ / (1 - alpha);
            alpha_sq_norm_ = alpha / sq_norm;
        }
        for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
            jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));

        residuals *= residual_scaling_;
    }
}


MarginalizationInfo::~MarginalizationInfo() {

    for (auto it = parameter_block_data.begin(); it != parameter_block_data.end(); ++it)
        delete it->second;
    if (m == 0 || n == 0)
        return;
    for (int i = 0; i < (int)factors.size(); i++) {

        delete[] factors[i]->raw_jacobians;

        delete factors[i]->cost_function;

        delete factors[i];
    }
}

void MarginalizationInfo::addResidualBlockInfo(ResidualBlockInfo* residual_block_info) {
    factors.emplace_back(residual_block_info);

    std::vector<double*>& parameter_blocks = residual_block_info->parameter_blocks;
    std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes();

    for (int i = 0; i < static_cast<int>(residual_block_info->parameter_blocks.size()); i++) {
        double* addr = parameter_blocks[i];
        int size = parameter_block_sizes[i];
        parameter_block_size[addr] = size;
    }

    for (int i = 0; i < static_cast<int>(residual_block_info->drop_set.size()); i++) {
        double* addr = parameter_blocks[residual_block_info->drop_set[i]];
        parameter_block_drop_idx[addr] = 0;
    }
    for (int i = 0; i < static_cast<int>(residual_block_info->sparse_set.size()); i++) {
        double* addr = parameter_blocks[residual_block_info->sparse_set[i]];
        parameter_block_sparse_idx[addr] = 0;
    }

}




void* ThreadsConstructA(void* threadsstruct) {
    ThreadsStruct* p = ((ThreadsStruct*)threadsstruct);
    for (auto it : p->sub_factors) {
        it->Evaluate();
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++) {
            int idx_i = p->parameter_block_idx[it->parameter_blocks[i]];
            int size_i = p->parameter_block_size[it->parameter_blocks[i]];
            if (size_i == 7)
                size_i = 6;
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++) {
                int idx_j = p->parameter_block_idx[it->parameter_blocks[j]];
                int size_j = p->parameter_block_size[it->parameter_blocks[j]];
                if (size_j == 7)
                    size_j = 6;
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else {
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    p->A.block(idx_j, idx_i, size_j, size_i) = p->A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            p->b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    return threadsstruct;
}


void MarginalizationInfo::marginalize(bool initialinformation) {

    int pos = 0;

    for (auto& it : parameter_block_drop_idx) {
        if (parameter_block_idx.find(it.first) == parameter_block_idx.end()) {
            parameter_block_idx[it.first] = pos;
            pos += localSize(parameter_block_size[it.first]);
        }
    }
    m = pos;

    for (auto& it : parameter_block_size) {
        if (parameter_block_idx.find(it.first) == parameter_block_idx.end()) {
            parameter_block_idx[it.first] = pos;
            pos += localSize(it.second);
        }
    }

    n = pos - m;
    if ((pos == 0) && !initialinformation) {
        printf("unstable tracking...\n");
        return;
    }
    if (n == 0)
        return;

    TicToc t_summing;
    A = Eigen::MatrixXd(pos, pos);
    b = Eigen::VectorXd(pos);
    A.setZero();
    b.setZero();

    ThreadsStruct threadsstruct;
    for (auto it : factors) threadsstruct.sub_factors.push_back(it);
    threadsstruct.A = Eigen::MatrixXd::Zero(pos, pos);
    threadsstruct.b = Eigen::VectorXd::Zero(pos);
    threadsstruct.parameter_block_size = parameter_block_size;
    threadsstruct.parameter_block_idx = parameter_block_idx;
    ThreadsConstructA((void*) & (threadsstruct));
    A = threadsstruct.A;
    b = threadsstruct.b;


    if (m != 0) {
        Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose());
        Eigen::MatrixXd Amm_inv;
        if (USE_LDLT_FOR_PSEUDO_INVERSE) {
            Eigen::VectorXd diag = Eigen::VectorXd(Amm.cols());
            for (int i = 0; i < diag.size(); i++)
                diag(i) = EPSS;
            Amm.diagonal() += diag;
            Eigen::LDLT<Eigen::MatrixXd, Eigen::Upper>ldlt = Amm.selfadjointView<Eigen::Upper>().ldlt();
            Amm_inv = ldlt.solve(Eigen::MatrixXd::Identity(m, m));
#if USE_ASSERT
            Eigen::Index a1, a2;
            ASSERT(ldlt.vectorD().minCoeff(&a1, &a2) > -0.1);
#endif
        } else {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);
            Amm_inv = saes.eigenvectors()
                      * Eigen::VectorXd((saes.eigenvalues().array() > EPSS).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal()
                      * saes.eigenvectors().transpose();
        }
        Eigen::VectorXd bmm = b.segment(0, m);
        Eigen::MatrixXd Amr = A.block(0, m, m, n);
        Eigen::MatrixXd Arm = A.block(m, 0, n, m);
        Eigen::MatrixXd Arr = A.block(m, m, n, n);
        Eigen::VectorXd brr = b.segment(m, n);

        A = Arr - Arm * Amm_inv * Amr;
        b = brr - Arm * Amm_inv * bmm;
    }
    for (auto it : factors) {
        std::vector<int> block_sizes = it->cost_function->parameter_block_sizes();
        for (int i = 0; i < static_cast<int>(block_sizes.size()); i++) {
            double* addr = it->parameter_blocks[i];
            int size = block_sizes[i];
            if (parameter_block_data.find(addr) == parameter_block_data.end()) {
                double* data = new double[size];
                memcpy(data, it->parameter_blocks[i], sizeof(double) * size);
                parameter_block_data[addr] = data;
            }
        }
    }
    m = 0;
}


void MarginalizationInfo::marginalize_pointers(std::set<double*>feature_marge_pointer) {



    std::vector<double*> new_addr;
    std::vector<int> new_size;
    std::vector<int>new_idx;

    Eigen::MatrixXd newA;
    Eigen::VectorXd newb;

    Eigen::VectorXd b_plus = b;

    std::unordered_map<const double*, int>addr2idx_old;

    for (int i = 0; i < (int)keep_block_addr.size(); i++)
        addr2idx_old[keep_block_addr[i]] = keep_block_idx[i];


    n = 0; m = 0;
    int pos = 0;

    std::map<int, double*, less<int>>keep_block_addr_mapping;
    std::map<int, int, less<int>>keep_block_size_mapping;

    for (int i = 0; i < (int)keep_block_addr.size(); i++) {
        int idx = keep_block_idx[i];
        keep_block_addr_mapping[idx] = keep_block_addr[i];
        keep_block_size_mapping[idx] = keep_block_size[i];
    }

    for (auto it = keep_block_addr_mapping.begin(); it != keep_block_addr_mapping.end(); it++) {
        double* addr = it->second;
        int size = keep_block_size_mapping[it->first];
        if (feature_marge_pointer.find(addr) != feature_marge_pointer.end()) {
            new_addr.push_back(addr); new_idx.push_back(pos); new_size.push_back(size);
            pos += localSize(size);
        }
    }
    m = pos;
    n = A.cols() - m;
    for (auto it = keep_block_addr_mapping.begin(); it != keep_block_addr_mapping.end(); it++) {
        double* addr = it->second;
        int size = keep_block_size_mapping[it->first];
        if (feature_marge_pointer.find(addr) == feature_marge_pointer.end()) {
            new_addr.push_back(addr); new_idx.push_back(pos); new_size.push_back(size);
            pos += localSize(size);
        }
    }
    ASSERT(pos == A.cols());

    newA = Eigen::MatrixXd::Zero(A.rows(), A.cols());
    newb = Eigen::VectorXd::Zero(b_plus.size());

    for (int i = 0; i < (int)new_addr.size(); i++) {
        const double* addri = new_addr[i];
        int new_idxi = new_idx[i];
        int old_idxi = addr2idx_old[addri];
        int sizei = localSize(new_size[i]);
        newb.segment(new_idxi, sizei) = b_plus.segment(old_idxi, sizei);
        for (int j = i; j < (int)new_addr.size(); j++) {
            const double* addrj = new_addr[j];
            int new_idxj = new_idx[j];
            int old_idxj = addr2idx_old[addrj];
            int sizej = localSize(new_size[j]);
            newA.block(new_idxi, new_idxj, sizei, sizej) = A.block(old_idxi, old_idxj, sizei, sizej);
        }
    }
    newA = newA.selfadjointView<Eigen::Upper>();

    Eigen::VectorXd bmm = newb.segment(0, m);
    Eigen::MatrixXd Amm = newA.block(0, 0, m, m);
    Eigen::MatrixXd Anm = newA.block(m, 0, n, m);
    Eigen::MatrixXd Ann = newA.block(m, m, n, n);
    Eigen::VectorXd bnn = newb.segment(m, n);
    Amm.diagonal().array() += kMinMu;
    Eigen::MatrixXd Amm_inv = Amm.template selfadjointView<Eigen::Upper>().llt().solve(Eigen::MatrixXd::Identity(Amm.rows(), Amm.rows()));


    A = Ann - Anm * Amm_inv * Anm.transpose();
    b = bnn - Anm * Amm_inv * bmm;

    new_addr.clear(); new_size.clear(); new_idx.clear(); pos = 0;
    for (auto it = keep_block_addr_mapping.begin(); it != keep_block_addr_mapping.end(); it++) {
        double* addr = it->second;
        int size = keep_block_size_mapping[it->first];
        if (feature_marge_pointer.find(addr) == feature_marge_pointer.end()) {
            new_addr.push_back(addr); new_idx.push_back(pos); new_size.push_back(size);
            pos += localSize(size);
        }
    }
    ASSERT(pos == n);

    keep_block_size = new_size;
    keep_block_idx = new_idx;
    keep_block_addr = new_addr;
    keep_block_addr_set.clear();
    keep_block_data.clear();

    for (int i = 0; i < (int)keep_block_addr.size(); i++) {
        ASSERT(parameter_block_data.find(keep_block_addr[i]) != parameter_block_data.end());
        keep_block_data.push_back(parameter_block_data[keep_block_addr[i]]);
        keep_block_addr_set[keep_block_addr[i]] = i;
        ASSERT(keep_block_data[keep_block_data.size() - 1] != 0);
    }

    GetJacobianAndResidual();

    m = 0;

}


void MarginalizationInfo::setmarginalizeinfo(std::vector<double*>& parameter_block_addr_,
                                             std::vector<int>& parameter_block_global_size_, Eigen::MatrixXd A_, Eigen::VectorXd& b_,
                                             bool Sqrt) {
    n = m = 0;
    A = A_;
    b = b_;


    for (int i = 0; i < (int)parameter_block_addr_.size(); i++) {
        double* addr = parameter_block_addr_[i];
        int globalsize = parameter_block_global_size_[i];
        int localsize = localSize(globalsize);
        parameter_block_size[addr] = globalsize;
        parameter_block_idx[addr] = n;
        n += localsize;
        double* data = new double[globalsize];
        memcpy(data, addr, sizeof(double) * globalsize);
        parameter_block_data[addr] = data;
    }
    ASSERT(n == A.rows());
    ASSERT(m == 0);



}

void MarginalizationInfo::resetLinerizationPoint() {
    ASSERT(m == 0);
    Eigen::VectorXd dx(n);
    for (int i = 0; i < static_cast<int>(keep_block_size.size()); i++) {
        int size = keep_block_size[i];
        int idx = keep_block_idx[i] - m;
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(keep_block_addr[i], size);
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(keep_block_data[i], size);
        if (size != 7)
            dx.segment(idx, size) = x - x0;
        else {
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
            dx.segment<3>(idx + 3) = 2.0 * (Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
                dx.segment<3>(idx + 3) = 2.0 * -(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
        }
    }
    b = A * dx + b;
    linearized_residuals = linearized_jacobians * dx + linearized_residuals;

    for (int i = 0; i < static_cast<int>(keep_block_size.size()); i++) {
        int size = keep_block_size[i];
        Eigen::Map<Eigen::VectorXd>(keep_block_data[i], size) = Eigen::Map<Eigen::VectorXd>(keep_block_addr[i], size);

    }
}

void MarginalizationInfo::GetJacobianAndResidual() {
    A.diagonal().array() += EPSS;
    Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> llt = A.selfadjointView<Eigen::Upper>().llt();
    linearized_jacobians = llt.matrixL().transpose();
    Eigen::VectorXd inc = llt.solve(b);
    linearized_residuals = linearized_jacobians * inc;
    if (llt.info() != Eigen::Success)
        assert(0);

    // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
    // Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > EPSS).select(saes2.eigenvalues().array(), 0));
    // Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    // linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();

    // Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > EPSS).select(saes2.eigenvalues().array().inverse(), 0));
    // Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();
    // Eigen::MatrixXd tmp = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    // linearized_residuals = tmp * b;

    // LOG_OUT << "0.5*linearized_residuals.squaredNorm():" << 0.5 * linearized_residuals.squaredNorm() << std::endl;
    // cost0 = 0.5 * linearized_residuals.squaredNorm();
    // A = linearized_jacobians.transpose() * linearized_jacobians;
    // b = linearized_jacobians.transpose() * linearized_residuals;
}

void MarginalizationInfo::getParameterBlocks() {
    keep_block_size.clear();
    keep_block_idx.clear();
    keep_block_data.clear();
    keep_block_addr_set.clear();
    keep_block_addr.clear();

    for (const auto& it : parameter_block_idx) {
        if (it.second >= m) {
            keep_block_size.push_back(parameter_block_size[it.first]);
            keep_block_idx.push_back(parameter_block_idx[it.first]);
            keep_block_data.push_back(parameter_block_data[it.first]);
            keep_block_addr.push_back(reinterpret_cast<double*>(it.first));
        }
    }
    for (int i = 0; i < (int)keep_block_addr.size(); i++)
        keep_block_addr_set[keep_block_addr[i]] = i;
    GetJacobianAndResidual();
    ASSERT(m == 0);


}


