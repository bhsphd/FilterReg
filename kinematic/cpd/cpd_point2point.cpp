#include "kinematic/cpd/cpd_point2point.h"

void poser::ConstructLinearEquationDense(
    const CoherentPointDriftKinematic &kinematic,
    const GeometricTargetBase &target,
    float lambda_times_sigmasquare,
    //The output buffer
    Eigen::Ref<Eigen::MatrixXf> A, Eigen::Ref<Eigen::MatrixXf> b
) {
    //Check the size of given buffer
    const auto x_0 = kinematic.GetReferencePoints();
    const auto n_point = x_0.Size();
    LOG_ASSERT(A.rows() == n_point);
    LOG_ASSERT(A.cols() == n_point);
    LOG_ASSERT(b.rows() == n_point);
    LOG_ASSERT(b.cols() == 3);

    //Get the target
    auto target_v = target.GetTargetVertexReadOnly();
    LOG_ASSERT(target_v.Size() == n_point);

    //Construct the A matrix
    A = kinematic.GetMotionRegularizerMatrix();
    
    //Note that the weight might be very small if no correspondence is found
    //This is the one in original paper, but may has numerical issues
    //for (auto i = 0; i < n_point; i++) {
    //    const float offset = lambda_times_sigmasquare * (1.0f / target_v[i].w);
    //    A(i, i) += offset;
    //}
    
    //Times the diagonal elements
    //Note that the matrix is no-longer symmetric
    for(auto r_idx = 0; r_idx < n_point; r_idx++) {
        const auto r_weight = target_v[r_idx].w;
        for(auto c_idx = 0; c_idx < n_point; c_idx++) {
            A(r_idx, c_idx) *= r_weight;
        }
        
        //The diagonal element
        A(r_idx, r_idx) += lambda_times_sigmasquare;
    }

    //Constrcut the b matrix
    for (auto i = 0; i < n_point; i++) {
        const float4 target_i = target_v[i];
        const float4 x0_i = x_0[i];
        const float weight = target_i.w;
        const auto *target_i_ptr = (const float *)(&target_i);
        const auto *x0_i_ptr = (const float *)(&x0_i);

        //Assign it, be careful about the notational difference with the paper
        for (auto k = 0; k < 3; k++) {
            b(i, k) = weight * (target_i_ptr[k] - x0_i_ptr[k]);
        }
    }
}