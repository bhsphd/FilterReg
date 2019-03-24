#pragma once

#include "common/geometric_target_interface.h"
#include "kinematic/cpd/cpd_kinematic_model.h"
#include <Eigen/Eigen>

namespace poser {
    
    //The method to construct the linear equation
    //The linear equation is: A w = b
    //A: (n_point, n_point), b: (n_point, 3) 
    void ConstructLinearEquationDense(
        const CoherentPointDriftKinematic &kinematic,
        const GeometricTargetBase &target,
        float lambda_times_sigmasquare,
        //The output buffer
        Eigen::Ref<Eigen::MatrixXf> A, Eigen::Ref<Eigen::MatrixXf> b);

} // posr
