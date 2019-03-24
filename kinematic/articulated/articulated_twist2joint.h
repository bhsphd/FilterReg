//
// Created by wei on 9/30/18.
//

#pragma once

#include "kinematic/articulated/articulated_kinematic_model.h"

namespace poser {
	
	
	/* The class takes input from JtJ computed by twist jaocbian
	 * and transfer them into joint space jacobian. The twist JtJ
	 * is just a float array, each 36 elements is the JtJ of one body.
	 * The twist JtError is similiar, but each 6 elements is twist JtError.
	 */
	struct ArticulatedJacobianTwist2Joint {
		static void AssembleJointSpaceJacobian(
			const ArticulatedKinematicModel& kinematic,
			const float* full_body_jtj,
			const float* body_jte, //The input, jtj is full matrix here
			Eigen::Ref<Eigen::MatrixXf> joint_jtj,
			Eigen::Ref<Eigen::VectorXf> joint_jte);
	};
}
