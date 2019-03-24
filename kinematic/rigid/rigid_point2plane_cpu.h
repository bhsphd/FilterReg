//
// Created by wei on 10/26/18.
//

#pragma once

#include "common/feature_map.h"
#include "common/macro_copyable.h"
#include "common/geometric_target_interface.h"
#include "kinematic/rigid/rigid_kinematic_model.h"

namespace poser {
	
	class RigidPoint2PlaneTermAssemblerCPU {
	private:
		//The weight parameter
		const float residual_weight_ = 1.0f;
	public:
		explicit RigidPoint2PlaneTermAssemblerCPU(float residual_weight = 1.0f) : residual_weight_(residual_weight) {};
		
		//Check the input and allocate internal buffer
		//The cpu version doesn't have buffer
		void CheckAndAllocate(
			const FeatureMap& geometric_model,
			const RigidKinematicModel& kinematic_model,
			const DenseGeometricTarget& target);
		
		void ProcessAssemble(
			const FeatureMap& geometric_model,
			const RigidKinematicModel& kinematic_model,
			const DenseGeometricTarget& target,
			Eigen::Ref<Eigen::Matrix6f> JtJ, Eigen::Ref<Eigen::Vector6f> JtError);
	};
	
}
