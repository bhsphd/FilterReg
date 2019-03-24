//
// Created by wei on 9/18/18.
//

#pragma once

#include "common/feature_map.h"
#include "common/macro_copyable.h"
#include "common/geometric_target_interface.h"
#include "kinematic/rigid/rigid_kinematic_model.h"

namespace poser {
	
	class RigidPoint2PointTermAssemblerCPU {
	private:
		//The weight parameter
		float residual_weight_ = 1.0f;
	public:
		explicit RigidPoint2PointTermAssemblerCPU(float residual_weight = 1.0f) : residual_weight_(residual_weight) {};
		
		//Check the input and allocate internal buffer
		//The cpu version doesn't have buffer
		void CheckAndAllocate(
			const FeatureMap& geometric_model, 
			const RigidKinematicModel& kinematic_model, 
			const DenseGeometricTarget& target);
		void CheckAndAllocate(
			const FeatureMap& geometric_model,
			const RigidKinematicModel& kinematic_model,
			const SparseGeometricTarget& target);
		
		//The processing interface depends on term type
		void ProcessAssemble(
			const FeatureMap& geometric_model,
			const RigidKinematicModel& kinematic_model,
			const DenseGeometricTarget& target,
			//The output is directly in cpu
			Eigen::Ref<Eigen::Matrix6f> JtJ, Eigen::Ref<Eigen::Vector6f> JtError);
		
		//The output for sparse term
		void ProcessAssemble(
			const FeatureMap& geometric_model,
			const RigidKinematicModel& kinematic_model,
			const SparseGeometricTarget& target,
			Eigen::Ref<Eigen::Matrix6f> JtJ, Eigen::Ref<Eigen::Vector6f> JtError);
	};
}
