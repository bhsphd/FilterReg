//
// Created by wei on 9/30/18.
//

#pragma once

#include "common/feature_map.h"
#include "common/macro_copyable.h"
#include "common/geometric_target_interface.h"
#include "kinematic/articulated/articulated_kinematic_model.h"

namespace poser {
	
	class ArticulatedPoint2PointTermAssemblerCPU {
	private:
		const float residual_weight_ = 1.0f;
		std::vector<float> body_twist_jtj_;
		std::vector<float> body_twist_jte_;
	public:
		explicit ArticulatedPoint2PointTermAssemblerCPU(float residual_weight = 1.0f)
		: residual_weight_(residual_weight) {}
		
		//Check the input and allocate internal buffer
		void CheckAndAllocate(
			const FeatureMap& geometric_model,
			const ArticulatedKinematicModel& kinematic_model,
			const GeometricTarget& target);
		
		//The processor for dense and sparse term
		void ProcessAssembleDenseTerm(
			const FeatureMap& geometric_model,
			const ArticulatedKinematicModel& kinematic_model,
			const GeometricTarget& target,
			//The output is directly in cpu
			Eigen::Ref<Eigen::MatrixXf> JtJ, Eigen::Ref<Eigen::VectorXf> JtError);
	};
}
