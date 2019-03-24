//
// Created by wei on 11/29/18.
//

#pragma once

#include "common/geometric_target_interface.h"
#include "kinematic/affine/affine_kinematic_model.h"

namespace poser {
	
	class AffinePoint2PointAnalyticalCPU {
	public:
		explicit AffinePoint2PointAnalyticalCPU() {}
		
		//Check the input and allocate internal buffer
		//The cpu version doesn't have buffer
		void CheckAndAllocate(
			const FeatureMap& geometric_model,
			const AffineKinematicModel& kinematic_model,
			const DenseGeometricTarget& target);
		void CheckAndAllocate(
			const FeatureMap& geometric_model,
			const AffineKinematicModel& kinematic_model,
			const SparseGeometricTarget& target);
		
		//The actual computation method
		void ComputeAffineTransform(
			const FeatureMap& geometric_model,
			const AffineKinematicModel& kinematic_model,
			const DenseGeometricTarget& target,
			Eigen::Ref<Eigen::Matrix4f> model2target);
		
		//The static version
		static void ComputeTransformBetweenClouds(
			int cloud_size,
			const float4* model,
			const float4* target,
			//The output
			Eigen::Ref<Eigen::Matrix4f> model2target);
	};
	
	
}