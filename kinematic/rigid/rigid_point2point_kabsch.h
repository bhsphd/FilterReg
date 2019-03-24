//
// Created by wei on 11/27/18.
//

#pragma once

#include "common/feature_map.h"
#include "common/macro_copyable.h"
#include "common/geometric_target_interface.h"
#include "kinematic/rigid/rigid_kinematic_model.h"


namespace poser {
	
	class RigidPoint2PointKabsch {
	private:
		TensorBlob centralized_model_;
		TensorBlob centralized_target_;
		
		//The buffer for sparse cloud
		TensorBlob sparse_cloud_buffer_;
	public:
		//Check the input and allocate internal buffer
		void CheckAndAllocate(
			const FeatureMap& geometric_model,
			const RigidKinematicModel& kinematic_model,
			const GeometricTargetBase& target);
		
		//The processing method
		void ComputeTransformToTarget(
			const FeatureMap& geometric_model,
			const RigidKinematicModel& kinematic_model,
			const DenseGeometricTarget& target,
			mat34& model2target);
		void ComputeTransformToTarget(
			const FeatureMap& geometric_model,
			const RigidKinematicModel& kinematic_model,
			const SparseGeometricTarget& target,
			mat34& model2target);
		
		//The processor
		static void ComputeTransformBetweenClouds(
			int cloud_size,
			const float4* model,
			const float4* target,
			float4* centralized_model,
			float4* centralized_target,
			mat34& model2target);
	};
	
}