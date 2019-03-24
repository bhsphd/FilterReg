//
// Created by wei on 9/18/18.
//

#pragma once

#include "common/feature_channel_type.h"
#include "common/feature_map.h"
#include "common/common_type.h"
#include "geometry_utils/device_mat.h"
#include "kinematic/kinematic_model_base.h"


namespace poser {
	
	/* The rigid kinematic model represent the motion parameterized by
	 * 6DOF transformation. The motion is defined from reference frame
	 * to live frame.
	 */
	class RigidKinematicModel : public KinematicModelBase {
	private:
		//The variable as the parameter
		mat34 transformation_;
	public:
		explicit RigidKinematicModel(
			MemoryContext context,
			FeatureChannelType reference_vertex = FeatureChannelType(),
			FeatureChannelType live_vertex = FeatureChannelType(),
			FeatureChannelType reference_normal = FeatureChannelType(),
			FeatureChannelType live_normal = FeatureChannelType());
		
		//Update the pose
		void SetMotionParameter(const mat34& transform);
		
		//Update the pose
		void UpdateWithTwist(const float3& twist_rot, const float3& twist_trans);
		void UpdateWithTwist(const Eigen::Ref<const Eigen::Twist6f>& twist);
		
		//The moition patameter
		const mat34& GetRigidTransform() const { return transformation_; }
	};
}
