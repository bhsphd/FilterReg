//
// Created by wei on 9/18/18.
//

#include "common/feature_channel_type.h"
#include "kinematic/rigid/rigid_kinematic_model.h"

poser::RigidKinematicModel::RigidKinematicModel(
	MemoryContext context,
	poser::FeatureChannelType reference_vertex,
	poser::FeatureChannelType live_vertex,
	poser::FeatureChannelType reference_normal,
	poser::FeatureChannelType live_normal)
	: KinematicModelBase(
		context,
		std::move(reference_vertex),
		std::move(live_vertex),
		std::move(reference_normal),
		std::move(live_normal))
{
	//Identity init
	transformation_ = mat34::identity();
}

void poser::RigidKinematicModel::SetMotionParameter(const poser::mat34 &transform) {
	transformation_ = transform;
}

void poser::RigidKinematicModel::UpdateWithTwist(const float3 &twist_rot, const float3 &twist_trans) {
	mat34 se3_update(twist_rot, twist_trans);
	transformation_ = se3_update * transformation_;
}

void poser::RigidKinematicModel::UpdateWithTwist(const Eigen::Ref<const Eigen::Twist6f>& twist) {
	float3 rot = make_float3(twist(0), twist(1), twist(2));
	float3 trans = make_float3(twist(3), twist(4), twist(5));
	UpdateWithTwist(rot, trans);
}