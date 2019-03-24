//
// Created by wei on 11/29/18.
//

#include "kinematic/affine/affine_kinematic_model.h"
#include "geometry_utils/device2eigen.h"

poser::AffineKinematicModel::AffineKinematicModel(
	poser::FeatureChannelType reference_vertex,
	poser::FeatureChannelType live_vertex,
	poser::FeatureChannelType reference_normal,
	poser::FeatureChannelType live_normal
) : KinematicModelBase(
		MemoryContext::CpuMemory,
		std::move(reference_vertex),
		std::move(live_vertex),
		std::move(reference_normal),
		std::move(live_normal)),
	transformation_map_(Eigen::Map<Eigen::Matrix4f>(transformation_raw_))
{
	transformation_map_.setIdentity();
	transformation_eigen_.setIdentity();
}

void poser::AffineKinematicModel::SetMotionParameter(const poser::mat34 &transform) {
	//Set the eigen version
	transformation_map_ = to_eigen(transform);
	transformation_eigen_.matrix() = transformation_map_;
}

void poser::AffineKinematicModel::SetMotionParameter(const Eigen::Affine3f &transformation) {
	transformation_eigen_ = transformation;
	transformation_map_ = transformation_eigen_.matrix();
}

void poser::AffineKinematicModel::SetMotionParameter(const Eigen::Ref<const Eigen::Matrix4f>& transformation) {
	transformation_map_ = transformation;
	transformation_eigen_.matrix() = transformation;
}


float3 poser::AffineKinematicModel::GetAffineTransformationSingularValues() const {
	Eigen::Matrix3f linear = transformation_eigen_.linear().matrix();
	Eigen::JacobiSVD<Eigen::Matrix3f> svd(linear, Eigen::ComputeFullU | Eigen::ComputeFullV);
	const Eigen::Vector3f& scales = svd.singularValues();
	return from_eigen(scales);
}