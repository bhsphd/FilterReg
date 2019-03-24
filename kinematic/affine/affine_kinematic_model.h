//
// Created by wei on 11/29/18.
//

#pragma once

#include "common/feature_channel_type.h"
#include "common/feature_map.h"
#include "common/common_type.h"
#include "geometry_utils/device_mat.h"
#include "kinematic/kinematic_model_base.h"


namespace poser {
	
	/* The affine kinematic model represent the motion parameterized by
	 * 4x4 transformation matrix. The motion is defined from reference frame
	 * to live frame.
	 */
	class AffineKinematicModel : public KinematicModelBase {
	private:
		//The raw element is for use outside Eigen
		//The storage is COLUMN-major
		float transformation_raw_[16];
		Eigen::Map<Eigen::Matrix4f> transformation_map_;
		Eigen::Affine3f transformation_eigen_;
	public:
		explicit AffineKinematicModel(
			FeatureChannelType reference_vertex = FeatureChannelType(),
			FeatureChannelType live_vertex = FeatureChannelType(),
			FeatureChannelType reference_normal = FeatureChannelType(),
			FeatureChannelType live_normal = FeatureChannelType());
		
		//Set the motion parameter
		void SetMotionParameter(const mat34& transform);
		void SetMotionParameter(const Eigen::Affine3f& transformation);
		void SetMotionParameter(const Eigen::Ref<const Eigen::Matrix4f>& transformation);
		
		//The getter
		const Eigen::Affine3f& GetAffineTransformationEigen() const { return transformation_eigen_; };
		const Eigen::Map<const Eigen::Matrix4f> GetAffineTransformationMappedMatrix() const { return Eigen::Map<const Eigen::Matrix4f>(transformation_raw_); }
		
		//The scale matrix
		float3 GetAffineTransformationSingularValues() const;
	};
}