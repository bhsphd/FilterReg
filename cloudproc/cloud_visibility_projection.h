//
// Created by wei on 10/9/18.
//

#pragma once

#include "common/common_type.h"
#include "common/feature_map.h"
#include "common/intrinsic_type.h"
#include "common/geometric_target_interface.h"
#include "geometry_utils/device_mat.h"

namespace poser {
	
	/* The method predict the visibility of the point cloud by project
	 * them using the extrinsic and instrinsic of a camera.
	 */
	class CloudVisibilityProjection {
	private:
		mat34 world2camera_;
		Intrinsic raw_intrinsic_;
		const int image_rows_;
		const int image_cols_;
		const int subsample_rate_;
		const float tolerance_depth_diff_; //If the different of depth is within this threshold, classified as visibile
		const float tolerance_viewangle_cos_;
		
		//The internal buffer for depth value
		TensorBlob subsampled_z_map_;
		FeatureChannelType vertex_channel_;
		FeatureChannelType normal_channel_;
		FeatureChannelType visibility_score_channel_;
	public:
		CloudVisibilityProjection(
			Intrinsic camera_intrinsic,
			FeatureChannelType vertex_channel,
			FeatureChannelType visibility_score_channel,
			int subsample_rate = 6,
			float tolerance_depth = 0.02f,
			int img_rows = 480, int img_cols = 640);
		CloudVisibilityProjection(
			Intrinsic camera_intrinsic,
			FeatureChannelType vertex_channel,
			FeatureChannelType normal_channel,
			FeatureChannelType visibility_score_channel,
			int subsample_rate = 6,
			float tolerance_depth = 0.02f,
			float tolerance_viewangle_cos = 0.3f,
			int img_rows = 480, int img_cols = 640);
		
		//Update the camera extrsinic
		void SetCamera2World(const mat34& camera2world);
		void SetCamera2World(const Eigen::Isometry3f& camera2world);
		
		//Check the vertex channel and allocate the visibility channel
		void CheckAndAllocate(FeatureMap& feature_map);
		
		//The method to predict the visbility
	private:
		void buildDepthMap(const FeatureMap& feature_map);
		void predictVisibilityVertexOnly(FeatureMap& feature_map);
		void predictVisibilityVertexNormal(FeatureMap& feature_map);
	public:
		void PredictVisibility(FeatureMap& feature_map);
		
		//Reweight the target with visibility
		//This is not the best method. The invisible target should not be computed
		void ReweightTarget(const FeatureMap& geometric_model, GeometricTargetBase& target);
	};
	
}