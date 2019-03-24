//
// Created by wei on 12/2/18.
//

#pragma once

#include "common/feature_map.h"
#include "geometry_utils/device_mat.h"
#include "geometry_utils/kdtree_flann.h"

namespace poser {
	
	/* The method to generate hypothesis for ransac-based
	 * pose estimation. The generator use general feature,
	 * might be learned or even 3d position (means almost random pose)
	 */
	void generateRansacHypothesisCorrespondence(
		const KDTreeSingleNN& obs_feature, const TensorView<float4>& obs_point,
		const BlobView& model_feature, const TensorView<float4>& model_point,
		int n_sample, int n_point_per_sample,
		TensorSlice<float4> selected_model_point, BlobSlice selected_model_feature,
		TensorSlice<float4> model_corresponded_point, TensorSlice<int> model_nn_idx);
	
	void computeRansacHypothesisRigid(
		const TensorView<float4>& selected_model_point,
		const TensorView<float4>& model_correspondence_point,
		int n_sample, int n_point_per_sample,
		TensorSlice<mat34> generated_hypothesis_pose,
		void* local_buffer);
	
	void computeRansacHypothesisAffine(
		const TensorView<float4>& selected_model_point,
		const TensorView<float4>& model_correspondence_point,
		int n_sample, int n_point_per_sample,
		TensorSlice<mat34> generated_hypothesis_transform,
		void* local_buffer);
}
