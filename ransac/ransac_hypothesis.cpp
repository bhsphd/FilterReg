//
// Created by wei on 12/2/18.
//

#include "ransac/ransac_hypothesis.h"
#include "kinematic/rigid/rigid_point2point_kabsch.h"
#include "kinematic/affine/affine_pt2pt_analytical_cpu.h"
#include <chrono>
#include <random>

void poser::generateRansacHypothesisCorrespondence(
	const KDTreeSingleNN &obs_feature, const TensorView<float4> &obs_point,
	const BlobView &model_feature, const TensorView<float4> &model_point,
	int n_sample, int n_point_per_sample,
	TensorSlice<float4> selected_model_point, BlobSlice selected_model_feature,
	TensorSlice<float4> model_corresponded_point, TensorSlice<int> model_nn_idx
) {
	//Sanity check
	LOG_ASSERT(model_feature.Size() == model_point.Size());
	LOG_ASSERT(n_sample * n_point_per_sample == selected_model_feature.Size());
	LOG_ASSERT(n_sample * n_point_per_sample == selected_model_point.Size());
	LOG_ASSERT(n_sample * n_point_per_sample == model_corresponded_point.Size());
	LOG_ASSERT(n_sample * n_point_per_sample == model_nn_idx.Size());
	
	//Select feature point randomly
	long seed = std::chrono::system_clock::now().time_since_epoch().count();
	//long seed = 1;
	std::default_random_engine r_engine(seed);
	std::uniform_int_distribution<int> distribution(0, model_point.Size() - 1);
	
	//Select the feature
	for(auto i = 0; i < n_point_per_sample * n_sample; i++) {
		const auto rand_idx = (distribution(r_engine)) % model_point.Size();
		const auto model_feature_i = model_feature.ElemVectorAt<float>(rand_idx);
		const auto model_point_i = model_point[rand_idx];
		
		//Write to selected value
		selected_model_point[i] = model_point_i;
		auto select_feature_i = selected_model_feature.ElemVectorAt<float>(i);
		for(auto k = 0; k < select_feature_i.typed_size; k++)
			select_feature_i[k] = model_feature_i[k];
	}
	
	//Do kdtree search
	obs_feature.SearchNN(selected_model_feature, model_nn_idx.RawPtr(), (float*)(model_corresponded_point.RawPtr()));
	
	//Write to the corresponded buffer
	for(auto i = 0; i < n_point_per_sample * n_sample; i++) {
		const auto model_closest_idx = model_nn_idx[i];
		model_corresponded_point[i] = obs_point[model_closest_idx];
	}
}

void poser::computeRansacHypothesisRigid(
	const poser::TensorView<float4> &selected_model_point,
	const poser::TensorView<float4> &model_correspondence_point,
	int n_sample, int n_point_per_sample,
	poser::TensorSlice<poser::mat34> generated_hypothesis_pose,
	void* local_buffer
) {
	//Sanity check
	LOG_ASSERT(selected_model_point.Size() == model_correspondence_point.Size());
	LOG_ASSERT(n_sample * n_point_per_sample == selected_model_point.Size());
	LOG_ASSERT(n_sample == generated_hypothesis_pose.Size());
	
	//The local buffer
	auto* buffer_float4 = (float4*)(local_buffer);
	float4* current_model =       &(buffer_float4[0 * n_point_per_sample]);
	float4* current_target =      &(buffer_float4[1 * n_point_per_sample]);
	float4* centeralized_model =  &(buffer_float4[2 * n_point_per_sample]);
	float4* centeralized_target = &(buffer_float4[3 * n_point_per_sample]);
	
	//Do it
	for(auto i = 0; i < n_sample; i++) {
		const auto offset_i = i * n_point_per_sample;
		
		//Do selection
		for(auto j = 0; j < n_point_per_sample; j++) {
			current_model[j] = selected_model_point[offset_i + j];
			current_target[j] = model_correspondence_point[offset_i + j];
			current_model[j].w = 1.0f;
			current_target[j].w = 1.0f;
		}
		
		//Compute the pose
		RigidPoint2PointKabsch::ComputeTransformBetweenClouds(
			n_point_per_sample,
			current_model, current_target,
			centeralized_model, centeralized_target,
			generated_hypothesis_pose[i]);
	}
}

void poser::computeRansacHypothesisAffine(
	const poser::TensorView<float4> &selected_model_point,
	const poser::TensorView<float4> &model_correspondence_point,
	int n_sample, int n_point_per_sample,
	poser::TensorSlice<poser::mat34> generated_hypothesis_transform,
	void* local_buffer
) {
	//Sanity check
	LOG_ASSERT(selected_model_point.Size() == model_correspondence_point.Size());
	LOG_ASSERT(n_sample * n_point_per_sample == selected_model_point.Size());
	LOG_ASSERT(n_sample == generated_hypothesis_transform.Size());
	
	//The local buffer
	auto* buffer_float4 = (float4*)(local_buffer);
	float4* current_model =       &(buffer_float4[0 * n_point_per_sample]);
	float4* current_target =      &(buffer_float4[1 * n_point_per_sample]);
	Eigen::Matrix4f affine_transform;
	
	//Do it
	for(auto i = 0; i < n_sample; i++) {
		const auto offset_i = i * n_point_per_sample;
		
		//Do selection
		for(auto j = 0; j < n_point_per_sample; j++) {
			current_model[j] = selected_model_point[offset_i + j];
			current_target[j] = model_correspondence_point[offset_i + j];
			current_model[j].w = 1.0f;
			current_target[j].w = 1.0f;
		}
		
		//Compute transform
		AffinePoint2PointAnalyticalCPU::ComputeTransformBetweenClouds(
			n_point_per_sample,
			current_model, current_target,
			affine_transform);
		generated_hypothesis_transform[i] = mat34(affine_transform);
	}
}