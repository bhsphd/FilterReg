//
// Created by wei on 11/27/18.
//

#include "corr_search/nn_search/trim_nn_pt2pt.h"

#include <algorithm>
#include <vector_functions.h>

poser::TrimmedNN::TrimmedNN(
	poser::FeatureChannelType observation_world_vertex,
	poser::FeatureChannelType model_feature,
	poser::FeatureChannelType observation_feature
) : TruncatedNN(
	std::move(observation_world_vertex),
	std::move(model_feature),
	std::move(observation_feature)),
    trimmed_ratio_(0.9f)
{}

void poser::TrimmedNN::CheckAndAllocateTarget(
	const poser::FeatureMap &observation,
	const poser::FeatureMap &model,
	poser::GeometricTargetBase &target
) {
	//The allocation method for base class
	TruncatedNN::CheckAndAllocateTarget(observation, model, target);
	
	//Allocate the buffer to sort the element
	sorted_distance_.resize(target.GetTargetFlattenSize());
}

void poser::TrimmedNN::ComputeTarget(
	const poser::FeatureMap &observation,
	const poser::FeatureMap &model,
	poser::GeometricTargetBase &target
) {
	searchKDTree(model);
	const auto model_nn_idx = result_index_.GetTypedTensorReadOnly<int>();
	const auto model_dist = result_distance_.GetTypedTensorReadOnly<float>();
	assignTarget(model_nn_idx, model_dist, observation, target);
}

void poser::TrimmedNN::assignTarget(
	const poser::TensorView<int> &typed_idx,
	const poser::TensorView<float> &typed_dist,
	const poser::FeatureMap &observation,
	poser::GeometricTargetBase &target
) {
	//Get the target
	auto target_vertex = target.GetTargetVertexReadWrite();
	const auto observation_vertex = observation.GetTypedFeatureValueReadOnly<float4>(observation_world_vertex_, MemoryContext::CpuMemory);
	LOG_ASSERT(sorted_distance_.size() == typed_dist.Size());
	
	//Copy and sort the distance
	for(auto i = 0; i < typed_dist.Size(); i++)
		sorted_distance_[i] = typed_dist[i];
	
	//Maybe a partial sort
	auto nth_elem = unsigned(trimmed_ratio_ * typed_dist.Size());
	if(nth_elem >= sorted_distance_.size())
		nth_elem = static_cast<unsigned>(sorted_distance_.size()) - 1;
	
	//The trimmed value
	float trimmed_value;
	if(trimmed_ratio_ > 0.5f) {
		auto reverse_nth_elem = sorted_distance_.size() - nth_elem;
		std::nth_element(sorted_distance_.begin(), sorted_distance_.begin() + reverse_nth_elem, sorted_distance_.end(), std::greater_equal<float>());
		trimmed_value = sorted_distance_[reverse_nth_elem];
	} else {
		std::nth_element(sorted_distance_.begin(), sorted_distance_.begin() + nth_elem, sorted_distance_.end());
		trimmed_value = sorted_distance_[nth_elem];
	}
	
	//Assign it
	for(auto i = 0; i < typed_idx.Size(); i++) {
		float4& target_i = target_vertex[i];
		const auto nn_idx = typed_idx[i];
		const auto dist_i = typed_dist[i];
		
		//Fill in the target
		target_i = observation_vertex[nn_idx];
		if(nn_idx < 0 || nn_idx >= observation_vertex.Size()) {
			target_i = make_float4(0, 0, 0, 0);
		} else {
			target_i = observation_vertex[nn_idx];
			if(dist_i <= trimmed_value) {
				target_i.w = 1.0f;
			}
			else
				target_i.w = 0.0f;
		}
	}
}