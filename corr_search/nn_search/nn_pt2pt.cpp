//
// Created by wei on 11/26/18.
//

#include "corr_search/nn_search/nn_pt2pt.h"
#include "nn_pt2pt.h"

#include <vector_functions.h>


poser::TruncatedNN::TruncatedNN(
	FeatureChannelType observation_world_vertex,
	FeatureChannelType model_feature,
	FeatureChannelType observation_feature)
	: SingleFeatureTargetComputerBase(
		MemoryContext::CpuMemory,
		std::move(observation_world_vertex),
		std::move(model_feature),
		std::move(observation_feature)),
	    distance_threshold_(1e6f)
{}

void poser::TruncatedNN::CheckAndAllocateTarget(
	const poser::FeatureMap &observation,
	const poser::FeatureMap &model,
	poser::GeometricTargetBase &target
) {
	SingleFeatureTargetComputerBase::CheckAndAllocateTarget(observation, model, target);
	
	//Self allocate of feature
	const auto& feature_blob = observation.GetFeatureValueRawBlobReadOnly(observation_feature_channel_, MemoryContext::CpuMemory);
	kd_tree_.ResetInputData(feature_blob);
	
	//Self allocate of target
	auto target_size = target.GetTargetFlattenSize();
	result_index_.Reset<int>(target_size);
	result_distance_.Reset<float>(target_size);
}

void poser::TruncatedNN::UpdateObservation(const poser::FeatureMap &observation) {
	const auto& feature_blob = observation.GetFeatureValueRawBlobReadOnly(observation_feature_channel_, MemoryContext::CpuMemory);
	kd_tree_.ResetInputData(feature_blob);
}

void poser::TruncatedNN::ComputeTarget(
	const poser::FeatureMap &observation,
	const poser::FeatureMap &model,
	poser::GeometricTargetBase &target
) {
	//Search kdtree
	searchKDTree(model);
	
	//Assign it
	const auto model_nn_idx = result_index_.GetTypedTensorReadOnly<int>();
	
	//Get the target
	auto target_vertex = target.GetTargetVertexReadWrite();
	const auto observation_vertex = observation.GetTypedFeatureValueReadOnly<float4>(observation_world_vertex_, MemoryContext::CpuMemory);
	
	//Write the result to target
	for(auto i = 0; i < model_nn_idx.Size(); i++) {
		float4& target_i = target_vertex[i];
		const auto nn_idx = model_nn_idx[i];
		
		//Fill in the target
		if(nn_idx < 0 || nn_idx >= observation_vertex.Size()) {
			target_i = make_float4(0, 0, 0, 0);
		} else {
			target_i = observation_vertex[nn_idx];
			target_i.w = 1.0f;
		}
	}
}

void poser::TruncatedNN::searchKDTree(const poser::FeatureMap &model) {
	//Search kdtree
	const auto model_feature = model.GetFeatureValueReadOnly(model_feature_channel_, MemoryContext::CpuMemory);
	LOG_ASSERT(model_feature.Size() == result_index_.TensorFlattenSize());
	LOG_ASSERT(model_feature.Size() == result_distance_.TensorFlattenSize());
	kd_tree_.SearchRadius(model_feature, distance_threshold_, (int*)result_index_.RawPtr(), (float*)result_distance_.RawPtr());
}


