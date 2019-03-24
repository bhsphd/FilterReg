#pragma once
#include "corr_search/gmm/gmm_permutohedral_base.h"

template<int FeatureDim, typename LatticeValueT>
poser::GMMPermutohedralBase<FeatureDim, LatticeValueT>::GMMPermutohedralBase(
	poser::FeatureChannelType observation_world_vertex,
	poser::FeatureChannelType model_feature,
	poser::FeatureChannelType observation_feature)
	: SingleFeatureTargetComputerBase(
		MemoryContext::CpuMemory,
		std::move(observation_world_vertex),
		std::move(model_feature),
		std::move(observation_feature)),
	  outlier_constant_(0.2f)
{}

template<int FeatureDim, typename LatticeValueT>
void poser::GMMPermutohedralBase<FeatureDim, LatticeValueT>::buildLatticeIndexNoBlur(
	const poser::FeatureMap &observation,
	const ValueInitializer& initializer,
	const ValueUpdater& updater
) {
	//Fetch the input
	const auto obs_feature = observation.GetFeatureValueReadOnly(observation_feature_channel_, MemoryContext::CpuMemory);
	const auto obs_vertex = observation.GetTypedFeatureValueReadOnly<float4>(observation_world_vertex_, MemoryContext::CpuMemory);
	
	//Pre-allocate memory
	float inv_sigma[FeatureDim];
	for(auto k = 0; k < FeatureDim; k++)
		inv_sigma[k] = 1.0f / sigma_value_[k];
	float scaled_feature[FeatureDim];
	LatticeCoordKey<FeatureDim> lattice_key[FeatureDim + 2];
	float lattice_weight[FeatureDim + 2];
	
	for(auto i = 0; i < obs_feature.Size(); i++) {
		//Load the feature value
		const auto feature_i = obs_feature.template ValidElemVectorAt<float>(i);
		const auto vertex_i = obs_vertex[i];
		
		//Scale the feature
		for(auto k = 0; k < FeatureDim; k++)
			scaled_feature[k] = feature_i[k] * inv_sigma[k];
		
		//Compute the lattice
		permutohedral_lattice_noblur<FeatureDim>(scaled_feature, lattice_key, lattice_weight);
		
		//Insert into lattice map
		for(auto lattice_j_idx = 0; lattice_j_idx < FeatureDim + 1; lattice_j_idx++) {
			//Get the lattice and weight
			const auto& lattice_j = lattice_key[lattice_j_idx];
			const float weight_j = lattice_weight[lattice_j_idx];
			
			//Find it
			auto iter = lattice_map_.find(lattice_j);
			
			//This lattice is not yet in the map
			if(iter == lattice_map_.end()) {
				LatticeValueT info;
				initializer(weight_j, vertex_i, i, info);
				lattice_map_.emplace(lattice_j, info);
				continue;
			} else {
				//Update it
				LatticeValueT& info = iter->second;
				updater(weight_j, vertex_i, i, info);
			}
		}
	}
}

template<int FeatureDim, typename LatticeValueT>
void poser::GMMPermutohedralBase<FeatureDim, LatticeValueT>::CheckAndAllocateTarget(
	const poser::FeatureMap &observation,
	const poser::FeatureMap &model,
	poser::GeometricTargetBase &target
) {
	SingleFeatureTargetComputerBase::CheckAndAllocateTarget(observation, model, target);
	
	//Reserve the memory for lattice
	const auto& feature_blob = observation.GetFeatureValueRawBlobReadOnly(observation_feature_channel_, MemoryContext::CpuMemory);
	lattice_map_.reserve(feature_blob.TypedCapacity());
}


template<int FeatureDim, typename LatticeValueT>
template<typename AggregatedT>
void poser::GMMPermutohedralBase<FeatureDim, LatticeValueT>::computeTargetNoBlur(
	const poser::FeatureMap &observation,
	const poser::FeatureMap &model,
	poser::GeometricTargetBase &target,
	const AggregateValueInitializer<AggregatedT> &initializer,
	const AggregateValueUpdater<AggregatedT> &updater,
	const TargetComputerFromAggregatedValue<AggregatedT>& result_computer
) {
	//Fetch the model feature and corresponded target
	const auto model_feature = model.GetFeatureValueReadOnly(model_feature_channel_, MemoryContext::CpuMemory);
	auto target_vertex = target.GetTargetVertexReadWrite();
	LOG_ASSERT(model_feature.Size() == target_vertex.Size());
	
	//Pre-allocate memory
	float inv_sigma[FeatureDim];
	for(auto k = 0; k < FeatureDim; k++)
		inv_sigma[k] = 1.0f / sigma_value_[k];
	float scaled_feature[FeatureDim];
	LatticeCoordKey<FeatureDim> lattice_key[FeatureDim + 1];
	float lattice_weight[FeatureDim + 2];
	
	//The aggregated value
	AggregatedT aggregated_value;
	
	//Iterate over input
	for(auto model_idx = 0; model_idx < model_feature.Size(); model_idx++) {
		//Load the model feature and scale it
		const auto feature_i = model_feature.template ValidElemVectorAt<float>(model_idx);
		auto& target_i = target_vertex[model_idx];
		for(auto k = 0; k < FeatureDim; k++)
			scaled_feature[k] = feature_i[k] * inv_sigma[k];
		
		//Compute the lattice
		permutohedral_lattice_noblur<FeatureDim>(scaled_feature, lattice_key, lattice_weight);
		
		//Reset vertex and weight
		initializer(aggregated_value);
		
		//Iterate over the lattice
		for(auto lattice_j_idx = 0; lattice_j_idx < FeatureDim + 1; lattice_j_idx++) {
			//Get the lattice and weight
			const auto& lattice_j = lattice_key[lattice_j_idx];
			const float weight_j = lattice_weight[lattice_j_idx];
			
			//Find it
			const auto iter = lattice_map_.find(lattice_j);
			
			//This lattice is not yet in the map
			if(iter == lattice_map_.end()) {
				continue;
			} else {
				//Update it
				const auto& info = iter->second;
				updater(aggregated_value, weight_j, info);
			}
		}
		
		//Write the result
		result_computer(target_i, model_idx, aggregated_value);
	}
}

