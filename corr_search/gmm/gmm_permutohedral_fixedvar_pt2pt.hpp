#pragma once
#include "corr_search/gmm/gmm_permutohedral_fixedvar_pt2pt.h"


template<int FeatureDim>
poser::GMMPermutohedralFixedSigma<FeatureDim>::GMMPermutohedralFixedSigma(
	FeatureChannelType observation_world_vertex,
	FeatureChannelType model_feature,
	FeatureChannelType observation_feature
) : GMMPermutohedralBase<FeatureDim, LatticeInfoFixedSigma>(
	    std::move(observation_world_vertex),
	    std::move(model_feature),
	    std::move(observation_feature))
{}

template<int FeatureDim>
void poser::GMMPermutohedralFixedSigma<FeatureDim>::UpdateObservation(const FeatureMap &observation, float sigma) {
	float sigma_array[FeatureDim];
	for(auto i = 0; i < FeatureDim; i++)
		sigma_array[i] = sigma;
	
	//Do it
	UpdateObservation(observation, sigma_array);
}

template<int FeatureDim>
void poser::GMMPermutohedralFixedSigma<FeatureDim>::UpdateObservation(const poser::FeatureMap &observation, float *sigma) {
	using base_class = GMMPermutohedralBase<FeatureDim, LatticeInfoFixedSigma>;
	
	for(auto i = 0; i < FeatureDim; i++)
		base_class::sigma_value_[i] = sigma[i];
	
	//The initializer
	auto initializer = [](float weight, const float4& vertex, int obs_idx, LatticeInfoFixedSigma& info) -> void {
		info.weight = weight;
		info.weight_vertex.x = weight * vertex.x;
		info.weight_vertex.y = weight * vertex.y;
		info.weight_vertex.z = weight * vertex.z;
	};
	
	auto updater = [](float weight, const float4& vertex, int obs_idx, LatticeInfoFixedSigma& info) -> void {
		info.weight_vertex.x += weight * vertex.x;
		info.weight_vertex.y += weight * vertex.y;
		info.weight_vertex.z += weight * vertex.z;
		info.weight += weight;
	};
	
	//Do it
	base_class::lattice_map_.clear();
	base_class::buildLatticeIndexNoBlur(observation, initializer, updater);
}

template<int FeatureDim>
void poser::GMMPermutohedralFixedSigma<FeatureDim>::ComputeTarget(
	const poser::FeatureMap &observation,
	const poser::FeatureMap &model,
	poser::GeometricTargetBase &target
) {
	//The struct used for aggregation
	struct AggregatedResult {
		float4 model_weighted_vertex;
		float model_weight;
	};
	
	//The method to initialize and zero-out the value
	const auto initializer = [](AggregatedResult& result) -> void {
		result.model_weight = 0;
		result.model_weighted_vertex = make_float4(0, 0, 0, 0);
	};
	
	//The method to update the aggregated value from lattice
	const auto updater = [](
		AggregatedResult& result,
		float lattice_weight,
		const LatticeInfoFixedSigma& info) -> void
	{
		result.model_weighted_vertex.x += lattice_weight * info.weight_vertex.x;
		result.model_weighted_vertex.y += lattice_weight * info.weight_vertex.y;
		result.model_weighted_vertex.z += lattice_weight * info.weight_vertex.z;
		result.model_weight += lattice_weight * info.weight;
	};
	
	//The method to compute the result
	using parent_class = GMMPermutohedralBase<FeatureDim, LatticeInfoFixedSigma>;
	
	//The method
	const auto target_computer = [this](float4& target_i, int model_idx, const AggregatedResult& result) -> void {
		//Do it
		if(result.model_weight > 1e-2f) {
			target_i.x = result.model_weighted_vertex.x / result.model_weight;
			target_i.y = result.model_weighted_vertex.y / result.model_weight;
			target_i.z = result.model_weighted_vertex.z / result.model_weight;
			target_i.w = result.model_weight / (result.model_weight + parent_class::outlier_constant_);
		} else {
			target_i = make_float4(0, 0, 0, 0);
		}
	};
	
	//Invoke
	this->template computeTargetNoBlur<AggregatedResult>(
		observation, model,
		target,
		initializer, updater,
		target_computer);
}
