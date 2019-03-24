#pragma once
#include "corr_search/gmm/gmm_permutohedral_fixedvar_pt2pl.h"

template<int FeatureDim>
poser::GMMPermutohedralFixedSigmaPt2Pl<FeatureDim>::GMMPermutohedralFixedSigmaPt2Pl(
	FeatureChannelType observation_world_vertex,
	FeatureChannelType observation_world_normal,
	FeatureChannelType model_feature,
	FeatureChannelType observation_feature
) : GMMPermutohedralBase<FeatureDim, LatticeInfoFixedSigmaPlane>(
	std::move(observation_world_vertex),
	std::move(model_feature),
	std::move(observation_feature)),
    observation_world_normal_(std::move(observation_world_normal))
{}

template<int FeatureDim>
void poser::GMMPermutohedralFixedSigmaPt2Pl<FeatureDim>::UpdateObservation(
	const poser::FeatureMap &observation,
	float sigma
) {
	using base_class = GMMPermutohedralBase<FeatureDim, LatticeInfoFixedSigmaPlane>;
	for(auto i = 0; i < FeatureDim; i++) {
		base_class::sigma_value_[i] = sigma;
	}
	
	//The initializer
	const auto observation_normal = observation.GetTypedFeatureValueReadOnly<float4>(observation_world_normal_, MemoryContext::CpuMemory);
	auto initializer = [&](float weight, const float4& vertex, int obs_idx, LatticeInfoFixedSigmaPlane& info) -> void {
		const auto& normal_i = observation_normal[obs_idx];
		info.weight = weight;
		info.weight_vertex.x = weight * vertex.x;
		info.weight_vertex.y = weight * vertex.y;
		info.weight_vertex.z = weight * vertex.z;
		info.weight_normal.x = weight * normal_i.x;
		info.weight_normal.y = weight * normal_i.y;
		info.weight_normal.z = weight * normal_i.z;
	};
	
	auto updater = [&](float weight, const float4& vertex, int obs_idx, LatticeInfoFixedSigmaPlane& info) -> void {
		const auto& normal_i = observation_normal[obs_idx];
		info.weight_vertex.x += weight * vertex.x;
		info.weight_vertex.y += weight * vertex.y;
		info.weight_vertex.z += weight * vertex.z;
		info.weight_normal.x += weight * normal_i.x;
		info.weight_normal.y += weight * normal_i.y;
		info.weight_normal.z += weight * normal_i.z;
		info.weight += weight;
	};
	
	//Do it
	base_class::lattice_map_.clear();
	base_class::buildLatticeIndexNoBlur(observation, initializer, updater);
}

template<int FeatureDim>
void poser::GMMPermutohedralFixedSigmaPt2Pl<FeatureDim>::CheckAndAllocateTarget(
	const poser::FeatureMap &observation,
	const poser::FeatureMap &model,
	poser::GeometricTargetBase &target
) {
	//Check the normal
	LOG_ASSERT(observation.ExistFeature(observation_world_normal_, MemoryContext::CpuMemory));
	
	//Do the remaining staff
	GMMPermutohedralBase<FeatureDim, LatticeInfoFixedSigmaPlane>::CheckAndAllocateTarget(observation, model, target);
}

template<int FeatureDim>
void poser::GMMPermutohedralFixedSigmaPt2Pl<FeatureDim>::checkAndAllocateDenseTarget(
	const poser::FeatureMap &observation,
	const poser::FeatureMap &model,
	poser::DenseGeometricTarget &target
) {
	using base_class = SingleFeatureTargetComputerBase;
	LOG_ASSERT(base_class::model_feature_channel_.is_dense());
	target.AllocateTargetForModel(model, base_class::context_, true);
}

template<int FeatureDim>
void poser::GMMPermutohedralFixedSigmaPt2Pl<FeatureDim>::ComputeTarget(
	const poser::FeatureMap &observation,
	const poser::FeatureMap &model,
	DenseGeometricTarget &target
) {
	//The struct used for aggregation
	struct AggregatedResult {
		float3 model_weighted_vertex;
		float3 model_weighted_normal;
		float model_weight;
	};
	
	//The method to initialize and zero-out the value
	const auto initializer = [](AggregatedResult& result) -> void {
		result.model_weight = 0;
		result.model_weighted_vertex = make_float3(0, 0, 0);
		result.model_weighted_normal = make_float3(0, 0, 0);
	};
	
	//The method to update the aggregated value from lattice
	const auto updater = [](
		AggregatedResult& result,
		float lattice_weight,
		const LatticeInfoFixedSigmaPlane& info) -> void
	{
		result.model_weighted_vertex.x += lattice_weight * info.weight_vertex.x;
		result.model_weighted_vertex.y += lattice_weight * info.weight_vertex.y;
		result.model_weighted_vertex.z += lattice_weight * info.weight_vertex.z;
		result.model_weighted_normal.x += lattice_weight * info.weight_normal.x;
		result.model_weighted_normal.y += lattice_weight * info.weight_normal.y;
		result.model_weighted_normal.z += lattice_weight * info.weight_normal.z;
		result.model_weight += lattice_weight * info.weight;
	};
	
	//The method to compute the result
	using parent_class = GMMPermutohedralBase<FeatureDim, LatticeInfoFixedSigmaPlane>;
	auto target_normal = target.GetTargetNormalReadWrite();
	
	//The method
	const auto target_computer = [&](float4& target_i, int model_idx, const AggregatedResult& result) -> void {
		//Do it
		auto& target_normal_i = target_normal[model_idx];
		if(result.model_weight > 1e-2f) {
			target_i.x = result.model_weighted_vertex.x / result.model_weight;
			target_i.y = result.model_weighted_vertex.y / result.model_weight;
			target_i.z = result.model_weighted_vertex.z / result.model_weight;
			target_normal_i.x = result.model_weighted_normal.x / result.model_weight;
			target_normal_i.y = result.model_weighted_normal.y / result.model_weight;
			target_normal_i.z = result.model_weighted_normal.z / result.model_weight;
			target_i.w = result.model_weight / (result.model_weight + parent_class::outlier_constant_);
		} else {
			target_i = make_float4(0, 0, 0, 0);
			target_normal_i = make_float4(0, 0, 0, 0);
		}
	};
	
	//Invoke
	this->template computeTargetNoBlur<AggregatedResult>(
		observation, model,
		target,
		initializer, updater,
		target_computer);
}
