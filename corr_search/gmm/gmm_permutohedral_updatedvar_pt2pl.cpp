//
// Created by wei on 12/6/18.
//

#include "corr_search/gmm/gmm_permutohedral_updatedvar_pt2pl.h"

poser::GMMPermutohedralUpdatedSigmaPt2Pl::GMMPermutohedralUpdatedSigmaPt2Pl(
	poser::FeatureChannelType observation_world_vertex,
	poser::FeatureChannelType observation_world_normal,
	poser::FeatureChannelType model_live_vertex
) : GMMPermutohedralBase(
	    std::move(observation_world_vertex),
	    std::move(model_live_vertex)),
	observation_world_normal_(std::move(observation_world_normal)
) {

}

void poser::GMMPermutohedralUpdatedSigmaPt2Pl::CheckAndAllocateTarget(
	const poser::FeatureMap &observation,
	const poser::FeatureMap &model,
	poser::GeometricTargetBase &target
) {
	//The original part
	GMMPermutohedralBase::CheckAndAllocateTarget(observation, model, target);
	
	//The normal
	LOG_ASSERT(observation.ExistFeature(observation_world_normal_, MemoryContext::CpuMemory));
	
	//The second order
	const auto model_size = model.GetDenseFeatureDim().total_size();
	second_order_momentum_.resize(model_size);
}

void poser::GMMPermutohedralUpdatedSigmaPt2Pl::checkAndAllocateDenseTarget(
	const poser::FeatureMap &observation,
	const poser::FeatureMap &model,
	poser::DenseGeometricTarget &target
) {
	LOG_ASSERT(model_feature_channel_.is_dense());
	target.AllocateTargetForModel(model, context_, true);
}


void poser::GMMPermutohedralUpdatedSigmaPt2Pl::UpdateObservation(
	const poser::FeatureMap &observation,
	float sigma
) {
	//Only support spherical covariance
	sigma_value_[0] = sigma_value_[1] = sigma_value_[2] = sigma;
	const auto observation_normal = observation.GetTypedFeatureValueReadOnly<float4>(observation_world_normal_, MemoryContext::CpuMemory);
	
	//The initializer
	auto initializer = [&](float weight, const float4& vertex, int obs_idx, LatticeInfoUpdatedSigmaPt2Pl& info) -> void {
		const auto& normal_i = observation_normal[obs_idx];
		info.weight = weight;
		info.weight_vertex.x = weight * vertex.x;
		info.weight_vertex.y = weight * vertex.y;
		info.weight_vertex.z = weight * vertex.z;
		info.weight_normal.x = weight * normal_i.x;
		info.weight_normal.y = weight * normal_i.y;
		info.weight_normal.z = weight * normal_i.z;
		info.weight_vTv = weight * dotxyz(vertex, vertex);
	};
	
	auto updater = [&](float weight, const float4& vertex, int obs_idx, LatticeInfoUpdatedSigmaPt2Pl& info) -> void {
		const auto& normal_i = observation_normal[obs_idx];
		info.weight_vertex.x += weight * vertex.x;
		info.weight_vertex.y += weight * vertex.y;
		info.weight_vertex.z += weight * vertex.z;
		info.weight_normal.x += weight * normal_i.x;
		info.weight_normal.y += weight * normal_i.y;
		info.weight_normal.z += weight * normal_i.z;
		info.weight_vTv += weight * dotxyz(vertex, vertex);
		info.weight += weight;
	};
	
	//Do it
	lattice_map_.clear();
	buildLatticeIndexNoBlur(observation, initializer, updater);
}


void poser::GMMPermutohedralUpdatedSigmaPt2Pl::ComputeTarget(
	const poser::FeatureMap &observation,
	const poser::FeatureMap &model,
	poser::DenseGeometricTarget &target
) {
	//The struct used for aggregation
	struct AggregatedResult {
		float4 model_weighted_vertex, model_weighted_normal;
		float model_weight, model_weight_momentum;
	};
	
	//The method to initialize and zero-out the value
	const auto initializer = [](AggregatedResult& result) -> void {
		result.model_weight = 0;
		result.model_weight_momentum = 0;
		result.model_weighted_vertex = make_float4(0, 0, 0, 0);
		result.model_weighted_normal = make_float4(0, 0, 0, 0);
	};
	
	//The method to update the aggregated value from lattice
	const auto updater = [](
		AggregatedResult& result,
		float lattice_weight,
		const LatticeInfoUpdatedSigmaPt2Pl& info) -> void
	{
		result.model_weighted_vertex.x += lattice_weight * info.weight_vertex.x;
		result.model_weighted_vertex.y += lattice_weight * info.weight_vertex.y;
		result.model_weighted_vertex.z += lattice_weight * info.weight_vertex.z;
		result.model_weighted_normal.x += lattice_weight * info.weight_normal.x;
		result.model_weighted_normal.y += lattice_weight * info.weight_normal.y;
		result.model_weighted_normal.z += lattice_weight * info.weight_normal.z;
		result.model_weight_momentum += lattice_weight * info.weight_vTv;
		result.model_weight += lattice_weight * info.weight;
	};
	
	//The method to compute the result
	using parent_class = GMMPermutohedralBase<3, LatticeInfoUpdatedSigmaPt2Pl>;
	auto target_normal = target.GetTargetNormalReadWrite();
	
	//The method
	const auto target_computer = [&](float4& target_i, int model_idx, const AggregatedResult& result) -> void {
		//Do it
		auto& target_normal_i = target_normal[model_idx];
		if(result.model_weight > 1e-2f) {
			target_i.x = result.model_weighted_vertex.x / result.model_weight;
			target_i.y = result.model_weighted_vertex.y / result.model_weight;
			target_i.z = result.model_weighted_vertex.z / result.model_weight;
			target_i.w = result.model_weight / (result.model_weight + parent_class::outlier_constant_);
			target_normal_i.x = result.model_weighted_normal.x / result.model_weight;
			target_normal_i.y = result.model_weighted_normal.y / result.model_weight;
			target_normal_i.z = result.model_weighted_normal.z / result.model_weight;
			second_order_momentum_[model_idx] = result.model_weight_momentum / result.model_weight;
		} else {
			target_i = make_float4(0, 0, 0, 0);
			target_normal_i = make_float4(0, 0, 0, 0);
			second_order_momentum_[model_idx] = 0.0;
		}
	};
	
	//Invoke
	this->template computeTargetNoBlur<AggregatedResult>(
		observation, model,
		target,
		initializer, updater,
		target_computer);
}


float poser::GMMPermutohedralUpdatedSigmaPt2Pl::ComputeSigmaValue(const poser::FeatureMap &model,
                                                                  const poser::GeometricTargetBase &target) {
	//Get the input and check the size
	const auto model_live_vertex = model.GetTypedFeatureValueReadOnly<float4>(model_feature_channel_, MemoryContext::CpuMemory);
	const auto target_vertex = target.GetTargetVertexReadOnly();
	
	//Do it
	GMMPermutohedralUpdatedSigma::ComputeSigmaValue(
		model_live_vertex,
		target_vertex,
		TensorView<float>(second_order_momentum_));
}

