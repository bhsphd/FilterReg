//
// Created by wei on 11/28/18.
//

#include "corr_search/gmm/gmm_permutohedral_updatedvar_pt2pt.h"

poser::GMMPermutohedralUpdatedSigma::GMMPermutohedralUpdatedSigma(
	poser::FeatureChannelType observation_world_vertex,
	poser::FeatureChannelType model_live_vertex
) : GMMPermutohedralBase<3, LatticeInfoUpdatedSigma>(
	    std::move(observation_world_vertex),
	    std::move(model_live_vertex))
{}

void poser::GMMPermutohedralUpdatedSigma::CheckAndAllocateTarget(
	const poser::FeatureMap &observation,
	const poser::FeatureMap &model,
	poser::GeometricTargetBase &target
) {
	//The original part
	GMMPermutohedralBase::CheckAndAllocateTarget(observation, model, target);
	
	//The second order
	const auto model_size = model.GetDenseFeatureDim().total_size();
	second_order_momentum_.resize(model_size);
}


void poser::GMMPermutohedralUpdatedSigma::UpdateObservation(
	const poser::FeatureMap &observation,
	float sigma
) {
	//Only support spherical covariance
	sigma_value_[0] = sigma_value_[1] = sigma_value_[2] = sigma;
	
	//The initializer
	auto initializer = [](float weight, const float4& vertex, int obs_idx, LatticeInfoUpdatedSigma& info) -> void {
		info.weight = weight;
		info.weight_vertex.x = weight * vertex.x;
		info.weight_vertex.y = weight * vertex.y;
		info.weight_vertex.z = weight * vertex.z;
		info.weight_vTv = weight * dotxyz(vertex, vertex);
	};
	
	auto updater = [](float weight, const float4& vertex, int obs_idx, LatticeInfoUpdatedSigma& info) -> void {
		info.weight_vertex.x += weight * vertex.x;
		info.weight_vertex.y += weight * vertex.y;
		info.weight_vertex.z += weight * vertex.z;
		info.weight_vTv += weight * dotxyz(vertex, vertex);
		info.weight += weight;
	};
	
	//Do it
	lattice_map_.clear();
	buildLatticeIndexNoBlur(observation, initializer, updater);
}

void poser::GMMPermutohedralUpdatedSigma::ComputeTarget(
	const poser::FeatureMap &observation,
	const poser::FeatureMap &model,
	poser::GeometricTargetBase &target
) {
	//The struct used for aggregation
	struct AggregatedResult {
		float4 model_weighted_vertex;
		float model_weight, model_weight_momentum;
	};
	
	//The method to initialize and zero-out the value
	const auto initializer = [](AggregatedResult& result) -> void {
		result.model_weight = 0;
		result.model_weight_momentum = 0;
		result.model_weighted_vertex = make_float4(0, 0, 0, 0);
	};
	
	//The method to update the aggregated value from lattice
	const auto updater = [](
		AggregatedResult& result,
		float lattice_weight,
		const LatticeInfoUpdatedSigma& info) -> void
	{
		result.model_weighted_vertex.x += lattice_weight * info.weight_vertex.x;
		result.model_weighted_vertex.y += lattice_weight * info.weight_vertex.y;
		result.model_weighted_vertex.z += lattice_weight * info.weight_vertex.z;
		result.model_weight_momentum += lattice_weight * info.weight_vTv;
		result.model_weight += lattice_weight * info.weight;
	};
	
	//The method to compute the result
	using parent_class = GMMPermutohedralBase<3, LatticeInfoUpdatedSigma>;
	
	//The method
	const auto target_computer = [this](float4& target_i, int model_idx, const AggregatedResult& result) -> void {
		//Do it
		if(result.model_weight > 1e-2f) {
			target_i.x = result.model_weighted_vertex.x / result.model_weight;
			target_i.y = result.model_weighted_vertex.y / result.model_weight;
			target_i.z = result.model_weighted_vertex.z / result.model_weight;
			target_i.w = result.model_weight / (result.model_weight + parent_class::outlier_constant_);
			second_order_momentum_[model_idx] = result.model_weight_momentum / result.model_weight;
		} else {
			target_i = make_float4(0, 0, 0, 0);
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

float poser::GMMPermutohedralUpdatedSigma::ComputeSigmaValue(
	const FeatureMap &model,
	const GeometricTargetBase &target
) {
	//Get the input and check the size
	const auto model_live_vertex = model.GetTypedFeatureValueReadOnly<float4>(model_feature_channel_, MemoryContext::CpuMemory);
	const auto target_vertex = target.GetTargetVertexReadOnly();
	
	//Do it
	ComputeSigmaValue(
		model_live_vertex,
		target_vertex,
		TensorView<float>(second_order_momentum_));
}

float poser::GMMPermutohedralUpdatedSigma::ComputeSigmaValue(
	const poser::TensorView<float4>& model_live_vertex,
	const poser::TensorView<float4>& target_vertex,
	const poser::TensorView<float>& M2
){
	LOG_ASSERT(M2.Size() == target_vertex.Size());
	LOG_ASSERT(M2.Size() == model_live_vertex.Size());
	
	//The iterate through model points
	float upper = 0.0f;
	float divisor = 0.0f;
	for(auto i = 0; i < model_live_vertex.Size(); i++) {
		//Get the data
		const float3 m1_i = make_float3(target_vertex[i].x, target_vertex[i].y, target_vertex[i].z);
		const float m2_i = M2[i];
		const float4 x_i = model_live_vertex[i];
		
		//Compute the numerator
		upper += target_vertex[i].w * (dotxyz(x_i, x_i) - 2.0f * dotxyz(m1_i, x_i) + m2_i);
		
		//Compute the divisor
		divisor += target_vertex[i].w;
	}
	
	//Seems ok
	return std::sqrt(upper / (divisor * 3.0f));
}





