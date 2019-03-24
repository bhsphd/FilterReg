//
// Created by wei on 12/3/18.
//

#include "ransac/ransac_common.h"
#include "ransac/ransac_hypothesis.h"


void poser::RansacBase::generateHypothesisCorrespondence(
	const poser::FeatureMap &observation,
	const poser::FeatureMap &model
) {
	//The input
	const auto obs_point = observation.GetTypedFeatureValueReadOnly<float4>(ransac_parameter_.obs_vertex, MemoryContext::CpuMemory);
	const auto& obs_feature = obs_feature_kdtree_;
	const auto model_feature = model.GetFeatureValueReadOnly(ransac_parameter_.model_feature, MemoryContext::CpuMemory);
	const auto model_point = model.GetTypedFeatureValueReadOnly<float4>(ransac_parameter_.model_reference_vertex, MemoryContext::CpuMemory);
	
	//The output
	auto selected_model_point = selected_model_point_.GetTypedTensorReadWrite<float4>();
	auto selected_model_feature = selected_model_feature_.GetTensorReadWrite();
	auto model_corres_point = model_corresponded_point_.GetTypedTensorReadWrite<float4>();
	auto model_nn_idx = model_nn_index_.GetTypedTensorReadWrite<int>();
	
	//Check the size
	LOG_ASSERT(selected_model_feature.Size() == ransac_parameter_.n_samples * ransac_parameter_.n_point_per_sample);
	LOG_ASSERT(selected_model_feature.Size() == selected_model_point.Size());
	LOG_ASSERT(selected_model_feature.Size() == model_corres_point.Size());
	LOG_ASSERT(selected_model_feature.Size() == model_nn_idx.Size());
	
	//Invoke the method
	generateRansacHypothesisCorrespondence(
		obs_feature, obs_point,
		model_feature, model_point,
		ransac_parameter_.n_samples, ransac_parameter_.n_point_per_sample,
		selected_model_point, selected_model_feature,
		model_corres_point, model_nn_idx);
}

void poser::RansacBase::CheckAndAllocate(const poser::FeatureMap &observation, const poser::FeatureMap &model) {
	//Check the existance of feature in observation
	LOG_ASSERT(observation.ExistFeature(ransac_parameter_.obs_vertex, MemoryContext::CpuMemory));
	if(ransac_parameter_.obs_feature.is_valid())
		LOG_ASSERT(observation.ExistFeature(ransac_parameter_.obs_feature, MemoryContext::CpuMemory));
	
	//Also check in model
	LOG_ASSERT(model.ExistFeature(ransac_parameter_.model_reference_vertex, MemoryContext::CpuMemory));
	if(ransac_parameter_.model_feature.is_valid())
		LOG_ASSERT(model.ExistFeature(ransac_parameter_.model_feature, MemoryContext::CpuMemory));
	
	//Build the kdtree
	if(ransac_parameter_.obs_feature.is_valid()) {
		const auto obs_feature = observation.GetFeatureValueReadOnly(ransac_parameter_.obs_feature, MemoryContext::CpuMemory);
		obs_feature_kdtree_.ResetInputData(obs_feature);
		
		//Allocate the model point buffer
		auto sampled_total_size = ransac_parameter_.n_point_per_sample * ransac_parameter_.n_samples;
		selected_model_point_.Reset<float4>(sampled_total_size, MemoryContext::CpuMemory);
		selected_model_feature_.Reserve(sampled_total_size, obs_feature.TypeByte(), MemoryContext::CpuMemory, obs_feature.ValidTypeByte());
		selected_model_feature_.ResizeOrException(sampled_total_size);
		
		//Allocate the corresponded buffer
		model_corresponded_point_.Reset<float4>(sampled_total_size, MemoryContext::CpuMemory);
		model_nn_index_.Reset<int>(sampled_total_size, MemoryContext::CpuMemory);
	}
	
	//Allocate the method for hypothesis
	hypothesis_fitness_.Reset<FitnessEvaluator::Result>(ransac_parameter_.n_samples, MemoryContext::CpuMemory);
	hypothesis_flag_.resize(ransac_parameter_.n_samples);
	std::fill(hypothesis_flag_.begin(), hypothesis_flag_.end(), true);
}

