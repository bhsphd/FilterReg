//
// Created by wei on 12/2/18.
//

#include "ransac/fitness_geometry_kdtree.h"

void poser::FitnessGeometricOnlyKDTree::UpdateObservation(const poser::FeatureMap &observation) {
	const auto vertex_in = observation.GetFeatureValueReadOnly(observation_vertex_channel_, MemoryContext::CpuMemory);
	kdtree_.ResetInputData(vertex_in);
	
	//Do some allocation
	result_idx_buffer_.reserve(2 * vertex_in.Size());
	result_distsquare_buffer_.reserve(2 * vertex_in.Size());
}

poser::FitnessEvaluator::Result poser::FitnessGeometricOnlyKDTree::Evaluate(const poser::FeatureMap &model) const {
	//Get the model vertex
	const auto model_v = model.GetFeatureValueReadOnly(model_vertex_channel_, MemoryContext::CpuMemory);
	result_idx_buffer_.resize(model_v.Size());
	result_distsquare_buffer_.resize(model_v.Size());
	
	//Do kdtree search
	kdtree_.SearchRadius(model_v, correspondence_radius_, result_idx_buffer_.data(), result_distsquare_buffer_.data());
	
	//Iterate through model
	unsigned inlier_count = 0;
	float total_residual = 0.0f;
	for(auto i = 0; i < model_v.Size(); i++) {
		const auto idx = result_idx_buffer_[i];
		const auto dist_square = result_distsquare_buffer_[i];
		if(idx >= 0 && idx < model_v.Size()) {
			inlier_count += 1;
			total_residual += dist_square;
		}
	}
	
	//Return the result
	Result result;
	result.fitness_score = float(inlier_count) / model_v.Size();
	result.inlier_l2_cost = total_residual / model_v.Size();
	return result;
}