//
// Created by wei on 12/2/18.
//

#include "ransac/fitness_geometry_permutohedral.h"

void poser::FitnessGeometryOnlyPermutohedral::UpdateObservation(const poser::FeatureMap &observation) {
	lattice_index_.UpdateObservation(observation, correspondence_sigma_);
}

poser::FitnessEvaluator::Result poser::FitnessGeometryOnlyPermutohedral::Evaluate(const poser::FeatureMap &model) const {
	//Get the model vertex
	const auto model_v = model.GetFeatureValueReadOnly(model_vertex_channel_, MemoryContext::CpuMemory);
	
	//Pre-allocate memory
	constexpr int FeatureDim = 3;
	const float inv_sigma = 1.0f / correspondence_sigma_;
	float scaled_feature[FeatureDim];
	LatticeCoordKey<FeatureDim> lattice_key[FeatureDim + 1];
	LatticeCoordKey<FeatureDim> lattice_max_weight;
	float lattice_weight[FeatureDim + 2];
	const auto& lattice_set = lattice_index_.GetLatticeMap();
	
	//Iterate over input
	unsigned inlier_count = 0;
	for(auto model_idx = 0; model_idx < model_v.Size(); model_idx++) {
		const auto feature_i = model_v.ValidElemVectorAt<float>(model_idx);
		
		//Scale the feature
		for(auto k = 0; k < FeatureDim; k++)
			scaled_feature[k] = feature_i[k] * inv_sigma;
		
		//Compute the lattice
		permutohedral_lattice_noblur<FeatureDim>(scaled_feature, lattice_key, lattice_weight);
		
		//Select the lattice key with max weight
		float max_weight = -1.0f;
		for(auto j = 0; j < FeatureDim + 1; j++) {
			if(lattice_weight[j] > max_weight) {
				max_weight = lattice_weight[j];
				lattice_max_weight = lattice_key[j];
			}
		}
		
		//Check if that lattice exist
		if(lattice_set.find(lattice_max_weight) != lattice_set.end()) {
			inlier_count++;
		}
	}
	
	//OK
	Result result;
	result.fitness_score = float(inlier_count) / float(model_v.Size());
	result.inlier_l2_cost = std::numeric_limits<float>::max();
	return result;
}
