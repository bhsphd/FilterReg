//
// Created by wei on 12/2/18.
//

#pragma once

#include "ransac/fitness_base.h"
#include "corr_search/gmm/gmm_permutohedral_fixedvar_pt2pt.h"

namespace poser {
	
	
	class FitnessGeometryOnlyPermutohedral : public FitnessEvaluator {
	public:
		explicit FitnessGeometryOnlyPermutohedral(
			const FeatureChannelType& obs_vertex = CommonFeatureChannelKey::ObservationVertexCamera(),
			const FeatureChannelType& model_vertex = CommonFeatureChannelKey::LiveVertex())
		: FitnessEvaluator(obs_vertex, model_vertex), 
		  lattice_index_(obs_vertex, model_vertex, obs_vertex),
		  correspondence_sigma_(0.01f /*1cm*/) { };
		~FitnessGeometryOnlyPermutohedral() override = default;
		
		//Setup of the parameter
		void SetCorrespondenceSigma(float sigma) {
			LOG_ASSERT(sigma > 0.0f);
			correspondence_sigma_ = sigma;
		}
		
		//Build the lattice index
		void UpdateObservation(const FeatureMap& observation) override;
		
		//Evaluate for one registration
		Result Evaluate(const FeatureMap& model) const override;
	
	private:
		//The sigma value and the internal permuhedral index
		float correspondence_sigma_;
		GMMPermutohedralFixedSigma<3> lattice_index_;
	};
	
}