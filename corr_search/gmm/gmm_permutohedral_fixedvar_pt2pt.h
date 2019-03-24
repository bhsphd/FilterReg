//
// Created by wei on 11/27/18.
//

#pragma once

#include "corr_search/gmm/gmm_permutohedral_base.h"

namespace poser {
	
	//The internal struct of lattice
	struct LatticeInfoFixedSigma {
		float weight;
		float3 weight_vertex;
	};
	
	template<int FeatureDim>
	class GMMPermutohedralFixedSigma : public GMMPermutohedralBase<FeatureDim, LatticeInfoFixedSigma> {
	public:
		GMMPermutohedralFixedSigma(
			FeatureChannelType observation_world_vertex,
			FeatureChannelType model_feature,
			FeatureChannelType observation_feature = FeatureChannelType());
		~GMMPermutohedralFixedSigma() override = default;
		
		//The actual interface
		void UpdateObservation(const FeatureMap& observation, float sigma);
		void UpdateObservation(const FeatureMap& observation, float sigma[FeatureDim]);
		void ComputeTarget(
			const FeatureMap& observation,
			const FeatureMap& model,
			GeometricTargetBase& target);
	};
}

#include "corr_search/gmm/gmm_permutohedral_fixedvar_pt2pt.hpp"