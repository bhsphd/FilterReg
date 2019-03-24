//
// Created by wei on 11/28/18.
//

#pragma once

#include "corr_search/gmm/gmm_permutohedral_base.h"

namespace poser {
	
	//The internal struct of lattice
	struct LatticeInfoFixedSigmaPlane {
		float weight;
		float3 weight_vertex;
		float3 weight_normal;
	};
	
	template<int FeatureDim>
	class GMMPermutohedralFixedSigmaPt2Pl : public GMMPermutohedralBase<FeatureDim, LatticeInfoFixedSigmaPlane> {
	private:
		//The channel for normal
		FeatureChannelType observation_world_normal_;
	public:
		GMMPermutohedralFixedSigmaPt2Pl(
			FeatureChannelType observation_world_vertex,
			FeatureChannelType observation_world_normal,
			FeatureChannelType model_feature,
			FeatureChannelType observation_feature = FeatureChannelType());
		~GMMPermutohedralFixedSigmaPt2Pl() override = default;
		
		//The actual interface
		void CheckAndAllocateTarget(
			const FeatureMap& observation,
			const FeatureMap& model,
			GeometricTargetBase& target) override;
	protected:
		void checkAndAllocateDenseTarget(
			const FeatureMap& observation,
			const FeatureMap& model,
			DenseGeometricTarget& target
		) override;
	public:
		void UpdateObservation(const FeatureMap& observation, float sigma);
		void ComputeTarget(
			const FeatureMap& observation,
			const FeatureMap& model,
			DenseGeometricTarget& target);
	};
}

#include "corr_search/gmm/gmm_permutohedral_fixedvar_pt2pl.hpp"