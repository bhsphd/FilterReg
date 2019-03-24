//
// Created by wei on 11/28/18.
//

#pragma once

#include "corr_search/gmm/gmm_permutohedral_base.h"

namespace poser {
	
	
	//The internal struct of lattice
	struct LatticeInfoUpdatedSigma {
		float weight;
		float3 weight_vertex;
		float weight_vTv;
	};
	
	class GMMPermutohedralUpdatedSigma : public GMMPermutohedralBase<3, LatticeInfoUpdatedSigma> {
	private:
		//The second order of the momentum
		std::vector<float> second_order_momentum_;
	public:
		GMMPermutohedralUpdatedSigma(
			FeatureChannelType observation_world_vertex,
			FeatureChannelType model_live_vertex);
		~GMMPermutohedralUpdatedSigma() override = default;
		
		//The actual interface
		void CheckAndAllocateTarget(
			const FeatureMap& observation,
			const FeatureMap& model,
			GeometricTargetBase& target) override;
		void UpdateObservation(const FeatureMap& observation, float sigma);
		
		/* Compute the target and the second order of momentum.
		 * The momentum is maintained in this class.
		 */
		void ComputeTarget(
			const FeatureMap& observation,
			const FeatureMap& model,
			GeometricTargetBase& target);
		
		/* The method to compute the new sigma value.
		 * Must be called after the ComputeTargetMethod.
		 * In other words, after the first iteration.
		 */
		float ComputeSigmaValue(const FeatureMap& model, const GeometricTargetBase& target);
		static float ComputeSigmaValue(
			const TensorView<float4>& model_live_v,
			const TensorView<float4>& target_v,
			const TensorView<float>& momentum);
	};
}
