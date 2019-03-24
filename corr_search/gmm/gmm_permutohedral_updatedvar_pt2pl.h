//
// Created by wei on 12/6/18.
//

#pragma once

#include "corr_search/gmm/gmm_permutohedral_base.h"
#include "corr_search/gmm/gmm_permutohedral_updatedvar_pt2pt.h"

namespace poser {
	
	//The internal struct of lattice
	struct LatticeInfoUpdatedSigmaPt2Pl {
		float weight;
		float3 weight_vertex;
		float weight_vTv;
		float3 weight_normal;
	};
	
	class GMMPermutohedralUpdatedSigmaPt2Pl : public GMMPermutohedralBase<3, LatticeInfoUpdatedSigmaPt2Pl> {
	protected:
		//The second order of the momentum
		FeatureChannelType observation_world_normal_;
		std::vector<float> second_order_momentum_;
		
		void checkAndAllocateDenseTarget(
			const FeatureMap& observation,
			const FeatureMap& model,
			DenseGeometricTarget& target
		) override;
	public:
		GMMPermutohedralUpdatedSigmaPt2Pl(
			FeatureChannelType observation_world_vertex,
			FeatureChannelType observation_world_normal,
			FeatureChannelType model_live_vertex);
		~GMMPermutohedralUpdatedSigmaPt2Pl() override = default;
		
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
			DenseGeometricTarget& target);
	
		/* The method to compute the new sigma value.
		 * Must be called after the ComputeTargetMethod.
		 * In other words, after the first iteration.
		 */
		float ComputeSigmaValue(const FeatureMap& model, const GeometricTargetBase& target);
	};
}
