//
// Created by wei on 11/28/18.
//

#pragma once

#include "corr_search/target_computer_base.h"


namespace poser {
	
	class FastGlobalRegistration : public SingleFeatureTargetComputerBase {
	private:
		//The algorithm rely on the live vertex of the model
		FeatureChannelType model_live_vertex_;
		
		//The shifted and scaled point
		std::vector<float4> normalized_obs_vertex_;
		std::vector<float4> normalized_model_vertex_;
		void buildNormalizedCloud(const FeatureMap& observation, const FeatureMap& model);
		
		//The method for reciprocity correspondence
		TensorBlob reciprocity_pairs_0_;
		TensorBlob reciprocity_pairs_1_;
		std::vector<int> feature_0_nn_, feature_1_nn_;
		std::vector<float> distance_buffer_;
		unsigned buildReciprocityCorrespondence(
			const BlobView& feature_0, const BlobView& feature_1,
			unsigned* corr_0, unsigned* corr_1);
		
		//The "tuple" test in the original paper.
		//This test only works for rigid registration
		int buildTupleTestedCorrespondence(
			unsigned* in_corr_0, unsigned* in_corr_1, unsigned num_in_corr,
			const float4* point_0, const float4* point_1,
			unsigned* updated_corr_0, unsigned* updated_corr_1);
	public:
		FastGlobalRegistration(
			FeatureChannelType observation_world_vertex,
			FeatureChannelType model_live_vertex,
			FeatureChannelType feature_channel);
		
		//Check the existence of feature, and allocate the target and internal buffer
		void CheckAndAllocateTarget(
			const FeatureMap& observation,
			const FeatureMap& model,
			GeometricTargetBase& target) override;
		
		void ComputeTarget(
			const FeatureMap& observation,
			const FeatureMap& model,
			SparseGeometricTarget& target);
	protected:
		void checkAndAllocateSparseTarget(
			const FeatureMap& observation,
			const FeatureMap& model,
			SparseGeometricTarget& target
		) override;
	};
	
}