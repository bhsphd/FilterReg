//
// Created by wei on 11/27/18.
//

#pragma once

#include "corr_search/nn_search/nn_pt2pt.h"

namespace poser {
	
	class TrimmedNN : public TruncatedNN {
	protected:
		//The ratio of correspondence that will be trimmed
		//The value should be in [0, 1]
		float trimmed_ratio_;
		std::vector<float> sorted_distance_;
	
		//Overload the assign target method
		void assignTarget(
			const TensorView<int>& model_nn_idx,
			const TensorView<float>& model_squared_dist,
			const FeatureMap& observation,
			GeometricTargetBase& target);
	public:
		TrimmedNN(
			FeatureChannelType observation_world_vertex,
			FeatureChannelType model_feature,
			FeatureChannelType observation_feature = FeatureChannelType());
		~TrimmedNN() override = default;
		
		//Update the trimmed ratio, before the check method
		void SetTrimmedRatio(float ratio) {
			LOG_ASSERT(!finalized_);
			LOG_ASSERT(ratio > 0.0f && ratio <= 1.0f);
			trimmed_ratio_ = ratio;
		}
		
		void CheckAndAllocateTarget(
			const FeatureMap& observation,
			const FeatureMap& model,
			GeometricTargetBase& target) override;
		
		//The interface to compute the target
		void ComputeTarget(
			const FeatureMap& observation,
			const FeatureMap& model,
			GeometricTargetBase& target) override;
	};
	
	
}
