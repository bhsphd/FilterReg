//
// Created by wei on 12/3/18.
//

#pragma once

#include "ransac/fitness_base.h"
#include "geometry_utils/kdtree_flann.h"

namespace poser {
	
	struct RansacParameters {
		//The required info
		FeatureChannelType model_reference_vertex;
		FeatureChannelType obs_vertex;
		
		//The feature used for generate hypothesis
		//For random-init ransac, this is not required
		FeatureChannelType model_feature;
		FeatureChannelType obs_feature;
		
		//The parameters for ransac
		int n_samples;
		int n_point_per_sample;
	};
	
	class RansacBase {
		//The general information
	protected:
		RansacParameters ransac_parameter_;
		const FitnessEvaluator& fitness_evaluator_;
	public:
		RansacBase(
			RansacParameters parameters,
			const FitnessEvaluator& fitness)
		: ransac_parameter_(std::move(parameters)),
		  fitness_evaluator_(fitness) {};
		virtual ~RansacBase() = default;
		
		//The internal buffer for hypothesis generation
	protected:
		KDTreeSingleNN obs_feature_kdtree_;
		TensorBlob selected_model_point_, selected_model_feature_;
		TensorBlob model_corresponded_point_, model_nn_index_;
		
		//The internal method
		void generateHypothesisCorrespondence(const FeatureMap& observation, const FeatureMap& model);
	public:
		virtual void CheckAndAllocate(const FeatureMap& observation, const FeatureMap& model);
		
		//The internal buffer and method for hypothesis evaluation
	//protected:
		TensorBlob hypothesis_fitness_;
		std::vector<bool> hypothesis_flag_;
		void initHypothesisFlag() { std::fill(hypothesis_flag_.begin(), hypothesis_flag_.end(), true); }
	};
	
}
