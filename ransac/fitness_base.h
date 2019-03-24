//
// Created by wei on 12/1/18.
//

#pragma once

#include "common/feature_map.h"
#include <limits>

namespace poser {
	
	/* The method to evaluate the fitness. Should
	 * be used in the RANSAC inner loop. The fitness
	 * evaluation is the most expensive step.
	 * Thus, the implementation should be careful.
	 */
	class FitnessEvaluator {
	protected:
		//The shared internal state
		FeatureChannelType observation_vertex_channel_;
		FeatureChannelType model_vertex_channel_;
		
	public:
		//The constructor group
		explicit FitnessEvaluator(
			FeatureChannelType obs_vertex = CommonFeatureChannelKey::ObservationVertexCamera(),
			FeatureChannelType model_vertex = CommonFeatureChannelKey::LiveVertex())
		: observation_vertex_channel_(std::move(obs_vertex)),
		  model_vertex_channel_(std::move(model_vertex)) {};
		virtual ~FitnessEvaluator() = default;
		
		//A small struct to hold the result
		struct Result {
			//The score of fitness, guaranteed implemented
			float fitness_score;
			
			//Note that some evalautor doesnt implement this
			float inlier_l2_cost;
			
			//The "invalid" constructor
			Result() : inlier_l2_cost(0.0f), fitness_score(std::numeric_limits<float>::max()) {}
		};
		
		//The default evaluation interface
		virtual void UpdateObservation(const FeatureMap& observation) = 0;
		virtual Result Evaluate(const FeatureMap& model) const = 0;
	};
}
