//
// Created by wei on 12/2/18.
//

#pragma once

#include "ransac/fitness_base.h"
#include "geometry_utils/kdtree_flann.h"

namespace poser {
	
	/* Compute the fitness using L2 distance. If a model point
	 * is in a L2 ball of an observation, it is counted as a fitted point.
	 * The final fitness is the ratio of fitted points.
	 */
	class FitnessGeometricOnlyKDTree : public FitnessEvaluator {
	public:
		explicit FitnessGeometricOnlyKDTree(
			FeatureChannelType obs_vertex = CommonFeatureChannelKey::ObservationVertexCamera(),
			FeatureChannelType model_vertex = CommonFeatureChannelKey::LiveVertex())
		: FitnessEvaluator(std::move(obs_vertex), std::move(model_vertex)),
		  correspondence_radius_(0.01f /*1cm*/) {};
		~FitnessGeometricOnlyKDTree() override = default;
		
		//Setup of the parameter
		void SetCorrespondenceThreshold(float l2_dist) {
			LOG_ASSERT(l2_dist > 0.0f);
			correspondence_radius_ = l2_dist;
		}
		
		//Build the voxel index
		void UpdateObservation(const FeatureMap& observation) override;
		
		//Evaluate for one registration
		Result Evaluate(const FeatureMap& model) const override;
		
		
	private:
		//The radis that a point will be considered as fitted point
		float correspondence_radius_;
		KDTreeSingleNN kdtree_;
		
		//The buffer to search
		mutable std::vector<int> result_idx_buffer_;
		mutable std::vector<float> result_distsquare_buffer_;
	};
	
}
