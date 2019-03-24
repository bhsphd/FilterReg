//
// Created by wei on 12/1/18.
//

#pragma once

#include "ransac/fitness_base.h"
#include <unordered_set>

namespace poser {
	
	/* Compute the fitness using voxel grid. If a model point
	 * is in a valid voxel, it is counted as a fitted point.
	 * The final fitness is the ratio of fitted points.
	 */
	class FitnessGeometricOnlyVoxel : public FitnessEvaluator {
	public:
		explicit FitnessGeometricOnlyVoxel(
			FeatureChannelType obs_vertex = CommonFeatureChannelKey::ObservationVertexCamera(),
			FeatureChannelType model_vertex = CommonFeatureChannelKey::LiveVertex())
			: FitnessEvaluator(std::move(obs_vertex), std::move(model_vertex)),
			  correspondence_l1_radius_(0.01f /*1cm*/) { inv_leaf_size_ = 1.0f / correspondence_l1_radius_; };
		~FitnessGeometricOnlyVoxel() override = default;
		
		//Setup of the parameter
		void SetCorrespondenceL1Threshold(float l1_dist) {
			LOG_ASSERT(l1_dist > 0.0f);
			correspondence_l1_radius_ = l1_dist;
			inv_leaf_size_ = 1.0f / correspondence_l1_radius_;
		}
		
		//Build the voxel index
		void UpdateObservation(const FeatureMap& observation) override;
		
		//Evaluate for one registration
		Result Evaluate(const FeatureMap& model) const override;
		
	private:
		//The radis that a point will be considered as fitted point
		float correspondence_l1_radius_;
		float inv_leaf_size_;
		
		//The voxel map utilities
		struct VoxelCoordHasher {
			std::size_t operator()(const int3& voxel) const {
				return std::hash<int>()(voxel.x)^std::hash<int>()(voxel.y)^std::hash<int>()(voxel.z);
			}
		};
		struct VoxelCoordComparator {
			bool operator()(const int3& lhs, const int3& rhs) const {
				return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
			}
		};
		
		//The voxel map
		std::unordered_set<int3, VoxelCoordHasher, VoxelCoordComparator> voxel_set_;
	};
	
}
