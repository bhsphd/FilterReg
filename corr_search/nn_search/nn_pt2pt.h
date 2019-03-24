//
// Created by wei on 11/26/18.
//

#pragma once


#include "common/feature_channel_type.h"
#include "common/feature_map.h"
#include "common/geometric_target_interface.h"
#include "geometry_utils/kdtree_flann.h"
#include "corr_search/target_computer_base.h"

namespace poser {
	
	class TruncatedNN : public SingleFeatureTargetComputerBase {
	protected:
		//The result idx and distance
		TensorBlob result_index_;
		TensorBlob result_distance_;
		
		//The parameter
		float distance_threshold_;
		
		//The kd tree and search interface. After the
		//search method, result_index and distance are ready for use
		KDTreeSingleNN kd_tree_;
		void searchKDTree(const poser::FeatureMap &model);
	public:
		TruncatedNN(FeatureChannelType observation_world_vertex,
		            FeatureChannelType model_feature,
		            FeatureChannelType observation_feature = FeatureChannelType());
		~TruncatedNN() override = default;
		
		//Update the truncated distance, before check method
		void SetTruncatedDistance(float distance) {
			LOG_ASSERT(!finalized_);
			distance_threshold_ = distance;
		}
		
		//The method to check and allocate the target
		void CheckAndAllocateTarget(
			const FeatureMap& observation,
			const FeatureMap& model,
			GeometricTargetBase& target) override;
		
		//Build index in this method
		virtual void UpdateObservation(const poser::FeatureMap &observation);
		
		//Do actual staff
		virtual void ComputeTarget(
			const poser::FeatureMap &observation,
			const poser::FeatureMap &model,
			GeometricTargetBase& target);
	};
	
}