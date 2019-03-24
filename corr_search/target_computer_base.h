//
// Created by wei on 9/21/18.
//

#pragma once

#include "common/feature_channel_type.h"
#include "common/feature_map.h"
#include "common/geometric_target_interface.h"

namespace poser {
	
	
	class SingleFeatureTargetComputerBase {
	protected:
		MemoryContext context_;
		FeatureChannelType observation_feature_channel_;
		FeatureChannelType observation_world_vertex_; //The vertex should represent in world frame
		FeatureChannelType model_feature_channel_;
		FeatureChannelType model_visibility_score_;
		bool finalized_;

	  public:
		explicit SingleFeatureTargetComputerBase(
			MemoryContext context = MemoryContext::CpuMemory,
			FeatureChannelType observation_world_vertex = FeatureChannelType(),
			FeatureChannelType model_feature = FeatureChannelType(),
			FeatureChannelType observation_feature = FeatureChannelType());
		virtual ~SingleFeatureTargetComputerBase() = default;
		
		//The method to modify the content
		void SetupMemoryContext(MemoryContext context) { LOG_ASSERT(!finalized_); context_ = context; }
		void SetupObservationFeatureChannel(FeatureChannelType channel) { LOG_ASSERT(!finalized_); observation_feature_channel_ = std::move(channel); }
		void SetupModelFeatureChannel(FeatureChannelType channel) { LOG_ASSERT(!finalized_); model_feature_channel_ = std::move(channel); }
		void SetupObservationWorldVertexChannel(FeatureChannelType channel) { LOG_ASSERT(!finalized_); observation_world_vertex_ = std::move(channel); }
		void SetupVisibilityScoreChannel(FeatureChannelType channel) { LOG_ASSERT(!finalized_); model_visibility_score_ = std::move(channel); };
		
		
		/* The checking method. After these method,
		 * the feature channel type should not change
		 */
	protected:
		//The basic check method
		void checkModelAndObservationBasic(
			const FeatureMap& observation,
			const FeatureMap& model);
		//The detailed allocation method
		virtual void checkAndAllocateDenseTarget(
			const FeatureMap& observation,
			const FeatureMap& model,
			DenseGeometricTarget& target
		);
		virtual void checkAndAllocateSparseTarget(
			const FeatureMap& observation,
			const FeatureMap& model,
			SparseGeometricTarget& target
		);
	public:
		virtual void CheckAndAllocateTarget(
			const FeatureMap& observation,
			const FeatureMap& model,
			GeometricTargetBase& target);
	};
}
