//
// Created by wei on 10/16/18.
//

#pragma once
#include "common/feature_map.h"
#include "common/feature_channel_type.h"
#include "geometry_utils/device_mat.h"

namespace poser {
	
	/* Perform rigid transform of point cloud, usually from
	 * camera frame to world frame.
	 * This class supports both camera and world frame.
	 */
	class CloudRigidTransformer {
	public:
		explicit CloudRigidTransformer(
			MemoryContext context,
			FeatureChannelType vertex_from = FeatureChannelType(),
			FeatureChannelType vertex_to = FeatureChannelType(),
			FeatureChannelType normal_from = FeatureChannelType(),
			FeatureChannelType normal_to = FeatureChannelType());
		
		//The update of channel type, before finalize
		void SetTransformVertexFromChannel(FeatureChannelType vertex_from) { LOG_ASSERT(!finalized_); vertex_from_ = std::move(vertex_from); }
		void SetTransformVertexToChannel(FeatureChannelType vertex_to) { LOG_ASSERT(!finalized_); vertex_to_ = std::move(vertex_to); }
		void SetTransformNormalFromChannel(FeatureChannelType normal_from) { LOG_ASSERT(!finalized_); normal_from_ = std::move(normal_from); }
		void SetTransformNormalToChannel(FeatureChannelType normal_to) { LOG_ASSERT(!finalized_); normal_to_ = std::move(normal_to); }
		
		//The update of transform
		void SetRigidTransform(const mat34& transform) { rigid_transform_ = transform; }
		void SetRigidTransform(const Eigen::Isometry3f& transform) { rigid_transform_ = mat34(transform); }
	
		//Check the vertex from is in input, and allocate the
		//transformed cloud if it doesn't show up in cloud_map
		void CheckAndAllocate(FeatureMap& cloud_map);
		void Process(FeatureMap& cloud_map);
	private:
		MemoryContext context_;
		FeatureChannelType vertex_from_;
		FeatureChannelType vertex_to_;
		FeatureChannelType normal_from_;
		FeatureChannelType normal_to_;
		bool finalized_;
		
		//The transformation
		mat34 rigid_transform_;
		
		//The actual processor
		void processVertexCpu(FeatureMap& cloud_map);
		void processVertexNormalCpu(FeatureMap& cloud_map);
	};
	
}
