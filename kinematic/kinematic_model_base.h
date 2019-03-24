//
// Created by wei on 9/20/18.
//

#pragma once

#include "common/feature_channel_type.h"
#include "common/feature_map.h"
#include "common/common_type.h"

namespace poser {
	
	
	class KinematicModelBase {
	protected:
		//The context that this model will operate on
		MemoryContext context_;
		
		//The reference and live vertex type
		//The vertex is compulsory on cpu and context
		FeatureChannelType reference_vertex_channel_;
		FeatureChannelType live_vertex_channel_;
		
		//The reference and live normal type
		//This one is optional
		FeatureChannelType reference_normal_channel_;
		FeatureChannelType live_normal_channel_;
	
	public:
		//The constructor
		explicit KinematicModelBase(MemoryContext context,
		                   FeatureChannelType reference_vertex = FeatureChannelType(),
		                   FeatureChannelType live_vertex = FeatureChannelType(),
		                   FeatureChannelType reference_normal = FeatureChannelType(),
		                   FeatureChannelType live_normal = FeatureChannelType());
		virtual ~KinematicModelBase() = default;
		
		//The method check the current geometric model,
		//The method allocate live/contexted vertex/normal if they do not present
		virtual void CheckGeometricModelAndAllocateAttribute(FeatureMap& geometric_model);
		
		
		//The getter method, must be accessed after CheckGeometricModelAndAllocateAttr method
		const FeatureChannelType& ReferenceVertexChannel() const { return reference_vertex_channel_; }
		const FeatureChannelType& LiveVertexChannel() const { return live_vertex_channel_; }
		const FeatureChannelType& ReferenceNormalChannel() const { return reference_normal_channel_; }
		const FeatureChannelType& LiveNormalChannel() const { return live_normal_channel_; }
		bool HasNormal() const { return reference_normal_channel_.is_valid(); }
		MemoryContext GetContext() const { return context_; }
	};
}
