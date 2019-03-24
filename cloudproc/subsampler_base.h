//
// Created by wei on 10/16/18.
//

#pragma once

#include "cloudproc/subsampler_common.h"

namespace poser {
	
	/* The base class for all subsampler processor. Maintains the
	 * vertex and gather index channel type and some common options.
	 */
	class SubsampleProcessorBase {
	public:
		explicit SubsampleProcessorBase(
			MemoryContext context,
			FeatureChannelType vertex = FeatureChannelType(),
			FeatureChannelType gather_index = FeatureChannelType(),
			SubsamplerCommonOption option = SubsamplerCommonOption());
		virtual ~SubsampleProcessorBase() = default;
		
		//The checking method
		virtual void CheckAndAllocate(const FeatureMap& in_cloud_map, FeatureMap& out_cloud_map);
		virtual void SetVertexChannel(FeatureChannelType vertex_channel);
		virtual void SetGatherIndexChannel(FeatureChannelType index_channel);
		virtual void SetSubsamplerOption(SubsamplerCommonOption option);
		
	protected:
		//The memory context
		MemoryContext context_;
		
		//The input
		FeatureChannelType vertex_channel_;  // Compulsory
		
		//The output
		FeatureChannelType gather_index_; // Compulsory
		
		//The option struct
		SubsamplerCommonOption subsampler_option_;
		bool finalized_;
	};
}
