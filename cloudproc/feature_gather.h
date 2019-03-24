//
// Created by wei on 9/17/18.
//

#pragma once


#include "common/feature_map.h"

namespace poser {
	
	
	class FeatureGatherProcessor {
	private:
		FeatureChannelType gather_index_;
		vector<FeatureChannelType> gathered_feature_;
		MemoryContext context_;
	public:
		explicit FeatureGatherProcessor(
			MemoryContext context = MemoryContext::CpuMemory,
			FeatureChannelType gather_idx = FeatureChannelType(),
			vector<FeatureChannelType> gathered_channel = std::vector<FeatureChannelType>());
		void InsertGatheredFeature(FeatureChannelType feature) {
			LOG_ASSERT(feature.is_dense());
			gathered_feature_.emplace_back(feature);
		};
		
		//Check if the gathered feature exists in gather_from and the gather_idx
		//exist in gathered_to, allocate feature in gather to
		void CheckAndAllocate(const FeatureMap& gather_from, FeatureMap& gather_to);
		
		//Do gather on cpu
		void PerformGatherCPU(const FeatureMap& gather_from, FeatureMap& gather_to);
	};
	
	
}