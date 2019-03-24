//
// Created by wei on 9/14/18.
//

#pragma once

#include "common/feature_map.h"

namespace poser {
	
	/* The interface for Feature Processor. The processor takes input
	 * from the feature map and produce output to the feature map.
	 * The processing interface should be STATE-LESS.
	 * the feature map will maintain FeatureMetaType to get the
	 * input feature and write to correct output features.
	 * This class is just a programming pattern, they won't
	 * be accessed by, for instance, the pointer of FeatureProcessor
	 */
	class FeatureProcessor {
	public:
		virtual ~FeatureProcessor() = default;
		
		//The method to check the input feature, allocate the OUTPUT feature
		//in feature map and internal state in processor itself.
		virtual void CheckAndAllocate(FeatureMap& feature_map) = 0;
		
		//The processing interface that fetch input from the feature map
		//and produce output to allocated feature map. This method should
		//be state-less.
		virtual void Process(FeatureMap& feature_map) = 0;
		virtual void ProcessStreamed(
			FeatureMap& feature_map,
			cudaStream_t stream) { Process(feature_map); };
	};
}
