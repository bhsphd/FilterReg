//
// Created by wei on 9/17/18.
//

#pragma once

#include "common/feature_map.h"
#include "common/intrinsic_type.h"

namespace poser {
	
	/* Compute the normal for given point cloud from the depth/vertex map.
	 * The point cloud must be sub-sampled from that map.
	 * This processor must be on CPU.
	 */
	class NormalEstimateWithMapIndex {
	private:
		Intrinsic intrinsic_; //Use to project vertex into map
		FeatureChannelType vertex_channel_;
		FeatureChannelType normal_channel_;
		
		//The options
		const int window_halfsize_ = 2;
	public:
		explicit NormalEstimateWithMapIndex(
			Intrinsic intrinsic = Intrinsic(),
			FeatureChannelType vertex = FeatureChannelType(), 
			FeatureChannelType normal = FeatureChannelType());
		
		void CheckAndAllocate(const FeatureMap& img_feature_map, FeatureMap& cloud_feature_map);
		void Process(const FeatureMap& img_feature_map, FeatureMap& cloud_feature_map);
	};
	
}
