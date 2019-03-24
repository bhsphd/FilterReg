//
// Created by wei on 12/6/18.
//

#pragma once

#include "common/feature_proc.h"
#include "common/feature_channel_type.h"


namespace poser {
	
	class ImageLoaderBase {
	public:
		explicit ImageLoaderBase(
			unsigned img_rows = 480, unsigned img_cols = 640,
			FeatureChannelType dense_depth_key = FeatureChannelType(),
			FeatureChannelType dense_rgba_key = FeatureChannelType(),
			bool cpu_only = true);
		virtual ~ImageLoaderBase() = default;
		
		//The general interface
		void CheckAndAllocate(FeatureMap& feature_map);
		void LoadDepthImage(FeatureMap& feature_map, cudaStream_t stream = 0);
		void LoadColorImage(FeatureMap& feature_map, cudaStream_t stream = 0);
		
		//The interface for actual processing
		virtual void FetchDepthImageCPU(TensorSlice<unsigned short> depth_map) = 0;
		virtual void FetchRGBImageCPU(TensorSlice<uchar4> rgba_map) = 0;
		
	protected:
		//The option variable
		const unsigned img_rows_, img_cols_;
		const bool cpu_only_; //Load cpu map or cpu&&gpu map
		FeatureChannelType depth_key_;
		FeatureChannelType rgba_key_;
	};
	
}