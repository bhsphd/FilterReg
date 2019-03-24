//
// Created by wei on 12/6/18.
//

#pragma once

#include "imgproc/image_loader_base.h"

namespace poser {
	
	class FrameLoaderBase : public ImageLoaderBase {
	public:
		explicit FrameLoaderBase(
			unsigned img_rows = 480, unsigned img_cols = 640,
			FeatureChannelType dense_depth_key = FeatureChannelType(),
			FeatureChannelType dense_rgba_key = FeatureChannelType(),
			bool cpu_only = true)
			: ImageLoaderBase(
				img_rows, img_cols,
				std::move(dense_depth_key), std::move(dense_rgba_key),
				cpu_only) {}
		~FrameLoaderBase() override = default;
	
		//The processing method
		void UpdateFrameIndex(int frame_idx) { frame_idx_ = frame_idx; }
		void FetchDepthImageCPU(TensorSlice<unsigned short> depth_map) override { FetchDepthImageCPU(frame_idx_, depth_map); };
		void FetchRGBImageCPU(TensorSlice<uchar4> rgba_map) override { FetchRGBImageCPU(frame_idx_, rgba_map); }
		
	protected:
		//The common internal state
		int frame_idx_;
		
		//The interface for actual processing
		virtual void FetchDepthImageCPU(int frame_idx, TensorSlice<unsigned short> depth_map) = 0;
		virtual void FetchRGBImageCPU(int frame_idx, TensorSlice<uchar4> rgba_map) = 0;
	};
	
}
