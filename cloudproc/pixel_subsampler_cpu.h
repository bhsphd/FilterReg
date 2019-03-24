//
// Created by wei on 9/16/18.
//

#pragma once

#include "cloudproc/subsampler_common.h"
#include "cloudproc/subsampler_base.h"

namespace poser {
	
	
	/* Do pixel sub-sampling on the given cloud, with the input from a RBGD image
	 * and the corresponding features. If a foreground mask/segmentation mask is
	 * available, also filter the cloud with the mask.
	 */
	class PixelSubsampleProcessorCPU : public SubsampleProcessorBase {
	public:
		explicit PixelSubsampleProcessorCPU(
			FeatureChannelType vertex = FeatureChannelType(),
			FeatureChannelType gather_index = FeatureChannelType(),
			SubsamplerCommonOption option = SubsamplerCommonOption());
		void SetBoundaryClip(int clip_pixel) { LOG_ASSERT(!finalized_); LOG_ASSERT(clip_pixel >= 0); boundary_clip_pixel_ = clip_pixel; }
		void SetSubsampleStride(int subsample_stride) { LOG_ASSERT(!finalized_); LOG_ASSERT(subsample_stride > 0); subsample_stride_ = subsample_stride; }
		
		//The checking and processing method
	protected:
		using PixelFilter = std::function<bool(int r_idx, int c_idx, const float4& vertex)>;
		void processInternal(const FeatureMap& in_cloud_map, FeatureMap& out_cloud_map, const PixelFilter& filter);
	public:
		void Process(const FeatureMap& in_cloud_map, FeatureMap& out_cloud_map);
		
		//The options for processing
	private:
		int boundary_clip_pixel_ = 10;
		int subsample_stride_ = 5;
	};
}
