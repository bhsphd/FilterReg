//
// Created by wei on 9/14/18.
//

#pragma once

#include "common/feature_proc.h"
#include "common/intrinsic_type.h"
#include "common/feature_channel_type.h"

namespace poser {
	
	/* Compute vertex map from depth image. Might be raw depth or
	 * background subtracted depth image, condition on the name_key.
	 * All the vertex is expressed in camera frame.
	 */
	class DepthVertexMapComputer : public FeatureProcessor {
	public:
		explicit DepthVertexMapComputer(
			MemoryContext context = MemoryContext::CpuMemory,
			const Intrinsic& intrinsic = Intrinsic(),
			std::string depth_img = std::string(),
			std::string vertex_map = std::string());
		
		//The public interface
		void CheckAndAllocate(FeatureMap& feature_map) override;
		void Process(FeatureMap& feature_map) override;
		void ProcessStreamed(FeatureMap& feature_map, cudaStream_t stream) override;
		
		//The actual processing method
		void ComputeVertexMapCPU(const TensorView<unsigned short>& depth_img, TensorSlice<float4> vertex_map);
		void ComputeVertexMapGPU(
			const TensorView<unsigned short>& depth_img,
			TensorSlice<float4> vertex_map,
			cudaStream_t stream = 0);
	
	private:
		//The key parameter to compute the vertex map
		MemoryContext context_;
		FeatureChannelType depth_map_key_;
		FeatureChannelType vertex_map_key_;
		
		//The intrinsic parameter
		const Intrinsic intrinsic_;
		const IntrinsicInverse intrinsic_inv_;
	};
}
