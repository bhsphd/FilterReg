//
// Created by wei on 9/14/18.
//

#pragma once

#include "common/feature_proc.h"

namespace poser {
	
	/* Compute normal map from vertex map and raw depth image.
	 * This version using window nearest neighbour and PCA.
	 */
	class DepthNormalMapComputer : public FeatureProcessor {
	public:
		explicit DepthNormalMapComputer(
			MemoryContext context = MemoryContext::CpuMemory,
			int window_halfsize = 2,
			std::string depth_map = std::string(), 
			std::string vertex_map = std::string(), 
			std::string normal_map = std::string());
		
		//The public interface
		void CheckAndAllocate(FeatureMap& feature_map) override;
		void Process(FeatureMap& feature_map) override;
		void ProcessStreamed(FeatureMap& feature_map, cudaStream_t stream) override;
		
		//The actual computation interface
		void ComputeNormalMapGPU(
			const TensorView<unsigned short>& depth_map,
			const TensorView<float4>& vertex_map,
			TensorSlice<float4> normal_map,
			cudaStream_t stream = 0);
		
		//Compute normal using pca, can be slow for cpu
		void ComputeNormalMapPCACPU(
			const TensorView<unsigned short>& depth_map,
			const TensorView<float4>& vertex_map,
			TensorSlice<float4> normal_map);
		void ComputeNormalCenteralDiffCPU(
			const TensorView<unsigned short>& depth_map,
			const TensorView<float4>& vertex_map,
			TensorSlice<float4> normal_map);
	private:
		//The input
		MemoryContext context_;
		FeatureChannelType depth_map_key_;
		FeatureChannelType vertex_map_key_;
		
		//The output
		FeatureChannelType normal_map_key_;
		
		//The windows size for nn search
		const int window_halfsize_;
		const int dense_threshold_;
		
		
	};
	
}