//
// Created by wei on 9/14/18.
//

#pragma once

#include "common/feature_proc.h"
#include <opencv2/opencv.hpp>

namespace poser {
	
	class DepthBilateralFilter : public FeatureProcessor {
	public:
		explicit DepthBilateralFilter(
			MemoryContext context = MemoryContext::CpuMemory,
			int kernel_size = 5, unsigned short depth_sigma = 20,
			std::string raw_depth = std::string(),
			std::string filter_depth = std::string());
		
		//The general interface
		void CheckAndAllocate(FeatureMap& feature_map) override;
		void Process(FeatureMap& feature_map) override;
		void ProcessStreamed(FeatureMap& feature_map, cudaStream_t stream) override;
		
		//The acutal computating interface
		void PerformBilateralFilterCPU(
			const TensorView<unsigned short>& raw_depth,
			TensorSlice<unsigned short> filter_depth);
		void PerformBilateralFilterGPU(
			const TensorView<unsigned short>& raw_depth,
			TensorSlice<unsigned short> filter_depth, 
			cudaStream_t stream = 0);
		
	private:
		//The input output pair
		MemoryContext context_;
		FeatureChannelType raw_depth_key_;
		FeatureChannelType filter_depth_key_;
		
		//The parameter for filtering
		const int kernel_size_;
		const unsigned short depth_sigma_;
		
		//The buffer for opencv bilateral filtering
		cv::Mat raw_depth_float_;
		cv::Mat filter_depth_float_;
	};
	
}
