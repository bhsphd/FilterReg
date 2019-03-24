//
// Created by wei on 9/14/18.
//

#include "imgproc/depth_bilateral_filter.h"
#include "common/feature_channel_type.h"


#include <opencv2/opencv.hpp>


poser::DepthBilateralFilter::DepthBilateralFilter(
	MemoryContext context,
	int kernel_size, unsigned short depth_sigma,
	std::string raw_depth,
	std::string filter_depth)
	: kernel_size_(kernel_size),
	  depth_sigma_(depth_sigma),
	  context_(context),
	  raw_depth_key_(std::move(raw_depth), sizeof(unsigned short)),
	  filter_depth_key_(std::move(filter_depth), sizeof(unsigned short))
{
	if(!raw_depth_key_.is_valid())
		raw_depth_key_ = CommonFeatureChannelKey::RawDepthImage();
	if(!filter_depth_key_.is_valid())
		filter_depth_key_ = CommonFeatureChannelKey::FilteredDepthImage();
}

void poser::DepthBilateralFilter::CheckAndAllocate(poser::FeatureMap &feature_map) {
	LOG_ASSERT(feature_map.ExistFeature(raw_depth_key_, context_));
	
	//Get the input map and allocate for new depth image
	auto raw_depth = feature_map.GetTypedFeatureValueReadOnly<unsigned short>(raw_depth_key_, context_);
	LOG_ASSERT(raw_depth.Rows() > 5);
	LOG_ASSERT(raw_depth.Cols() > 5);
	
	//Allocate the result
	feature_map.AllocateDenseFeature<unsigned short>(
		filter_depth_key_,
		raw_depth.DimensionalSize(),
		context_);
	
	//Allocate internal storage in case of cpu
	if(context_ == MemoryContext::CpuMemory) {
		raw_depth_float_ = cv::Mat(raw_depth.Rows(), raw_depth.Cols(), CV_32FC1);
		filter_depth_float_ = cv::Mat(raw_depth.Rows(), raw_depth.Cols(), CV_32FC1);
	}
}

void poser::DepthBilateralFilter::Process(poser::FeatureMap &feature_map) {
	ProcessStreamed(feature_map, 0);
}

void poser::DepthBilateralFilter::ProcessStreamed(poser::FeatureMap &feature_map, cudaStream_t stream) {
	//Fetch data
	auto raw_depth = feature_map.GetTypedFeatureValueReadWrite<unsigned short>(raw_depth_key_, context_);
	auto filter_depth = feature_map.GetTypedFeatureValueReadWrite<unsigned short>(filter_depth_key_, context_);
	LOG_ASSERT(raw_depth.Rows() == filter_depth.Rows());
	LOG_ASSERT(raw_depth.Cols() == filter_depth.Cols());
	
	//Depend on context
	if(raw_depth.IsCpuTensor()) {
		LOG_ASSERT(filter_depth.IsCpuTensor());
		PerformBilateralFilterCPU(raw_depth, filter_depth);
	} else {
		LOG_ASSERT(filter_depth.IsGpuTensor());
		PerformBilateralFilterGPU(raw_depth, filter_depth, stream);
	}
}

void poser::DepthBilateralFilter::PerformBilateralFilterCPU(
	const poser::TensorView<unsigned short> &raw_depth,
	poser::TensorSlice<unsigned short> filter_depth)
{
	//Convert raw_depth to float
	for(auto r_idx = 0; r_idx < raw_depth.Rows(); r_idx++)
		for(auto c_idx = 0; c_idx < raw_depth.Cols(); c_idx++)
			raw_depth_float_.at<float>(r_idx, c_idx) = static_cast<float>(raw_depth(r_idx, c_idx));
	
	//Do filtering
	cv::bilateralFilter(raw_depth_float_, filter_depth_float_, kernel_size_, depth_sigma_, kernel_size_);
	
	//Convert back
	for(auto r_idx = 0; r_idx < raw_depth.Rows(); r_idx++)
		for(auto c_idx = 0; c_idx < raw_depth.Cols(); c_idx++)
			filter_depth(r_idx, c_idx) = static_cast<unsigned short>(filter_depth_float_.at<float>(r_idx, c_idx));
}
