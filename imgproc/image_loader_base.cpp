//
// Created by wei on 12/6/18.
//

#include "imgproc/image_loader_base.h"

poser::ImageLoaderBase::ImageLoaderBase(
	unsigned int img_rows, unsigned int img_cols,
	poser::FeatureChannelType dense_depth_key,
	poser::FeatureChannelType dense_rgba_key,
	bool cpu_only
) : img_rows_(img_rows), img_cols_(img_cols), cpu_only_(cpu_only),
    depth_key_(std::move(dense_depth_key)), rgba_key_(std::move(dense_rgba_key))
{
	if(!depth_key_.is_valid())
		depth_key_ = CommonFeatureChannelKey::RawDepthImage();
	if(!rgba_key_.is_valid())
		rgba_key_ = CommonFeatureChannelKey::RawRGBImage();
}

void poser::ImageLoaderBase::CheckAndAllocate(poser::FeatureMap &feature_map) {
	//Allocate the cpu buffer
	feature_map.AllocateDenseFeature<unsigned short>(
		depth_key_,
		TensorDim(img_rows_, img_cols_),
		MemoryContext::CpuMemory);
	feature_map.AllocateDenseFeature<uchar4>(
		rgba_key_,
		TensorDim(img_rows_, img_cols_),
		MemoryContext::CpuMemory);
	
	//If required, also allocate gpu memory
	if(!cpu_only_) {
		feature_map.AllocateDenseFeature<unsigned short>(
			depth_key_,
			TensorDim(img_rows_, img_cols_),
			MemoryContext::GpuMemory);
		feature_map.AllocateDenseFeature<uchar4>(
			rgba_key_,
			TensorDim(img_rows_, img_cols_),
			MemoryContext::GpuMemory);
	}
}

void poser::ImageLoaderBase::LoadDepthImage(poser::FeatureMap &feature_map, cudaStream_t stream) {
	//Check the existence
	LOG_ASSERT(feature_map.ExistDenseFeature(depth_key_.get_name_key(), MemoryContext::CpuMemory));
	
	//Get the component and process it
	auto depth_slice = feature_map.GetTypedDenseFeatureReadWrite<unsigned short>(depth_key_.get_name_key());
	LOG_ASSERT(depth_slice.Rows() == img_rows_);
	LOG_ASSERT(depth_slice.Cols() == img_cols_);
	FetchDepthImageCPU(depth_slice);
	
	//Upload them to gpu, if required
	if(!cpu_only_) {
		auto depth_gpu = feature_map.GetTypedDenseFeatureReadWrite<unsigned short>(depth_key_.get_name_key(), MemoryContext::GpuMemory);
		TensorCopyNoSync<unsigned short>(depth_slice, depth_gpu, stream);
	}
}

void poser::ImageLoaderBase::LoadColorImage(poser::FeatureMap &feature_map, cudaStream_t stream) {
	//Check the existence
	LOG_ASSERT(feature_map.ExistDenseFeature(rgba_key_.get_name_key(), MemoryContext::CpuMemory));
	
	//Get the component and process it
	auto rgba_slice = feature_map.GetTypedDenseFeatureReadWrite<uchar4>(rgba_key_.get_name_key());
	LOG_ASSERT(rgba_slice.Rows() == img_rows_);
	LOG_ASSERT(rgba_slice.Cols() == img_cols_);
	FetchRGBImageCPU(rgba_slice);
	
	//Upload them to gpu, if required
	if(!cpu_only_) {
		auto rgba_gpu = feature_map.GetTypedDenseFeatureReadWrite<uchar4>(rgba_key_.get_name_key(), MemoryContext::GpuMemory);
		TensorCopyNoSync<uchar4>(rgba_slice, rgba_gpu, stream);
	}
}
