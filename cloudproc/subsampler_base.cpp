//
// Created by wei on 10/16/18.
//

#include "cloudproc/subsampler_base.h"

poser::SubsampleProcessorBase::SubsampleProcessorBase(
	MemoryContext context,
	poser::FeatureChannelType vertex,
	poser::FeatureChannelType gather_index,
	poser::SubsamplerCommonOption option)
	: context_(context),
	  vertex_channel_(std::move(vertex)),
	  gather_index_(std::move(gather_index)),
	  subsampler_option_(std::move(option)),
	  finalized_(false)
{
	if(!gather_index_.is_valid())
		gather_index_ = CommonFeatureChannelKey::GatherIndex();
	if(!vertex_channel_.is_valid())
		vertex_channel_ = CommonFeatureChannelKey::ObservationVertexCamera();
}

void poser::SubsampleProcessorBase::CheckAndAllocate(
	const poser::FeatureMap &in_cloud_map,
	poser::FeatureMap &out_cloud_map
) {
	LOG_ASSERT(in_cloud_map.ExistFeature(vertex_channel_, context_));
	if(subsampler_option_.foreground_mask.is_valid())
		LOG_ASSERT(in_cloud_map.ExistFeature(subsampler_option_.foreground_mask, context_));
	
	//As it is subsampling, the max output size is just input size
	auto input_capacity = in_cloud_map.GetDenseFeatureCapacity();
	auto max_out_size = input_capacity;
	
	//Actual allocation
	out_cloud_map.AllocateDenseFeature<unsigned>(gather_index_, max_out_size, context_);
	out_cloud_map.AllocateDenseFeature<float4>(vertex_channel_, max_out_size, context_);
	
	//Mark finalized
	finalized_ = true;
}

void poser::SubsampleProcessorBase::SetVertexChannel(poser::FeatureChannelType vertex_channel) {
	vertex_channel_ = std::move(vertex_channel);
	LOG_ASSERT(vertex_channel_.is_valid());
	LOG_ASSERT(!finalized_);
}

void poser::SubsampleProcessorBase::SetGatherIndexChannel(poser::FeatureChannelType index_channel) {
	gather_index_ = std::move(index_channel);
	LOG_ASSERT(gather_index_.is_valid());
	LOG_ASSERT(!finalized_);
}

void poser::SubsampleProcessorBase::SetSubsamplerOption(poser::SubsamplerCommonOption option) {
	subsampler_option_ = std::move(option);
	LOG_ASSERT(!finalized_);
}