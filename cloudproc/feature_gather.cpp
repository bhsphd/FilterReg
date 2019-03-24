//
// Created by wei on 9/17/18.
//

#include "cloudproc/feature_gather.h"
#include "common/feature_channel_type.h"


poser::FeatureGatherProcessor::FeatureGatherProcessor(
	MemoryContext context,
	poser::FeatureChannelType gather_idx,
	vector<FeatureChannelType> gathered_channel) :
	context_(context),
	gather_index_(std::move(gather_idx)),
	gathered_feature_(std::move(gathered_channel))
{
	if(!gather_index_.is_valid())
		gather_index_ = CommonFeatureChannelKey::GatherIndex();
	
	//Currently only support to dense feature gathering
	for(const auto& feature : gathered_feature_)
		LOG_ASSERT(feature.is_dense());
}

void poser::FeatureGatherProcessor::CheckAndAllocate(const poser::FeatureMap &gather_from, poser::FeatureMap &gather_to) {
	//Check the result in gather from
	for(const auto& channel : gathered_feature_) {
		LOG_ASSERT(gather_from.ExistFeature(channel, context_));
	}
	
	//Check the result in gather to
	LOG_ASSERT(gather_to.ExistFeature(gather_index_, context_));
	
	//Allocate the feature
	for(const auto& channel : gathered_feature_) {
		gather_to.AllocateDenseFeature(channel, context_);
	}
}

void poser::FeatureGatherProcessor::PerformGatherCPU(
	const poser::FeatureMap &gather_from,
	poser::FeatureMap &gather_to
) {
	const auto gather_index = gather_to.GetTypedFeatureValueReadOnly<unsigned>(gather_index_, context_);
	gather_to.ResizeDenseFeatureOrException(gather_index.DimensionalSize());
	
	//For each feature channel
	for(const auto& channel : gathered_feature_) {
		//Fetch the feature
		const auto feature_from = gather_from.GetFeatureValueReadOnly(channel, context_);
		auto feature_to = gather_to.GetFeatureValueReadWrite(channel, context_);
		LOG_ASSERT(feature_from.TypeByte() == feature_to.TypeByte());
		
		for(auto i = 0; i < gather_index.FlattenSize(); i++) {
			auto from_idx = gather_index[i];
			const auto from_ptr = feature_from.ElemVectorAt<char>(from_idx);
			auto to_ptr = feature_to.ElemVectorAt<char>(i);
			for(auto k = 0; k < to_ptr.typed_size; k++)
				to_ptr[k] = from_ptr[k];
		}
	}
}
