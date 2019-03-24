#include "common/geometric_target_interface.h"
#include "geometric_target_interface.h"


void poser::DenseGeometricTarget::AllocateTargetForModel(
	const poser::FeatureMap &model_feature,
	poser::MemoryContext context,
	bool with_normal
) {
	//Query the size
	auto model_capacity = model_feature.GetDenseFeatureCapacity();
	auto model_dim = model_feature.GetDenseFeatureDim();
	
	//Do malloc
	target_vertex_.Reserve<float4>(model_capacity, context);
	target_vertex_.Reset<float4>(model_dim, context);
	
	if(with_normal) {
		target_normal_.Reserve<float4>(model_capacity, context);
		target_normal_.Reset<float4>(model_dim, context);
	}
}

void poser::SparseGeometricTarget::AllocateForSparseFeature(
	const poser::FeatureMap &model_feature,
	const poser::FeatureChannelType &channel,
	bool with_observation_idx
) {
	//Query the size
	LOG_ASSERT(channel.is_sparse());
	const auto& blob = model_feature.GetFeatureValueRawBlobReadOnly(channel, MemoryContext::CpuMemory);
	const auto& index_blob = model_feature.GetSparseFeatureValueRawBlob(channel.get_name_key(), MemoryContext::CpuMemory);
	auto model_capacity = blob.TypedCapacity();
	auto model_dim = blob.TensorDimension();
	
	//Do malloc
	target_vertex_.Reserve<float4>(model_capacity, MemoryContext::CpuMemory);
	target_vertex_.Reset<float4>(model_dim, MemoryContext::CpuMemory);
	index_blob.CloneTo(model_index_);
	
	if(with_observation_idx) {
		observation_index_.Reserve<unsigned>(model_capacity, MemoryContext::CpuMemory);
		observation_index_.Reset<unsigned>(model_dim, MemoryContext::CpuMemory);
	}
}

void poser::SparseGeometricTarget::AllocateTargetForModel(int capacity) {
	//Do malloc
	target_vertex_.Reserve<float4>(capacity, MemoryContext::CpuMemory);
	target_vertex_.Reset<float4>(0, MemoryContext::CpuMemory);
	model_index_.Reserve<unsigned>(capacity, MemoryContext::CpuMemory);
	model_index_.ResizeOrException(0);
	observation_index_.Reserve<unsigned>(capacity, MemoryContext::CpuMemory);
	observation_index_.Reset<unsigned>(0, MemoryContext::CpuMemory);
}

void poser::SparseGeometricTarget::ResizeSparseTarget(int size) {
	target_vertex_.ResizeOrException(size);
	model_index_.ResizeOrException(size);
	observation_index_.ResizeOrException(size);
}
