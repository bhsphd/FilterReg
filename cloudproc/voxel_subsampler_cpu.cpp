//
// Created by wei on 10/16/18.
//

#include "cloudproc/voxel_subsampler_cpu.h"
#include "geometry_utils/vector_operations.hpp"


poser::VoxelGridSubsamplerCPU::VoxelGridSubsamplerCPU(
	poser::FeatureChannelType vertex,
	poser::FeatureChannelType gather_index,
	poser::SubsamplerCommonOption option
) : SubsampleProcessorBase(
	    MemoryContext::CpuMemory,
	    std::move(vertex),
	    std::move(gather_index),
	    std::move(option))
{
	float3 default_leaf_size = make_float3(0.01f, 0.01f, 0.01f);
	SetVoxelGridLeafSize(default_leaf_size);
}

void poser::VoxelGridSubsamplerCPU::SetVoxelGridLeafSize(const float3 &leaf_size) {
	LOG_ASSERT(!finalized_);
	LOG_ASSERT(leaf_size.x > 0);
	LOG_ASSERT(leaf_size.y > 0);
	LOG_ASSERT(leaf_size.z > 0);
	inv_leaf_size_.x = 1.0f / leaf_size.x;
	inv_leaf_size_.y = 1.0f / leaf_size.y;
	inv_leaf_size_.z = 1.0f / leaf_size.z;
}

void poser::VoxelGridSubsamplerCPU::CheckAndAllocate(
	const poser::FeatureMap &in_cloud_map,
	poser::FeatureMap &out_cloud_map
) {
	//The basic checking
	SubsampleProcessorBase::CheckAndAllocate(in_cloud_map, out_cloud_map);
	
	//For allcation of voxel map
	const auto max_voxel_size = out_cloud_map.GetDenseFeatureCapacity();
	voxel_map_.clear();
	voxel_map_.reserve(max_voxel_size);
}

void poser::VoxelGridSubsamplerCPU::processInternal(
	const FeatureMap &in_cloud_map,
	FeatureMap &out_cloud_map,
	const VoxelGridSubsamplerCPU::PointFilter &filter
) {
	//Get the input cloud
	const auto vertex_in = in_cloud_map.GetTypedFeatureValueReadOnly<float4>(vertex_channel_, MemoryContext::CpuMemory);
	
	//The bounding box
	const float3& aabb_min = subsampler_option_.bounding_box_min;
	const float3& aabb_max = subsampler_option_.bounding_box_max;
	
	//Iterate through the input cloud
	voxel_map_.clear();
	for(unsigned i = 0; i < vertex_in.Size(); i++) {
		//Check it against bounding box
		const auto& vertex_i = vertex_in[i];
		if(!filter(i, vertex_i))
			continue;
		
		//This is within bounding box
		int3 voxel_location;
		voxel_location.x = int(rintf(vertex_i.x * inv_leaf_size_.x));
		voxel_location.y = int(rintf(vertex_i.y * inv_leaf_size_.y));
		voxel_location.z = int(rintf(vertex_i.z * inv_leaf_size_.z));
		
		auto iter = voxel_map_.find(voxel_location);
		if(iter == voxel_map_.end()) { //This is an new voxel
			VoxelInfo info;
			info.gather_index = i;
			info.weighted_vertex = vertex_i;
			info.weighted_vertex.w = 1.0f;
			voxel_map_.emplace(voxel_location, info);
		} else {
			VoxelInfo& info = iter->second;
			info.weighted_vertex.x += vertex_i.x;
			info.weighted_vertex.y += vertex_i.y;
			info.weighted_vertex.z += vertex_i.z;
			info.weighted_vertex.w += 1.0f;
		}
	}
	
	//Get the output cloud
	out_cloud_map.ResizeDenseFeatureOrException(TensorDim(voxel_map_.size()));
	auto vertex_out = out_cloud_map.GetTypedFeatureValueReadWrite<float4>(vertex_channel_, MemoryContext::CpuMemory);
	auto index_out = out_cloud_map.GetTypedFeatureValueReadWrite<unsigned>(gather_index_, MemoryContext::CpuMemory);
	unsigned offset = 0;
	for(auto iter = voxel_map_.begin(); iter != voxel_map_.end(); iter++) {
		//Get the input
		const VoxelInfo& info = iter->second;
		const auto& weighted_vertex = info.weighted_vertex;
		
		//Get the output
		auto& vertex_out_i = vertex_out[offset];
		auto& index_out_i = index_out[offset];
		
		//Write it
		index_out_i = info.gather_index;
		vertex_out_i.x = weighted_vertex.x / weighted_vertex.w;
		vertex_out_i.y = weighted_vertex.y / weighted_vertex.w;
		vertex_out_i.z = weighted_vertex.z / weighted_vertex.w;
		vertex_out_i.w = 1.0f;
		
		//The counter
		offset++;
	}
}

void poser::VoxelGridSubsamplerCPU::Process(
	const poser::FeatureMap &in_cloud_map,
	poser::FeatureMap &out_cloud_map
) {
	//The bounding box
	const float3& aabb_min = subsampler_option_.bounding_box_min;
	const float3& aabb_max = subsampler_option_.bounding_box_max;
	
	if(subsampler_option_.foreground_mask.is_valid()) {
		//Get the filtered mask
		auto foreground_mask = in_cloud_map.GetFeatureValueReadOnly(subsampler_option_.foreground_mask, MemoryContext::CpuMemory);
		
		//Do it
		LOG_ASSERT(foreground_mask.TypeByte() == sizeof(unsigned char));
		auto filter_prog = [&](int idx_in_input, const float4& vertex) -> bool {
			//Check the position
			if(!vertex_in_aabb(vertex, aabb_min, aabb_max))
				return false;
			
			//Check the mask
			if(foreground_mask.At<unsigned char>(idx_in_input) == 0)
				return false;
			
			return true;
		};
		
		//Hand in to processor
		processInternal(in_cloud_map, out_cloud_map, filter_prog);
	} else {
		//The filter program
		auto filter_prog = [&](int idx_in_input, const float4& vertex) -> bool {
			//Check if the vertex is in
			return vertex_in_aabb(vertex, aabb_min, aabb_max);
		};
		
		//Hand in to processor
		processInternal(in_cloud_map, out_cloud_map, filter_prog);
	}
}