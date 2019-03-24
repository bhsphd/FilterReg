//
// Created by wei on 9/16/18.
//

#include "cloudproc/pixel_subsampler_cpu.h"
#include "common/feature_channel_type.h"
#include "geometry_utils/vector_operations.hpp"

poser::PixelSubsampleProcessorCPU::PixelSubsampleProcessorCPU(
	FeatureChannelType vertex,
	FeatureChannelType gather_index,
	poser::SubsamplerCommonOption option)
	: SubsampleProcessorBase(
		MemoryContext::CpuMemory,
		std::move(vertex),
		std::move(gather_index),
		std::move(option)),
	  boundary_clip_pixel_(10), subsample_stride_(5)
{}

void poser::PixelSubsampleProcessorCPU::processInternal(
	const FeatureMap &in_image_map,
	FeatureMap &out_cloud_map,
	const PixelSubsampleProcessorCPU::PixelFilter &filter
) {
	//Get the map
	const auto vertex_map = in_image_map.GetTypedFeatureValueReadOnly<float4>(vertex_channel_, MemoryContext::CpuMemory);
	
	//Get the output: the size is max capacity now
	auto output_vertex = out_cloud_map.GetTypedFeatureValueReadWrite<float4>(vertex_channel_, MemoryContext::CpuMemory);
	auto output_index = out_cloud_map.GetTypedFeatureValueReadWrite<unsigned>(gather_index_, MemoryContext::CpuMemory);
	
	//Iterate inside the map
	unsigned valid_counter = 0;
	for(auto y = boundary_clip_pixel_; y < vertex_map.Rows() - boundary_clip_pixel_; y += subsample_stride_) {
		for(auto x = boundary_clip_pixel_; x < vertex_map.Cols() - boundary_clip_pixel_; x += subsample_stride_) {
			//Get the vertex and check it
			const float4 vertex = vertex_map(y, x);
			if(!filter(y, x, vertex))
				continue;
			
			//This should be valid, it's time for gathering
			output_vertex[valid_counter] = vertex;
			output_index[valid_counter] = x + y * vertex_map.Cols();
			
			//Increase the counter
			valid_counter++;
		}
	}
	
	//Resize all the output counter
	out_cloud_map.ResizeDenseFeatureOrException(TensorDim(valid_counter));
}

void poser::PixelSubsampleProcessorCPU::Process(const poser::FeatureMap &in_image_map, poser::FeatureMap &out_cloud_map) {
	//The bounding box
	const float3& aabb_min = subsampler_option_.bounding_box_min;
	const float3& aabb_max = subsampler_option_.bounding_box_max;
	
	if(subsampler_option_.foreground_mask.is_valid()) {
		//Get the filtered mask
		auto foreground_mask = in_image_map.GetFeatureValueReadOnly(subsampler_option_.foreground_mask, MemoryContext::CpuMemory);
		
		//Do it
		LOG_ASSERT(foreground_mask.TypeByte() == sizeof(unsigned char));
		auto filter_prog = [&](int r_idx, int c_idx, const float4& vertex) -> bool {
			//Check the position
			if(!vertex_in_aabb(vertex, aabb_min, aabb_max))
				return false;
			
			//Check the mask
			if(foreground_mask.At<unsigned char>(r_idx, c_idx) == 0)
				return false;
			
			return true;
		};
		
		//Hand in to processor
		processInternal(in_image_map, out_cloud_map, filter_prog);
	} else {
		//The filter program
		auto filter_prog = [&](int r_idx, int c_idx, const float4& vertex) -> bool {
			//Check the position
			return vertex_in_aabb(vertex, aabb_min, aabb_max);
		};
		
		//Hand in to processor
		processInternal(in_image_map, out_cloud_map, filter_prog);
	}
}