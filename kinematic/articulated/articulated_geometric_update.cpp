//
// Created by wei on 9/29/18.
//

#include "kinematic/articulated/articulated_geometric_update.h"

void poser::UpdateLiveVertexCPU(
	const poser::ArticulatedKinematicModel &kinematic,
	poser::FeatureMap &geometric_model
) {
	//Get the vertex
	const auto& ref_vertex_channel = kinematic.ReferenceVertexChannel();
	const auto& reference_vertex = geometric_model.GetTypedFeatureValueReadOnly<float4>(ref_vertex_channel, MemoryContext::CpuMemory);
	const auto& live_vertex_channel = kinematic.LiveVertexChannel();
	auto live_vertex = geometric_model.GetTypedFeatureValueReadWrite<float4>(live_vertex_channel, MemoryContext::CpuMemory);
	LOG_ASSERT(live_vertex.Size() == reference_vertex.Size());
	
	//Get the map and transform
	const auto& body2vertex_map = kinematic.GetBody2GeometricMap();
	const auto& body2world_vec = kinematic.GetBody2WorldTransformCPU();
	
	//Iterate over each body
	for(auto body_idx = 0; body_idx < body2vertex_map.size(); body_idx++) {
		const auto& body2world = body2world_vec[body_idx];
		auto start_idx = body2vertex_map[body_idx].geometry_start;
		auto end_idx = body2vertex_map[body_idx].geometry_end;
		
		//Iterate of geometric
		for(auto j = start_idx; j < end_idx; j++) {
			const auto& transform_from = reference_vertex[j];
			auto& transform_to = live_vertex[j];
			float3& transform_to_vec3 = *(float3*)(&transform_to);
			transform_to_vec3 = body2world.rot * transform_from + body2world.trans;
			transform_to.w = transform_from.w;
		}
	}
}