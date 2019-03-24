//
// Created by wei on 9/18/18.
//

#include "kinematic/rigid/rigid_geometric_update.h"

void poser::UpdateLiveVertexCPU(
	const poser::RigidKinematicModel &kinematic,
	poser::FeatureMap &geometric_model
) {
	//Get the vertex
	const auto& ref_vertex_channel = kinematic.ReferenceVertexChannel();
	const auto& reference_vertex = geometric_model.GetTypedFeatureValueReadOnly<float4>(ref_vertex_channel, MemoryContext::CpuMemory);
	const auto& live_vertex_channel = kinematic.LiveVertexChannel();
	auto live_vertex = geometric_model.GetTypedFeatureValueReadWrite<float4>(live_vertex_channel, MemoryContext::CpuMemory);
	LOG_ASSERT(live_vertex.Size() == reference_vertex.Size());
	
	//Get the motion parameter
	const auto& transform = kinematic.GetRigidTransform();
	
	//Do update
	for(auto i = 0; i < reference_vertex.Size(); i++) {
		auto& transform_to = live_vertex[i];
		float3& transform_to_vec3 = *(float3*)(&transform_to);
		const auto& transform_from = reference_vertex[i];
		transform_to_vec3 = transform.rotation() * transform_from + transform.translation;
		transform_to.w = transform_from.w;
	}
}

void poser::UpdateLiveVertexAndNormalCPU(
	const poser::RigidKinematicModel &kinematic,
	poser::FeatureMap &geometric_model
) {
	//Get the vertex
	const auto& ref_vertex_channel = kinematic.ReferenceVertexChannel();
	const auto& reference_vertex = geometric_model.GetTypedFeatureValueReadOnly<float4>(ref_vertex_channel, MemoryContext::CpuMemory);
	const auto& live_vertex_channel = kinematic.LiveVertexChannel();
	auto live_vertex = geometric_model.GetTypedFeatureValueReadWrite<float4>(live_vertex_channel, MemoryContext::CpuMemory);
	LOG_ASSERT(live_vertex.Size() == reference_vertex.Size());
	
	//Get the normal
	const auto& ref_normal_channel = kinematic.ReferenceNormalChannel();
	auto reference_normal = geometric_model.GetTypedFeatureValueReadOnly<float4>(ref_normal_channel, MemoryContext::CpuMemory);
	const auto& live_normal_channel = kinematic.LiveNormalChannel();
	auto live_normal = geometric_model.GetTypedFeatureValueReadWrite<float4>(live_normal_channel, MemoryContext::CpuMemory);
	
	//Get the motion parameter
	const auto& transform = kinematic.GetRigidTransform();
	
	//Do update
	for(auto i = 0; i < reference_vertex.Size(); i++) {
		auto& transformed_v = live_vertex[i];
		auto& transformed_n = live_normal[i];
		float3& transformed_v_vec3 = *(float3*)(&transformed_v);
		float3& transformed_n_vec3 = *(float3*)(&transformed_n);
		
		const auto& transform_from_v = reference_vertex[i];
		const auto& transform_from_n = reference_normal[i];
		transformed_v_vec3 = transform.rotation() * transform_from_v + transform.translation;
		transformed_n_vec3 = transform.rotation() * transform_from_n;
		
		//The last element stay the same
		transformed_v.w = transform_from_v.w;
		transformed_n.w = transform_from_n.w;
	}
}