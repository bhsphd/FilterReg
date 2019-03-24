//
// Created by wei on 11/29/18.
//

#include "kinematic/affine/affine_geometric_update.h"
#include "geometry_utils/device2eigen.h"

void poser::UpdateLiveVertexCPU(
	const poser::AffineKinematicModel &kinematic,
	poser::FeatureMap &geometric_model
) {
	//Get the vertex
	const auto& ref_vertex_channel = kinematic.ReferenceVertexChannel();
	const auto& reference_vertex = geometric_model.GetTypedFeatureValueReadOnly<float4>(ref_vertex_channel, MemoryContext::CpuMemory);
	const auto& live_vertex_channel = kinematic.LiveVertexChannel();
	auto live_vertex = geometric_model.GetTypedFeatureValueReadWrite<float4>(live_vertex_channel, MemoryContext::CpuMemory);
	LOG_ASSERT(live_vertex.Size() == reference_vertex.Size());
	
	//Directly use matrix
	const Eigen::Affine3f& transform = kinematic.GetAffineTransformationEigen();
	Eigen::Vector3f transformed_v;
	for(auto i = 0; i < reference_vertex.Size(); i++) {
		const auto& ref_vertex_i = reference_vertex[i];
		Eigen::Map<const Eigen::Vector3f> mapped_vertex_i((const float*)(&reference_vertex[i]));
		transformed_v = transform * mapped_vertex_i;
		
		//Store the result
		auto& live_vertex_i = live_vertex[i];
		live_vertex_i.x = transformed_v(0);
		live_vertex_i.y = transformed_v(1);
		live_vertex_i.z = transformed_v(2);
		live_vertex_i.w = ref_vertex_i.w;
	}
}

void poser::UpdateLiveVertexAndNormalCPU(
	const poser::AffineKinematicModel &kinematic,
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
	const Eigen::Affine3f& transform = kinematic.GetAffineTransformationEigen();
	const Eigen::Matrix3f& transform_normal_eigen = transform.linear().inverse().transpose();
	const mat33 transform_normal(transform_normal_eigen);
	
	Eigen::Vector3f transformed_v;
	for(auto i = 0; i < reference_vertex.Size(); i++) {
		const auto& transform_from_v = reference_vertex[i];
		const auto& transform_from_n = reference_normal[i];
		auto& transform_to_v = live_vertex[i];
		auto& transform_to_n = live_normal[i];
		float3& transformed_n_vec3 = *(float3*)(&transform_to_n);
		
		//Do transform
		Eigen::Map<const Eigen::Vector3f> mapped_vertex_i((const float*)(&reference_vertex[i]));
		transformed_v = transform * mapped_vertex_i;
		transformed_n_vec3 = transform_normal * transform_from_n;
		
		//Store the result
		transform_to_v.x = transformed_v(0);
		transform_to_v.y = transformed_v(1);
		transform_to_v.z = transformed_v(2);
		transform_to_v.w = transform_from_v.w;
		transform_to_n.w = transform_from_n.w;
	}
}