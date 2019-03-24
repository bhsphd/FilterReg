//
// Created by wei on 10/26/18.
//

#include "kinematic/rigid/rigid_point2plane_cpu.h"

void poser::RigidPoint2PlaneTermAssemblerCPU::CheckAndAllocate(
	const poser::FeatureMap &geometric_model,
	const poser::RigidKinematicModel &kinematic_model,
	const poser::DenseGeometricTarget &target
) {
	//Must be dense
	LOG_ASSERT(!target.IsSparseTarget());
	
	//The live vertex must be on cpu
	const auto& live_vertex_channel = kinematic_model.LiveVertexChannel();
	LOG_ASSERT(geometric_model.ExistFeature(live_vertex_channel, MemoryContext::CpuMemory));
	LOG_ASSERT(geometric_model.GetDenseFeatureDim().total_size() == target.GetTargetFlattenSize());
	
	//Must have normal
	LOG_ASSERT(target.has_normal());
}

void poser::RigidPoint2PlaneTermAssemblerCPU::ProcessAssemble(
	const poser::FeatureMap &geometric_model,
	const poser::RigidKinematicModel &kinematic_model,
	const poser::DenseGeometricTarget &target,
	Eigen::Ref<Eigen::Matrix6f> JtJ,
	Eigen::Ref<Eigen::Vector6f> Jt_error
) {
	//Fetch the vertex
	const auto& live_vertex_channel = kinematic_model.LiveVertexChannel();
	const auto live_vertex = geometric_model.GetTypedFeatureValueReadOnly<float4>(live_vertex_channel, MemoryContext::CpuMemory);
	
	//Fetch the target
	const auto vertex_target = target.GetTargetVertexReadOnly();
	const auto normal_target = target.GetTargetNormalReadOnly();
	
	//Check it
	LOG_ASSERT(live_vertex.Size() == vertex_target.Size());
	LOG_ASSERT(live_vertex.Size() == normal_target.Size());
	
	//Zero init
	JtJ.setZero();
	Jt_error.setZero();
	
	//The method to add the element to matrix
	float jacobian[6] = {0};
	float residual = 0;
	auto add_to_mat = [&](float vertex_weight = 1.0f) -> void {
		for (int i = 0; i < 6; i++) { //Row index
			for (int j = i; j < 6; j++) { //Column index, the matrix is symmetry
				float jacobian_ij = (vertex_weight * residual_weight_) * (jacobian[i] * jacobian[j]);
				JtJ(i, j) += jacobian_ij;
			}
		}
		
		//Reduce on vector
		for (int i = 0; i < 6; i++) {
			float data = (vertex_weight * residual_weight_) * (-residual * jacobian[i]);
			Jt_error(i) += data;
		}
	};
	
	//Iterate over points and compute jacobian
	for(auto i = 0; i < live_vertex.Size(); i++) {
		const auto& vertex_i = live_vertex[i];
		const auto& target_i = vertex_target[i];
		const auto& normal_i = normal_target[i];
		
		//Into 3 channel
		const float3& vertex_i_float3 = *((const float3*)(&vertex_i));
		const float3& target_i_float3 = *((const float3*)(&target_i));
		const float3& normal_i_float3 = *((const float3*)(&normal_i));
		
		//Only compute the first three channel
		residual = dot(normal_i_float3, vertex_i_float3 - target_i_float3);
		*(float3*)jacobian = cross(vertex_i_float3, normal_i_float3);
		*(float3*)(jacobian + 3) = normal_i_float3;
		add_to_mat(target_i.w);
	}
	
	//The other part
	for (int i = 0; i < 6; i++) { //Row index
		for (int j = i + 1; j < 6; j++) { //Column index, the matrix is symmetry
			JtJ(j, i) = JtJ(i, j);
		}
	}
}