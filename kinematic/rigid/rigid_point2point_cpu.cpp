//
// Created by wei on 9/18/18.
//

#include "kinematic/rigid/rigid_point2point_cpu.h"


void poser::RigidPoint2PointTermAssemblerCPU::CheckAndAllocate(
	const poser::FeatureMap &geometric_model,
	const poser::RigidKinematicModel &kinematic_model,
	const poser::DenseGeometricTarget &target
) {
	//The live vertex must be on cpu
	const auto& live_vertex_channel = kinematic_model.LiveVertexChannel();
	LOG_ASSERT(geometric_model.ExistFeature(live_vertex_channel, MemoryContext::CpuMemory));
	LOG_ASSERT(geometric_model.GetDenseFeatureDim().total_size() == target.GetTargetFlattenSize());
}

void poser::RigidPoint2PointTermAssemblerCPU::CheckAndAllocate(
	const poser::FeatureMap &geometric_model,
	const poser::RigidKinematicModel &kinematic_model,
	const poser::SparseGeometricTarget &target
) {
	//The reference vertex on cpu should always exist
	const auto channel = target.GetSparseFeatureChannel();
	if(channel.is_valid())
		LOG_ASSERT(geometric_model.ExistFeature(channel, MemoryContext::CpuMemory));
	
	//Check the size
	const auto target_size = target.GetTargetFlattenSize();
	const auto index = target.GetTargetModelIndexReadOnly();
	LOG_ASSERT(index.Size() == target_size);
}

/* The processing interface for dense and sparse term
 */
void poser::RigidPoint2PointTermAssemblerCPU::ProcessAssemble(
	const poser::FeatureMap &geometric_model,
	const poser::RigidKinematicModel &kinematic_model,
	const poser::DenseGeometricTarget &target,
	//The output is directly in cpu
	Eigen::Ref<Eigen::Matrix6f> JtJ, Eigen::Ref<Eigen::Vector6f> Jt_error
) {
	//Fetch the vertex
	const auto& live_vertex_channel = kinematic_model.LiveVertexChannel();
	const auto live_vertex = geometric_model.GetTypedFeatureValueReadOnly<float4>(live_vertex_channel, MemoryContext::CpuMemory);
	
	//Fetch the target
	const auto vertex_target = target.GetTargetVertexReadOnly();
	
	//Check it
	LOG_ASSERT(live_vertex.Size() == vertex_target.Size());
	
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
		
		//Only compute the first three channel
		residual = vertex_i.x - target_i.x;
		*(float3*)jacobian = make_float3(0.0f, vertex_i.z, -vertex_i.y);
		*(float3*)(jacobian + 3) = make_float3(1.0f, 0.0f, 0.0f);
		add_to_mat(target_i.w);
		
		residual = vertex_i.y - target_i.y;
		*(float3*)jacobian = make_float3(-vertex_i.z, 0.0f, vertex_i.x);
		*(float3*)(jacobian + 3) = make_float3(0.0f, 1.0f, 0.0f);
		add_to_mat(target_i.w);
		
		residual = vertex_i.z - target_i.z;
		*(float3*)jacobian = make_float3(vertex_i.y, -vertex_i.x, 0.0f);
		*(float3*)(jacobian + 3) = make_float3(0.0f, 0.0f, 1.0f);
		add_to_mat(target_i.w);
	}
	
	//The other part
	for (int i = 0; i < 6; i++) { //Row index
		for (int j = i + 1; j < 6; j++) { //Column index, the matrix is symmetry
			JtJ(j, i) = JtJ(i, j);
		}
	}
}


void poser::RigidPoint2PointTermAssemblerCPU::ProcessAssemble(
	const poser::FeatureMap &geometric_model,
	const poser::RigidKinematicModel &kinematic_model,
	const poser::SparseGeometricTarget &target,
	Eigen::Ref<Eigen::Matrix6f> JtJ, Eigen::Ref<Eigen::Vector6f> Jt_error
) {
	//Get information from sparse target
	LOG_ASSERT(target.IsSparseTarget());
	const auto vertex_target = target.GetTargetVertexReadOnly();
	
	//Get information from geometric target
	const auto vertex_index = target.GetTargetModelIndexReadOnly();
	LOG_ASSERT(vertex_index.Size() == vertex_target.Size());
	
	//Assuming the existence of live vertex
	const auto ref_vertex = geometric_model.GetTypedFeatureValueReadOnly<float4>(kinematic_model.ReferenceVertexChannel(), MemoryContext::CpuMemory);
	
	//Zero init
	JtJ.setZero();
	Jt_error.setZero();
	
	//The method to add the element to matrix
	float jacobian[6] = {0};
	float residual = 0;
	auto add_to_mat = [&](float vertex_weight) -> void {
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
	
	const mat34& transform = kinematic_model.GetRigidTransform();
	for(auto i = 0; i < vertex_index.Size(); i++) {
		const auto& ref_vertex_i = ref_vertex[vertex_index[i]];
		float3 vertex_i = transform.rotation() * ref_vertex_i + transform.translation;
		const auto& target_i = vertex_target[i];
		
		//Only compute the first three channel
		residual = vertex_i.x - target_i.x;
		*(float3*)jacobian = make_float3(0.0f, vertex_i.z, -vertex_i.y);
		*(float3*)(jacobian + 3) = make_float3(1.0f, 0.0f, 0.0f);
		add_to_mat(target_i.w);
		
		residual = vertex_i.y - target_i.y;
		*(float3*)jacobian = make_float3(-vertex_i.z, 0.0f, vertex_i.x);
		*(float3*)(jacobian + 3) = make_float3(0.0f, 1.0f, 0.0f);
		add_to_mat(target_i.w);
		
		residual = vertex_i.z - target_i.z;
		*(float3*)jacobian = make_float3(vertex_i.y, -vertex_i.x, 0.0f);
		*(float3*)(jacobian + 3) = make_float3(0.0f, 0.0f, 1.0f);
		add_to_mat(target_i.w);
	}
	
	//The other part
	for (int i = 0; i < 6; i++) { //Row index
		for (int j = i + 1; j < 6; j++) { //Column index, the matrix is symmetry
			JtJ(j, i) = JtJ(i, j);
		}
	}
}