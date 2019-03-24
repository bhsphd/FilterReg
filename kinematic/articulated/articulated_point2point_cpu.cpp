//
// Created by wei on 9/30/18.
//

#include "kinematic/articulated/articulated_point2point_cpu.h"
#include "kinematic/articulated/articulated_twist2joint.h"

void poser::ArticulatedPoint2PointTermAssemblerCPU::CheckAndAllocate(
	const poser::FeatureMap &geometric_model,
	const poser::ArticulatedKinematicModel& kinematic_model,
	const poser::GeometricTarget &target
) {
	if(target.IsSparseTarget()) {
		//The reference vertex on cpu should always exist
		const auto channel = target.GetSparseFeatureChannel();
		LOG_ASSERT(geometric_model.ExistFeature(channel, MemoryContext::CpuMemory));
		
		//Check the size
		const auto target_size = target.GetTargetFlattenSize();
		const auto index = geometric_model.GetSparseFeatureIndexReadOnly(channel.get_name_key(), MemoryContext::CpuMemory);
		LOG_ASSERT(index.Size() == target_size);
	} else {
		//The live vertex must be on cpu
		const auto& live_vertex_channel = kinematic_model.LiveVertexChannel();
		LOG_ASSERT(geometric_model.ExistFeature(live_vertex_channel, MemoryContext::CpuMemory));
		LOG_ASSERT(geometric_model.GetDenseFeatureDim().total_size() == target.GetTargetFlattenSize());
	}
	
	//Allocate internal buffer
	const auto& body_map = kinematic_model.GetBody2GeometricMap();
	body_twist_jtj_.resize(36 * body_map.size());
	body_twist_jte_.resize(6 * body_map.size());
}


void poser::ArticulatedPoint2PointTermAssemblerCPU::ProcessAssembleDenseTerm(
	const poser::FeatureMap &geometric_model,
	const poser::ArticulatedKinematicModel &kinematic_model,
	const poser::GeometricTarget &target,
	//The output
	Eigen::Ref<Eigen::MatrixXf> JtJ,
	Eigen::Ref<Eigen::VectorXf> JtError
) {
	//Fetch the vertex
	const auto &body_map = kinematic_model.GetBody2GeometricMap();
	const auto &live_vertex_channel = kinematic_model.LiveVertexChannel();
	const auto live_vertex = geometric_model.GetTypedFeatureValueReadOnly<float4>(live_vertex_channel, MemoryContext::CpuMemory);
	
	//Fetch the target
	const auto vertex_target = target.GetTargetVertexReadOnly();
	
	//Check it
	LOG_ASSERT(live_vertex.Size() == vertex_target.Size());
	float jacobian[6] = {0};
	float residual = 0;

	//Iterate over each rigid body
	for (auto body_i = 0; body_i < body_map.size(); body_i++) {
		Eigen::Map<Eigen::Matrix6f> twist_jtj_i(body_twist_jtj_.data() + 36 * body_i);
		Eigen::Map<Eigen::Vector6f> twist_jte_i(body_twist_jte_.data() + 6 * body_i);
		const unsigned start_idx = body_map[body_i].geometry_start;
		const unsigned end_idx = body_map[body_i].geometry_end;

		//Zero-init on the matrix
		twist_jtj_i.setZero();
		twist_jte_i.setZero();

		//The method to add the element to matrix
		auto add_to_mat = [&](float vertex_weight = 1.0f) -> void {
			for (int i = 0; i < 6; i++) { //Row index
				for (int j = i; j < 6; j++) { //Column index, the matrix is symmetry
					float jacobian_ij = (vertex_weight * residual_weight_) * (jacobian[i] * jacobian[j]);
					twist_jtj_i(i, j) += jacobian_ij;
				}
			}
		
			//Reduce on vector
			for (int i = 0; i < 6; i++) {
				float data = (vertex_weight * residual_weight_) * (-residual * jacobian[i]);
				twist_jte_i(i) += data;
			}
		};

		//Iterate over points and compute jacobian
		for (auto i = start_idx; i < end_idx; i++) {
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
				twist_jtj_i(j, i) = twist_jtj_i(i, j);
			}
		}
	}

	//Assemble the global matrix
	ArticulatedJacobianTwist2Joint::AssembleJointSpaceJacobian(
		kinematic_model, 
		body_twist_jtj_.data(), body_twist_jte_.data(),
		JtJ, JtError
	);
}