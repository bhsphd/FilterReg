//
// Created by wei on 11/29/18.
//

#include "kinematic/affine/affine_pt2pt_analytical_cpu.h"

void poser::AffinePoint2PointAnalyticalCPU::CheckAndAllocate(
	const poser::FeatureMap &geometric_model,
	const poser::AffineKinematicModel &kinematic_model,
	const poser::DenseGeometricTarget &target
) {
	//The live vertex must be on cpu
	const auto& live_vertex_channel = kinematic_model.LiveVertexChannel();
	LOG_ASSERT(geometric_model.ExistFeature(live_vertex_channel, MemoryContext::CpuMemory));
	LOG_ASSERT(geometric_model.GetDenseFeatureDim().total_size() == target.GetTargetFlattenSize());
}

void poser::AffinePoint2PointAnalyticalCPU::CheckAndAllocate(
	const poser::FeatureMap &geometric_model,
	const poser::AffineKinematicModel &kinematic_model,
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

void poser::AffinePoint2PointAnalyticalCPU::ComputeAffineTransform(
	const poser::FeatureMap &geometric_model,
	const poser::AffineKinematicModel &kinematic_model,
	const poser::DenseGeometricTarget &target,
	Eigen::Ref<Eigen::Matrix4f> model2target
) {
	//Fetch the vertex
	const auto& ref_vertex_channel = kinematic_model.ReferenceVertexChannel();
	const auto vertex_from = geometric_model.GetTypedFeatureValueReadOnly<float4>(ref_vertex_channel, MemoryContext::CpuMemory);
	
	//Fetch the target
	const auto target_v = target.GetTargetVertexReadOnly();
	LOG_ASSERT(vertex_from.Size() == target_v.Size());
	
	//Do it
	ComputeTransformBetweenClouds(
		vertex_from.Size(),
		vertex_from.RawPtr(), target_v.RawPtr(),
		model2target);
}

void poser::AffinePoint2PointAnalyticalCPU::ComputeTransformBetweenClouds(
	int cloud_size,
	const float4 *model, const float4 *target,
	Eigen::Ref<Eigen::Matrix4f> model2target
) {
	//The jtj and jer for each for
	Eigen::Matrix4f jtj; jtj.setZero();
	Eigen::Vector4f jte_x, jte_y, jte_z;
	jte_x.setZero(); jte_y.setZero(); jte_z.setZero();
	
	//The assembler
	float jacobian[4];
	float residual;
	auto add_to_jtj = [&](float vertex_weight) -> void {
		for(auto i = 0; i < 4; i++) {
			for(auto j = i; j < 4; j++) {
				jtj(i, j) += (vertex_weight) * (jacobian[i] * jacobian[j]);
			}
		}
	};
	
	auto add_to_jte = [&](float vertex_weight, Eigen::Vector4f& jte) -> void {
		//Reduce on vector
		for (int i = 0; i < 4; i++) {
			float data = (vertex_weight) * (-residual * jacobian[i]);
			jte(i) += data;
		}
	};
	
	//Do processing
	for(auto i = 0; i < cloud_size; i++) {
		//Get the ith element
		const auto& vertex_from_i = model[i];
		const auto& target_v_i = target[i];
		
		//Insert it, the jacobian is the same for all x, y, z
		jacobian[0] = vertex_from_i.x;
		jacobian[1] = vertex_from_i.y;
		jacobian[2] = vertex_from_i.z;
		jacobian[3] = 1.0f;
		add_to_jtj(target_v_i.w);
		
		//For x
		residual = - target_v_i.x;
		add_to_jte(target_v_i.w, jte_x);
		
		//For y
		residual = - target_v_i.y;
		add_to_jte(target_v_i.w, jte_y);
		
		//For z
		residual = - target_v_i.z;
		add_to_jte(target_v_i.w, jte_z);
	}
	
	//Complete the matrix
	for(auto i = 0; i < 4; i++) {
		for(auto j = i + 1; j < 4; j++) {
			jtj(j, i) = jtj(i, j);
		}
	}
	
	//Solve it
	const auto jtj_ldlt = jtj.ldlt();
	Eigen::Vector4f dx = jtj_ldlt.solve(jte_x);
	Eigen::Vector4f dy = jtj_ldlt.solve(jte_y);
	Eigen::Vector4f dz = jtj_ldlt.solve(jte_z);
	
	//Store it
	model2target.setIdentity();
	model2target.row(0) = dx;
	model2target.row(1) = dy;
	model2target.row(2) = dz;
}