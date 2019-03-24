//
// Created by wei on 11/27/18.
//

#include "geometry_utils/device2eigen.h"
#include "kinematic/rigid/rigid_point2point_kabsch.h"

#include <Eigen/SVD>


void poser::RigidPoint2PointKabsch::CheckAndAllocate(
	const poser::FeatureMap &geometric_model,
	const poser::RigidKinematicModel &kinematic_model,
	const poser::GeometricTargetBase &target
) {
	const auto target_size = geometric_model.GetDenseFeatureCapacity();
	LOG_ASSERT(target_size > 0);
	centralized_model_.Reset<float4>(target_size);
	centralized_target_.Reset<float4>(target_size);
	sparse_cloud_buffer_.Reset<float4>(target_size);
}

void poser::RigidPoint2PointKabsch::ComputeTransformToTarget(
	const poser::FeatureMap &geometric_model,
	const poser::RigidKinematicModel &kinematic_model,
	const poser::DenseGeometricTarget &target,
	poser::mat34 &model2target
) {
	//The center for model and target
	const auto target_vertex = target.GetTargetVertexReadOnly();
	const auto& ref_vertex_channel = kinematic_model.ReferenceVertexChannel();
	const auto model_vertex = geometric_model.GetTypedFeatureValueReadOnly<float4>(ref_vertex_channel, MemoryContext::CpuMemory);
	LOG_ASSERT(model_vertex.Size() == target_vertex.Size());
	LOG_ASSERT(model_vertex.Size() <= centralized_model_.TensorFlattenSize());
	LOG_ASSERT(model_vertex.Size() <= centralized_target_.TensorFlattenSize());
	
	//Do it
	ComputeTransformBetweenClouds(
		model_vertex.Size(),
		model_vertex.RawPtr(), target_vertex.RawPtr(),
		(float4*)centralized_model_.RawPtr(), (float4*)centralized_target_.RawPtr(),
		model2target);
}

void poser::RigidPoint2PointKabsch::ComputeTransformToTarget(
	const poser::FeatureMap &geometric_model,
	const poser::RigidKinematicModel &kinematic_model,
	const poser::SparseGeometricTarget &target,
	poser::mat34 &model2target
) {
	//The center for model and target
	const auto target_vertex = target.GetTargetVertexReadOnly();
	const auto& ref_vertex_channel = kinematic_model.ReferenceVertexChannel();
	const auto model_vertex_all = geometric_model.GetTypedFeatureValueReadOnly<float4>(ref_vertex_channel, MemoryContext::CpuMemory);
	
	//Select the matched model vertex
	sparse_cloud_buffer_.ResizeOrException(target.GetTargetFlattenSize());
	const auto target_model_idx = target.GetTargetModelIndexReadOnly();
	auto selected_vertex = sparse_cloud_buffer_.GetTypedTensorReadWrite<float4>();
	for(auto i = 0; i < target.GetTargetFlattenSize(); i++) {
		selected_vertex[i] = model_vertex_all[target_model_idx[i]];
	}
	
	//Do it
	ComputeTransformBetweenClouds(
		selected_vertex.Size(),
		selected_vertex.RawPtr(), target_vertex.RawPtr(),
		(float4*)centralized_model_.RawPtr(), (float4*)centralized_target_.RawPtr(),
		model2target);
}

void poser::RigidPoint2PointKabsch::ComputeTransformBetweenClouds(
	int cloud_size,
	const float4 *model,
	const float4 *target,
	float4 *centralized_model, float4 *centralized_target,
	poser::mat34 &model2target
) {
	//Compute the center
	Eigen::Vector3f model_center; model_center.setZero();
	Eigen::Vector3f target_center; target_center.setZero();
	float total_weight = 0.0f;
	for(auto i = 0; i < cloud_size; i++) {
		const float weight = target[i].w;
		total_weight += weight;
		
		model_center(0) += weight * model[i].x;
		model_center(1) += weight * model[i].y;
		model_center(2) += weight * model[i].z;
		
		target_center(0) += weight * target[i].x;
		target_center(1) += weight * target[i].y;
		target_center(2) += weight * target[i].z;
	}
	float divided_by = 1.0f / total_weight;
	model_center *= divided_by;
	target_center *= divided_by;
	
	//Centralize them
	for(auto i = 0; i < cloud_size; i++) {
		const auto& model_i = model[i];
		auto& centralized_model_i = centralized_model[i];
		centralized_model_i.x = model_i.x - model_center(0);
		centralized_model_i.y = model_i.y - model_center(1);
		centralized_model_i.z = model_i.z - model_center(2);
		
		const auto& target_i = target[i];
		auto& centralized_target_i = centralized_target[i];
		centralized_target_i.x = target_i.x - target_center(0);
		centralized_target_i.y = target_i.y - target_center(1);
		centralized_target_i.z = target_i.z - target_center(2);
	}
	
	//Compute the H matrix
	float h_weight = 0.0f;
	Eigen::Matrix3f H; H.setZero();
	for(auto k = 0; k < cloud_size; k++) {
		const auto* model_k = (const float*)&(centralized_model[k]);
		const auto* target_k = (const float*)&(centralized_target[k]);
		const float this_weight = target[k].w;
		h_weight += this_weight * this_weight;
		for(auto i = 0; i < 3; i++) {
			for(auto j = 0; j < 3; j++) {
				H(i, j) += (this_weight * this_weight) * model_k[i] * target_k[j];
			}
		}
	}
	
	//Do svd
	using namespace Eigen;
	H /= h_weight;
	Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, ComputeFullU | ComputeFullV);
	Vector3f S = Vector3f::Ones(3);
	S(2) = (svd.matrixU() * svd.matrixV()).determinant();
	Eigen::Matrix3f R = svd.matrixV() * S.asDiagonal() * svd.matrixU().transpose();
	
	//The translation
	Vector3f translation = target_center;
	translation -= R * model_center;
	
	//The result
	Eigen::Isometry3f transform; transform.setIdentity();
	transform.linear() = R;
	transform.translation() = translation;
	model2target = mat34(transform);
}

