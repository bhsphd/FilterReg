//
// Created by wei on 12/5/18.
//

#include "cloudproc/normal_kdtree.h"
#include "geometry_utils/vector_operations.hpp"
#include "geometry_utils/eigen_pca33.h"


poser::NormalEstimationKDTree::NormalEstimationKDTree(
	poser::FeatureChannelType vertex,
	poser::FeatureChannelType normal
) : vertex_channel_(std::move(vertex)),
    normal_channel_(std::move(normal)) {
	if(!vertex_channel_.is_valid())
		vertex_channel_ = CommonFeatureChannelKey::ObservationVertexCamera();
	if(!normal_channel_.is_valid())
		normal_channel_ = CommonFeatureChannelKey::ObservationNormalCamera();
}

void poser::NormalEstimationKDTree::CheckAndAllocate(poser::FeatureMap &cloud_feature_map) {
	//Allocate the buffer at map
	LOG_ASSERT(cloud_feature_map.ExistFeature(vertex_channel_, MemoryContext::CpuMemory));
	cloud_feature_map.AllocateDenseFeature(normal_channel_, MemoryContext::CpuMemory);
	
	//Allocate internal buffer
	const auto capacity = cloud_feature_map.GetDenseFeatureCapacity();
	const auto size = cloud_feature_map.GetDenseFeatureDim().total_size();
	knn_index_.Reserve(capacity, sizeof(int) * max_knn_, MemoryContext::CpuMemory);
	knn_dist_.Reserve(capacity, sizeof(float) * max_knn_, MemoryContext::CpuMemory);
	knn_index_.ResizeOrException(size);
	knn_dist_.ResizeOrException(size);
}

void poser::NormalEstimationKDTree::Process(poser::FeatureMap &cloud_feature_map) {
	//Build kdtree
	const auto vertex_untyped = cloud_feature_map.GetFeatureValueReadOnly(vertex_channel_, MemoryContext::CpuMemory);
	kdtree_.ResetInputData(vertex_untyped);
	
	//Search it
	knn_index_.ResizeOrException(vertex_untyped.Size());
	knn_dist_.ResizeOrException(vertex_untyped.Size());
	kdtree_.SearchRadiusKNN(vertex_untyped, search_radius_, max_knn_, knn_index_.GetTensorReadWrite(), knn_dist_.GetTensorReadWrite());
	
	//Compute the normal
	const auto vertex = cloud_feature_map.GetTypedFeatureValueReadOnly<float4>(vertex_channel_, MemoryContext::CpuMemory);
	const auto knn = knn_index_.GetTensorReadOnly();
	auto normal = cloud_feature_map.GetTypedFeatureValueReadWrite<float4>(normal_channel_, MemoryContext::CpuMemory);
	for(auto i = 0; i < normal.Size(); i++) {
		const float4& vertex_i = vertex[i];
		auto& normal_i = normal[i];
		const auto knn_i = knn.ElemVectorAt<int>(i);
		float3 computed_normal = ComputeNormal(vertex, vertex_i, knn_i);
		
		//Write to result
		*(float3*)(&normal_i) = computed_normal;
		normal_i.w = 0.0f;
	}
}

float3 poser::NormalEstimationKDTree::ComputeNormal(
	const poser::TensorView<float4> &cloud,
	const float4 &vertex,
	const poser::BlobView::ElemVector<int> &knn
) {
	float cumulants[9] = {0};
	int counter = 0;
	for(auto i = 0; i < knn.typed_size; i++) {
		auto knn_i = knn[i];
		
		//The invalid knn is -1
		if(knn_i < 0 || knn_i >= cloud.Size())
			break;
		
		//OK
		const float4 p = cloud[knn_i];
		cumulants[0] += p.x;
		cumulants[1] += p.y;
		cumulants[2] += p.z;
		cumulants[3] += p.x * p.x;
		cumulants[4] += p.x * p.y;
		cumulants[5] += p.x * p.z;
		cumulants[6] += p.y * p.y;
		cumulants[7] += p.y * p.z;
		cumulants[8] += p.z * p.z;
		counter++;
	}
	
	//Compute the center
	const float inv_size = 1.0f / float(counter);
	for(auto k = 0; k < 9; k++) cumulants[k] *= inv_size;
	float covariance[6] = { 0 };
	covariance[0] = cumulants[3] - cumulants[0] * cumulants[0]; //(0, 0)
	covariance[1] = cumulants[4] - cumulants[0] * cumulants[1]; //(0, 1)
	covariance[2] = cumulants[5] - cumulants[0] * cumulants[2]; //(0, 2)
	covariance[3] = cumulants[6] - cumulants[1] * cumulants[1]; //(1, 1)
	covariance[4] = cumulants[7] - cumulants[1] * cumulants[2]; //(1, 2)
	covariance[5] = cumulants[8] - cumulants[2] * cumulants[2]; //(2, 2)
	
	//The eigen value for normal
	eigen_pca33 eigen(covariance);
	float3 normal3;
	eigen.compute(normal3);
	normalize(normal3);
	return normal3;
}

