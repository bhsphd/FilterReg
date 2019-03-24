//
// Created by wei on 9/17/18.
//

#include "common/feature_channel_type.h"
#include "cloudproc/normal_map_index.h"
#include "geometry_utils/eigen_pca33.h"

poser::NormalEstimateWithMapIndex::NormalEstimateWithMapIndex(
	poser::Intrinsic intrinsic,
	poser::FeatureChannelType vertex,
	poser::FeatureChannelType normal
) : intrinsic_(intrinsic),
    vertex_channel_(std::move(vertex)),
    normal_channel_(std::move(normal))
{
	if(!vertex_channel_.is_valid())
		vertex_channel_ = CommonFeatureChannelKey::ObservationVertexCamera();
	if(!normal_channel_.is_valid())
		normal_channel_ = CommonFeatureChannelKey::ObservationNormalCamera();
}


void poser::NormalEstimateWithMapIndex::CheckAndAllocate(
	const poser::FeatureMap &img_feature_map,
	poser::FeatureMap &cloud_feature_map
) {
	//Vertex channel exist in both, and should be a map in image feature map
	LOG_ASSERT(img_feature_map.ExistFeature(vertex_channel_, MemoryContext::CpuMemory));
	auto vertex_map = img_feature_map.GetTypedDenseFeatureReadOnly<float4>(vertex_channel_.get_name_key(), MemoryContext::CpuMemory);
	LOG_ASSERT(vertex_map.Rows() > 2 * window_halfsize_);
	LOG_ASSERT(vertex_map.Cols() > 2 * window_halfsize_);
	
	//Allocate it
	LOG_ASSERT(cloud_feature_map.ExistFeature(vertex_channel_, MemoryContext::CpuMemory));
	cloud_feature_map.AllocateDenseFeature(normal_channel_, MemoryContext::CpuMemory);
}


void poser::NormalEstimateWithMapIndex::Process(
	const poser::FeatureMap &img_feature_map,
	poser::FeatureMap &cloud_feature_map
) {
	//Get the input
	auto vertex_map = img_feature_map.GetTypedDenseFeatureReadOnly<float4>(vertex_channel_.get_name_key(), MemoryContext::CpuMemory);
	auto point_cloud = cloud_feature_map.GetTypedFeatureValueReadOnly<float4>(vertex_channel_, MemoryContext::CpuMemory);
	
	//Get the output
	auto normal_cloud = cloud_feature_map.GetTypedFeatureValueReadWrite<float4>(normal_channel_, MemoryContext::CpuMemory);
	
	//The processing loop
	for(auto i = 0; i < point_cloud.FlattenSize(); i++) {
		//Get the point and normal
		const float4 vertex = point_cloud[i];
		float4& normal = normal_cloud[i];
		
		//Project it
		const int img_x = int(((vertex.x / (vertex.z + 1e-10)) * intrinsic_.focal_x) + intrinsic_.principal_x);
		const int img_y = int(((vertex.y / (vertex.z + 1e-10)) * intrinsic_.focal_y) + intrinsic_.principal_y);
		if(img_x < window_halfsize_ || img_x >= vertex_map.Cols() - window_halfsize_
		|| img_y < window_halfsize_ || img_y >= vertex_map.Rows() - window_halfsize_) {
			normal = make_float4(0, 0, 0, 0);
			continue;
		}
		
		//Do window search
		int counter = 0;
		float cumulants[9] = {0};
		for (int cy = img_y - window_halfsize_; cy <= img_y + window_halfsize_; cy += 1) {
			for (int cx = img_x - window_halfsize_; cx <= img_x + window_halfsize_; cx += 1) {
				//Must be inside window
				const float4 p = vertex_map(cy, cx);
				if (!is_zero_vertex(p)) {
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
			}
		}//End of first window search
		
		//Check density
		if(counter < (2 * window_halfsize_ + 1) * (2 * window_halfsize_ + 1) / 2) {
			normal = make_float4(0, 0, 0, 0);
			continue;
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
		if (dotxyz(normal3, vertex) >= 0.0f) normal3 *= -1;
		
		//Save it
		normal = make_float4(normal3.x, normal3.y, normal3.z, 0.0);
	}
}