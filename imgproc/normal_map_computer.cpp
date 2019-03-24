//
// Created by wei on 9/14/18.
//

#include "common/feature_channel_type.h"
#include "imgproc/normal_map_computer.h"
#include "geometry_utils/eigen_pca33.h"


poser::DepthNormalMapComputer::DepthNormalMapComputer(
	MemoryContext context,
	int window_halfsize,
	std::string depth_map, 
	std::string vertex_map,
	std::string normal_map) 
	: context_(context),
	  depth_map_key_(std::move(depth_map), sizeof(unsigned short)),
	  vertex_map_key_(std::move(vertex_map), sizeof(float4)),
	  normal_map_key_(std::move(normal_map), sizeof(float4)),
	  window_halfsize_(window_halfsize),
	  dense_threshold_((2 * window_halfsize + 1) * (2 * window_halfsize + 1) / 2)
{
	//Check and replace with default
	if(!depth_map_key_.is_valid())
		depth_map_key_ = CommonFeatureChannelKey::RawDepthImage();
	if(!vertex_map_key_.is_valid())
		vertex_map_key_ = CommonFeatureChannelKey::ObservationVertexCamera();
	if(!normal_map_key_.is_valid())
		normal_map_key_ = CommonFeatureChannelKey::ObservationNormalCamera();
}

void poser::DepthNormalMapComputer::CheckAndAllocate(poser::FeatureMap &feature_map) {
	//Check existance
	LOG_ASSERT(feature_map.ExistFeature(depth_map_key_, context_));
	LOG_ASSERT(feature_map.ExistFeature(vertex_map_key_, context_));
	
	//Check size
	auto depth_map = feature_map.GetTypedFeatureValueReadOnly<unsigned short>(depth_map_key_, context_);
	auto vertex_map = feature_map.GetTypedFeatureValueReadOnly<float4>(vertex_map_key_, context_);
	LOG_ASSERT(depth_map.Rows() > window_halfsize_ && depth_map.Cols() > window_halfsize_);
	LOG_ASSERT(vertex_map.Rows() == depth_map.Rows() && vertex_map.Cols() == depth_map.Cols());
	
	//Allocate normal map
	feature_map.AllocateDenseFeature<float4>(
		normal_map_key_,
		depth_map.DimensionalSize(),
		context_);
}

void poser::DepthNormalMapComputer::Process(poser::FeatureMap &feature_map) {
	ProcessStreamed(feature_map, 0);
}

void poser::DepthNormalMapComputer::ProcessStreamed(poser::FeatureMap &feature_map, cudaStream_t stream) {
	//Fetch the tensor
	auto depth_map = feature_map.GetTypedFeatureValueReadOnly<unsigned short>(depth_map_key_, context_);
	auto vertex_map = feature_map.GetTypedFeatureValueReadOnly<float4>(vertex_map_key_, context_);
	auto normal_map = feature_map.GetTypedFeatureValueReadWrite<float4>(normal_map_key_, context_);
	
	//Check the size
	LOG_ASSERT(depth_map.DimensionalSize() == vertex_map.DimensionalSize());
	LOG_ASSERT(depth_map.DimensionalSize() == normal_map.DimensionalSize());
	
	//Do it
	if(depth_map.IsCpuTensor()) {
		ComputeNormalMapPCACPU(depth_map, vertex_map, normal_map);
		//ComputeNormalCenteralDiffCPU(depth_map, vertex_map, normal_map);
	} else {
		ComputeNormalMapGPU(depth_map, vertex_map, normal_map, stream);
	}
}

void poser::DepthNormalMapComputer::ComputeNormalMapPCACPU(
	const poser::TensorView<unsigned short> &depth_map,
	const poser::TensorView<float4> &vertex_map,
	poser::TensorSlice<float4> normal_map
) {
	LOG_ASSERT(depth_map.IsCpuTensor());
	LOG_ASSERT(vertex_map.IsCpuTensor());
	LOG_ASSERT(normal_map.IsCpuTensor());
	
	//Iterate with the input
	for(int y = 0; y < vertex_map.Rows(); y++) {
		for(int x = 0; x < vertex_map.Cols(); x++) {
			//The x, y is on boundary
			if(y < window_halfsize_ || x < window_halfsize_
			|| y >= vertex_map.Rows() - window_halfsize_ || x >= vertex_map.Cols() - window_halfsize_) {
				normal_map(y, x) = make_float4(0, 0, 0, 0);
				continue;
			}
			
			//Check it
			const float4 vertex_center = vertex_map(y, x);
			if(is_zero_vertex(vertex_center)) {
				normal_map(y, x) = make_float4(0, 0, 0, 0);
				continue;
			}
			
			//The window search
			int counter = 0;
			float cumulants[9] = {0};
			for (int cy = y - window_halfsize_; cy <= y + window_halfsize_; cy += 1) {
				for (int cx = x - window_halfsize_; cx <= x + window_halfsize_; cx += 1) {
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
			if(counter < dense_threshold_) {
				normal_map(y, x) = make_float4(0, 0, 0, 0);
				continue;
			}
			
			//Compute the center
			const float inv_size = 1.0f / float(counter);
			for(auto i = 0; i < 9; i++) cumulants[i] *= inv_size;
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
			if (dotxyz(normal3, vertex_center) >= 0.0f) normal3 *= -1;
			
			//Save it
			normal_map(y, x) = make_float4(normal3.x, normal3.y, normal3.z, 0.0);
		}
	}
}

void poser::DepthNormalMapComputer::ComputeNormalCenteralDiffCPU(
	const poser::TensorView<unsigned short> &depth_map,
	const poser::TensorView<float4> &vertex_map,
	poser::TensorSlice<float4> normal_map
) {
	LOG_ASSERT(depth_map.IsCpuTensor());
	LOG_ASSERT(vertex_map.IsCpuTensor());
	LOG_ASSERT(normal_map.IsCpuTensor());
	
	//Iterate with the input
	for(int y = 0; y < vertex_map.Rows(); y++) {
		for(int x = 0; x < vertex_map.Cols(); x++) {
			//The x, y is on boundary
			if(y < window_halfsize_ || x < window_halfsize_
			   || y >= vertex_map.Rows() - window_halfsize_ || x >= vertex_map.Cols() - window_halfsize_) {
				normal_map(y, x) = make_float4(0, 0, 0, 0);
				continue;
			}
			
			//The vertex position
			const float4 vertex_center = vertex_map(y, x);
			const float4 vertex_right = vertex_map(y, x + 1);
			const float4 vertex_top = vertex_map(y - 1, x);
			
			//Compute the normal
			const float3 diff_x = make_float3(
				vertex_right.x - vertex_center.x,
				vertex_right.y - vertex_center.y,
				vertex_right.z - vertex_center.z);
			const float3 diff_y = make_float3(
				vertex_top.x - vertex_center.x,
				vertex_top.y - vertex_center.y,
				vertex_top.z - vertex_center.z);
			
			//Compute by cross product
			float3 normal3 = cross(diff_x, diff_y);
			normalize(normal3);
			if(std::isnan(normal3.x) || std::isnan(normal3.y) || std::isnan(normal3.z)) {
				normal_map(y, x) = make_float4(0, 0, 0, 0);
				continue;
			}
			
			//Correct the direction of normal
			if (dotxyz(normal3, vertex_center) >= 0.0f) normal3 *= -1;
			
			//Save it
			normal_map(y, x) = make_float4(normal3.x, normal3.y, normal3.z, 0.0);
		}
	}
}
