//
// Created by wei on 9/14/18.
//

#include "imgproc/vertex_map_computer.h"
#include "common/feature_channel_type.h"


poser::DepthVertexMapComputer::DepthVertexMapComputer(
	MemoryContext context,
	const poser::Intrinsic& intrinsic,
	std::string depth_img,
	std::string vertex_map)
	: context_(context),
	  depth_map_key_(std::move(depth_img), sizeof(unsigned short)),
	  vertex_map_key_(std::move(vertex_map), sizeof(float4)),
	  intrinsic_(intrinsic),
	  intrinsic_inv_(intrinsic)
{
	if(depth_map_key_.get_name_key().empty())
		depth_map_key_ = CommonFeatureChannelKey::FilteredDepthImage();
	if(vertex_map_key_.get_name_key().empty())
		vertex_map_key_ = CommonFeatureChannelKey::ObservationVertexCamera();
}

void poser::DepthVertexMapComputer::CheckAndAllocate(poser::FeatureMap &feature_map) {
	//Check the input
	LOG_ASSERT(feature_map.ExistFeature(depth_map_key_, context_));
	
	//Check the size
	auto dense_depth = feature_map.GetTypedFeatureValueReadOnly<unsigned short>(depth_map_key_, context_);
	//This must be a map, instead of a point cloud
	LOG_ASSERT(dense_depth.Rows() > 5);
	LOG_ASSERT(dense_depth.Cols() > 5);
	
	//Allocate the output
	feature_map.AllocateDenseFeature<float4>(
		vertex_map_key_,
		dense_depth.DimensionalSize(),
		context_);
}

void poser::DepthVertexMapComputer::Process(poser::FeatureMap &feature_map) {
	ProcessStreamed(feature_map, 0);
}

void poser::DepthVertexMapComputer::ProcessStreamed(poser::FeatureMap &feature_map, cudaStream_t stream) {
	auto depth_map = feature_map.GetTypedFeatureValueReadOnly<unsigned short>(depth_map_key_, context_);
	auto vertex_map = feature_map.GetTypedFeatureValueReadWrite<float4>(vertex_map_key_, context_);
	LOG_ASSERT(depth_map.DimensionalSize() == vertex_map.DimensionalSize());
	
	if(depth_map.IsCpuTensor()) {
		LOG_ASSERT(vertex_map.IsCpuTensor());
		ComputeVertexMapCPU(depth_map, vertex_map);
	} else {
		LOG_ASSERT(vertex_map.IsGpuTensor());
		ComputeVertexMapGPU(depth_map, vertex_map, stream);
	}
}

void poser::DepthVertexMapComputer::ComputeVertexMapCPU(
	const poser::TensorView<unsigned short> &depth_map,
	poser::TensorSlice<float4> vertex_map
) {
	for(auto y = 0; y < depth_map.Rows(); y++) {
		for(auto x = 0; x < depth_map.Cols(); x++) {
			float4 vertex;
			const auto raw_depth = depth_map(y, x); //y is row index
			vertex.z = float(raw_depth) / (1000.f); // into meter
			vertex.x = (float(x) - intrinsic_inv_.principal_x) * intrinsic_inv_.inv_focal_x * vertex.z;
			vertex.y = (float(y) - intrinsic_inv_.principal_y) * intrinsic_inv_.inv_focal_y * vertex.z;
			vertex.w = 1.0f;
			
			//Save it
			vertex_map(y, x) = vertex;
		}
	}
}