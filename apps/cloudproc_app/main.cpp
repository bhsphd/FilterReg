//
// Created by wei on 9/14/18.
//

#include "imgproc/imgproc.h"
#include "cloudproc/cloudproc.h"
#include "visualizer/debug_visualizer.h"
#include "geometry_utils/vector_operations.hpp"

#include <chrono>
#include <fstream>
#include <cuda_runtime_api.h>

void test_cpu_process() {
	using namespace poser;
	using namespace std::chrono;
	
	//The feature map for image
	FeatureMap img_feature_map, cloud_feature_map;
	
	//Construct the processor
	std::string data_prefix("/home/wei/Documents/programs/surfelwarp/data/boxing");
	FrameLoaderFile loader(data_prefix);
	DepthBilateralFilter depth_filter;
	DepthVertexMapComputer vertex_map_processor;
	DepthNormalMapComputer normal_map_processor;
	
	PixelSubsampleProcessorCPU subsampler;
	//subsampler.SetSubsampleStride(3);
	//VoxelGridSubsamplerCPU subsampler;
	//subsampler.SetVoxelGridLeafSize(0.01f);
	
	//Gather required feature
	FeatureGatherProcessor feature_gather;
	feature_gather.InsertGatheredFeature(CommonFeatureChannelKey::ObservationNormalCamera());
	
	//NormalEstimateWithMapIndex normal_estimator;
	NormalEstimationKDTree normal_estimator;
	normal_estimator.SetSearchRadius(0.05f);
	
	//Allocate the buffer
	loader.CheckAndAllocate(img_feature_map);
	depth_filter.CheckAndAllocate(img_feature_map);
	vertex_map_processor.CheckAndAllocate(img_feature_map);
	normal_map_processor.CheckAndAllocate(img_feature_map);
	subsampler.CheckAndAllocate(img_feature_map, cloud_feature_map);
	feature_gather.CheckAndAllocate(img_feature_map, cloud_feature_map);
	normal_estimator.CheckAndAllocate(cloud_feature_map);
	
	//Do processing of image
	loader.UpdateFrameIndex(1);
	loader.LoadDepthImage(img_feature_map);
	loader.LoadColorImage(img_feature_map);
	depth_filter.Process(img_feature_map);
	vertex_map_processor.Process(img_feature_map);
	normal_map_processor.Process(img_feature_map);
	
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	//Do subsampling
	subsampler.Process(img_feature_map, cloud_feature_map);
	feature_gather.PerformGatherCPU(img_feature_map, cloud_feature_map);
	normal_estimator.Process(cloud_feature_map);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	auto delta_t_ms = duration_cast<milliseconds>(t2 - t1);
	LOG(INFO) << "The time is ms is " << delta_t_ms.count();
	
	
	//Do visualize
	auto depth_map = img_feature_map.GetTypedFeatureValueReadOnly<unsigned short>(CommonFeatureChannelKey::FilteredDepthImage(), MemoryContext::CpuMemory);
	auto rgba_map = img_feature_map.GetTypedFeatureValueReadOnly<uchar4>(CommonFeatureChannelKey::RawRGBImage(), MemoryContext::CpuMemory);
	auto vertex_map = img_feature_map.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::ObservationVertexCamera(), MemoryContext::CpuMemory);
	auto normal_map = img_feature_map.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::ObservationNormalCamera(), MemoryContext::CpuMemory);
	auto subsampled_vertex = cloud_feature_map.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::ObservationVertexCamera(), MemoryContext::CpuMemory);
	auto subsampled_normal = cloud_feature_map.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::ObservationNormalCamera(), MemoryContext::CpuMemory);
	
	
	//Check it
	auto gather_index = cloud_feature_map.GetTypedFeatureValueReadOnly<unsigned>(CommonFeatureChannelKey::GatherIndex(), MemoryContext::CpuMemory);
	LOG_ASSERT(gather_index.Size() == subsampled_normal.Size());
	for(auto i = 0; i < gather_index.Size(); i++) {
		auto from_normal = normal_map[gather_index[i]];
		auto to_normal = subsampled_normal[i];
		auto diff = from_normal - to_normal;
		//LOG_ASSERT(norm(diff) < 1e-7f);
	}
	
	//DebugVisualizer::DrawDepthImage(depth_map);
	//DebugVisualizer::DrawRGBImage(rgba_map);
	//DebugVisualizer::DrawPointCloud(vertex_map);
	//DebugVisualizer::DrawPointCloudWithNormal(vertex_map, normal_map);
	
	LOG(INFO) << "The size after subsampling is " << subsampled_vertex.FlattenSize();
	//DebugVisualizer::DrawPointCloud(subsampled_vertex);
	DebugVisualizer::DrawPointCloudWithNormal(subsampled_vertex, subsampled_normal);
	
	//Save it
	/*json j_img = img_feature_map;
	json j_cloud = cloud_feature_map;
	std::ofstream output_img("test_image.json");
	output_img << j_img;
	output_img.close();
	
	std::ofstream output_cloud("test_cloud.json");
	output_cloud << j_cloud;
	output_cloud.close();
	
	//Insert the normal type
	FeatureMap test_model;
	test_model.InsertDenseFeature<float4>(CommonFeatureChannelKey::ReferenceVertex(), subsampled_vertex);
	test_model.InsertDenseFeature<float4>(CommonFeatureChannelKey::ReferenceNormal(), subsampled_normal);
	std::ofstream output_model("test_model.json");
	json j_model = test_model;
	output_model << j_model;
	output_model.close();*/
}

void test_load() {
	using namespace poser;
	std::ifstream input_cloud("test_cloud.json");
	json j_cloud;
	input_cloud >> j_cloud;
	input_cloud.close();
	
	auto cloud_feature_map = j_cloud.get<FeatureMap>();
	auto subsampled_vertex = cloud_feature_map.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::ObservationVertexCamera(), MemoryContext::CpuMemory);
	auto subsampled_normal = cloud_feature_map.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::ObservationNormalCamera(), MemoryContext::CpuMemory);
	LOG_ASSERT(subsampled_vertex.FlattenSize() == cloud_feature_map.GetDenseFeatureDim().total_size());
	
	LOG(INFO) << "The size after subsampling is " << subsampled_vertex.FlattenSize();
	//DebugVisualizer::DrawPointCloud(subsampled_vertex);
	//DebugVisualizer::DrawPointCloudWithNormal(subsampled_vertex, subsampled_normal);
	
	{
		//Load the model
		std::ifstream input_model("test_model.json");
		json j_model;
		input_model >> j_model;
		input_model.close();
		
		auto model_feature_map = j_model.get<FeatureMap>();
		auto ref_vertex = model_feature_map.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::ReferenceVertex(), MemoryContext::CpuMemory);
		auto ref_normal = model_feature_map.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::ReferenceNormal(), MemoryContext::CpuMemory);
		LOG_ASSERT(ref_vertex.FlattenSize() == model_feature_map.GetDenseFeatureDim().total_size());
		DebugVisualizer::DrawPointCloudWithNormal(ref_vertex, ref_normal);
		
		//Check the valid size
		auto ref_vertex_blob = model_feature_map.GetFeatureValueReadOnly(CommonFeatureChannelKey::ReferenceVertex(), MemoryContext::CpuMemory);
		LOG_ASSERT(ref_vertex_blob.ValidTypeByte() == sizeof(float3));
	}
}

void test_rigid_transform() {
	using namespace poser;
	std::ifstream input_cloud("test_cloud.json");
	json j_cloud;
	input_cloud >> j_cloud;
	input_cloud.close();
	
	//Get the vertex in
	auto cloud_feature_map = j_cloud.get<FeatureMap>();
	auto subsampled_vertex_camera = cloud_feature_map.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::ObservationVertexCamera(), MemoryContext::CpuMemory);
	
	CloudRigidTransformer rigid_transform(MemoryContext::CpuMemory);
	rigid_transform.SetTransformNormalFromChannel(FeatureChannelType());
	rigid_transform.CheckAndAllocate(cloud_feature_map);
	
	
	//Do it
	mat34 camera2world = mat34::identity();
	camera2world.translation = make_float3(0.05, 0, 0);
	rigid_transform.SetRigidTransform(camera2world);
	rigid_transform.Process(cloud_feature_map);
	auto vertex_world = cloud_feature_map.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::ObservationVertexWorld(), MemoryContext::CpuMemory);
	DebugVisualizer::DrawMatchedCloudPair(subsampled_vertex_camera, vertex_world);
}


int main() {
	test_cpu_process();
	//test_load();
	//test_rigid_transform();
}