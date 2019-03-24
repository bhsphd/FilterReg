//
// Created by wei on 9/14/18.
//

#include "imgproc/imgproc.h"
#include "visualizer/debug_visualizer.h"

#include <chrono>
#include <cuda_runtime_api.h>

void test_cpu_process() {
	using namespace poser;
	using namespace std::chrono;
	
	FeatureMap feature_map;
	
	//Construct the processor
	std::string data_prefix("/home/wei/Documents/programs/surfelwarp/data/boxing");
	FrameLoaderFile loader(data_prefix);
	DepthBilateralFilter depth_filter;
	DepthVertexMapComputer vertex_map_processor;
	DepthNormalMapComputer normal_map_processor;
	
	//Allocate the buffer
	loader.CheckAndAllocate(feature_map);
	depth_filter.CheckAndAllocate(feature_map);
	vertex_map_processor.CheckAndAllocate(feature_map);
	normal_map_processor.CheckAndAllocate(feature_map);
	
	//Do processing
	loader.UpdateFrameIndex(1);
	loader.LoadDepthImage(feature_map);
	depth_filter.Process(feature_map);
	
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	vertex_map_processor.Process(feature_map);
	normal_map_processor.Process(feature_map);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	auto delta_t_ms = duration_cast<milliseconds>(t2 - t1);
	LOG(INFO) << "The time is ms is " << delta_t_ms.count();
	
	
	//Do visualize
	auto depth_map = feature_map.GetTypedFeatureValueReadOnly<unsigned short>(CommonFeatureChannelKey::FilteredDepthImage(), MemoryContext::CpuMemory);
	auto rgba_map = feature_map.GetTypedFeatureValueReadOnly<uchar4>(CommonFeatureChannelKey::RawRGBImage(), MemoryContext::CpuMemory);
	auto vertex_map = feature_map.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::ObservationVertexCamera(), MemoryContext::CpuMemory);
	auto normal_map = feature_map.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::ObservationNormalCamera(), MemoryContext::CpuMemory);
	//DebugVisualizer::DrawDepthImage(depth_map);
	//DebugVisualizer::DrawRGBImage(rgba_map);
	//DebugVisualizer::DrawPointCloud(vertex_map);
	DebugVisualizer::DrawPointCloudWithNormal(vertex_map, normal_map);
}

void test_gpu_process() {
	using namespace poser;
	using namespace std::chrono;
	
	FeatureMap feature_map;
	
	//Construct the processor
	std::string data_prefix("/home/wei/Documents/programs/surfelwarp/data/boxing");
	FrameLoaderFile loader(data_prefix, false);
	DepthBilateralFilter depth_filter(MemoryContext::GpuMemory);
	DepthVertexMapComputer vertex_map_processor(MemoryContext::GpuMemory);
	DepthNormalMapComputer normal_map_processor(MemoryContext::GpuMemory);
	
	//Allocate the buffer
	loader.CheckAndAllocate(feature_map);
	depth_filter.CheckAndAllocate(feature_map);
	vertex_map_processor.CheckAndAllocate(feature_map);
	normal_map_processor.CheckAndAllocate(feature_map);
	
	//Do processing
	loader.UpdateFrameIndex(1);
	loader.LoadDepthImage(feature_map);
	
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	depth_filter.Process(feature_map);
	vertex_map_processor.Process(feature_map);
	normal_map_processor.Process(feature_map);
	cudaSafeCall(cudaDeviceSynchronize());
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	auto delta_t_ms = duration_cast<milliseconds>(t2 - t1);
	LOG(INFO) << "The time is ms is " << delta_t_ms.count();
	
	//Do visualize
	auto depth_map = feature_map.GetTypedFeatureValueReadOnly<unsigned short>(CommonFeatureChannelKey::FilteredDepthImage(), MemoryContext::GpuMemory);
	auto vertex_map = feature_map.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::ObservationVertexCamera(), MemoryContext::GpuMemory);
	auto normal_map = feature_map.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::ObservationNormalCamera(), MemoryContext::GpuMemory);
	//DebugVisualizer::DrawDepthImage(depth_map);
	//DebugVisualizer::DrawPointCloud(vertex_map);
	DebugVisualizer::DrawPointCloudWithNormal(vertex_map, normal_map);
}

int main() {
	test_gpu_process();
	//test_cpu_process();
}