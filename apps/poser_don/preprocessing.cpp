//
// Created by wei on 12/9/18.
//

#include "preprocessing.h"

#include <pcl/io/pcd_io.h>

//The npy loader
#include "npy.hpp"

void poser::load_geometric_template(poser::FeatureMap &feature_map, const std::string &pcd_path) {
	//First load the point cloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
	if(pcl::io::loadPCDFile<pcl::PointXYZ> (pcd_path, *cloud) == -1) {
		PCL_ERROR ("Couldn't read vertex \n");
	}
	
	//Next load the normal cloud
	pcl::PointCloud<pcl::Normal>::Ptr normal(new pcl::PointCloud<pcl::Normal>());
	if(pcl::io::loadPCDFile<pcl::Normal>(pcd_path, *normal) == -1) {
		PCL_ERROR ("Couldn't read normal \n");
	}
	
	//Check the size
	LOG_ASSERT(normal->points.size() == cloud->points.size());
	
	//Convert to feature map
	using namespace poser;
	feature_map.AllocateDenseFeature(CommonFeatureChannelKey::ReferenceVertex(), cloud->points.size(), MemoryContext::CpuMemory);
	feature_map.AllocateDenseFeature(ImageFeatureChannelKey::DONDescriptor(3), MemoryContext::CpuMemory);
	auto vertex_feature = feature_map.GetTypedFeatureValueReadWrite<float4>(CommonFeatureChannelKey::ReferenceVertex(), MemoryContext::CpuMemory);
	auto don_feature = feature_map.GetTypedFeatureValueReadWrite<float3>(ImageFeatureChannelKey::DONDescriptor(3), MemoryContext::CpuMemory);
	
	//Get the feature
	for(auto i = 0; i < cloud->points.size(); i++) {
		auto& vertex_i = vertex_feature[i];
		const auto& pcl_point_i = cloud->points[i];
		vertex_i = make_float4(pcl_point_i.x, pcl_point_i.y, pcl_point_i.z, 1.0f);
		
		//The normal
		auto& don_feature_i = don_feature[i];
		const auto& pcl_normal_i = normal->points[i];
		don_feature_i = make_float3(pcl_normal_i.normal_x, pcl_normal_i.normal_y, pcl_normal_i.normal_z);
	}
}

void poser::load_depth_image(poser::FeatureMap& feature_map, const std::string& depth_path) {
	//Allocate the space
	using namespace poser;
	feature_map.AllocateDenseFeature<unsigned short>(
		CommonFeatureChannelKey::RawDepthImage(),
		TensorDim(480, 640),
		MemoryContext::CpuMemory);
	auto depth_map = feature_map.GetTypedFeatureValueReadWrite<unsigned short>(CommonFeatureChannelKey::RawDepthImage(), MemoryContext::CpuMemory);
	
	//Load it
	cv::Mat cv_depth_img = cv::imread(depth_path, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
	
	//Copy to the feature map
	cv::Mat wrapped(depth_map.Rows(), depth_map.Cols(), CV_16UC1, depth_map.RawPtr());
	cv_depth_img.copyTo(wrapped);
	
	//Visualize: OK
	//DebugVisualizer::DrawDepthImage(depth_map);
}


void poser::load_segment_mask(poser::FeatureMap& feature_map, const std::string& mask_path) {
	//Allocate the space
	using namespace poser;
	feature_map.AllocateDenseFeature<unsigned char>(
		CommonFeatureChannelKey::ForegroundMask(),
		TensorDim(480, 640),
		MemoryContext::CpuMemory);
	auto mask = feature_map.GetTypedFeatureValueReadWrite<unsigned char>(CommonFeatureChannelKey::ForegroundMask(), MemoryContext::CpuMemory);
	
	//Load it
	cv::Mat cv_mask = cv::imread(mask_path, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
	
	//Copy to the feature map
	cv::Mat wrapped(mask.Rows(), mask.Cols(), CV_8UC1, mask.RawPtr());
	cv_mask.copyTo(wrapped);
	
	//Visualize: OK
	//DebugVisualizer::DrawForegroundMask(mask);
}

poser::FeatureChannelType poser::load_descriptor_image(poser::FeatureMap& feature_map, const std::string& npy_path) {
	using namespace poser;
	vector<unsigned long> shape;
	vector<float> data;
	
	//Load it
	npy::LoadArrayFromNumpy(npy_path, shape, data);
	
	//Copy to feature map
	auto channel_type = ImageFeatureChannelKey::DONDescriptor(shape[2]);
	feature_map.AllocateDenseFeature(
		channel_type,
		TensorDim(480, 640),
		MemoryContext::CpuMemory);
	auto descriptor_map = feature_map.GetFeatureValueReadWrite(channel_type, MemoryContext::CpuMemory);
	
	//Need loop to copy it
	LOG_ASSERT(descriptor_map.ByteSize() == data.size() * sizeof(float));
	memcpy(descriptor_map.RawPtr(), data.data(), descriptor_map.ByteSize());
	
	//The channel type
	return channel_type;
}

void poser::process_image(
	poser::FeatureMap& image_map,
	const poser::mat34& camera2world
) {
	using namespace poser;
	DepthBilateralFilter depth_filter;
	DepthVertexMapComputer vertex_map_processor;
	//Transform
	CloudRigidTransformer transformer(MemoryContext::CpuMemory);
	transformer.SetRigidTransform(camera2world);
	transformer.SetTransformVertexFromChannel(CommonFeatureChannelKey::ObservationVertexCamera());
	transformer.SetTransformVertexToChannel(CommonFeatureChannelKey::ObservationVertexWorld());
	
	//Do it
	depth_filter.CheckAndAllocate(image_map);
	vertex_map_processor.CheckAndAllocate(image_map);
	transformer.CheckAndAllocate(image_map);
	
	depth_filter.Process(image_map);
	vertex_map_processor.Process(image_map);
	transformer.Process(image_map);
	
	//Visualize: OK
	//auto vertex_map = image_map.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::ObservationVertexCamera(), MemoryContext::CpuMemory);
	//auto descriptor_value = image_map.GetFeatureValueReadOnly(ImageFeatureChannelKey::DONDescriptor(3), MemoryContext::CpuMemory);
	//DebugVisualizer::DrawPointCloud(vertex_map, 1000);
	//DebugVisualizer::DrawColoredPointCloud(vertex_map, descriptor_value);
}

void poser::process_cloud(
	const poser::FeatureMap& image_map,
	const poser::FeatureChannelType& descriptor_channel,
	poser::FeatureMap& cloud_map
) {
	using namespace poser;
	//The subsampler
	VoxelGridSubsamplerCPU subsampler(CommonFeatureChannelKey::ObservationVertexWorld());
	subsampler.SetVoxelGridLeafSize(0.002f); // 2mm
	
	//Note that the mask is optional, and the bounding box parameter is only valid for Kuka-1
	SubsamplerCommonOption subsample_option;
	subsample_option.bounding_box_min = make_float3(0.66757267f - 0.25f, -0.35f, -0.0f);
	subsample_option.bounding_box_max = make_float3(0.66757267f + 0.25f, +0.35f, 0.18953078f + 0.2f);
	if(image_map.ExistFeature(CommonFeatureChannelKey::ForegroundMask(), MemoryContext::CpuMemory)) {
		subsample_option.foreground_mask = CommonFeatureChannelKey::ForegroundMask();
	}
	subsampler.SetSubsamplerOption(subsample_option);
	
	//The feature gather
	FeatureGatherProcessor feature_gather;
	feature_gather.InsertGatheredFeature(descriptor_channel);
	
	//Processing
	subsampler.CheckAndAllocate(image_map, cloud_map);
	feature_gather.CheckAndAllocate(image_map, cloud_map);
	
	subsampler.Process(image_map, cloud_map);
	feature_gather.PerformGatherCPU(image_map, cloud_map);
	
	//Visualize: OK
	//auto point_cloud = cloud_map.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::ObservationVertexWorld(), MemoryContext::CpuMemory);
	//auto descriptor_value = cloud_map.GetFeatureValueReadOnly(ImageFeatureChannelKey::DONDescriptor(3), MemoryContext::CpuMemory);
	//DebugVisualizer::DrawPointCloud(point_cloud, 1000);
	//DebugVisualizer::DrawColoredPointCloud(point_cloud, descriptor_value);
}