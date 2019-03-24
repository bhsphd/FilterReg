//
// Created by wei on 9/12/18.
//

#include "common/tensor_blob.h"
#include "common/data_transfer.h"
#include "visualizer/debug_visualizer.h"
#include "geometry_utils/vector_operations.hpp"

#include <pcl/visualization/pcl_visualizer.h>

/* The actual implementation of visualizer methods
 */
void poser::DebugVisualizer::DrawPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud) {
	pcl::visualization::PCLVisualizer viewer("simple point cloud viewer");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler(point_cloud, 255, 255, 255);
	viewer.addPointCloud(point_cloud, "point cloud");
	viewer.addCoordinateSystem(2.0, "point cloud", 0);
	viewer.setBackgroundColor(0.05, 0.05, 0.05, 1);
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "point cloud");
	while (!viewer.wasStopped()) {
		viewer.spinOnce();
	}
}


void poser::DebugVisualizer::DrawPointCloud(const poser::TensorView<float4> &point_cloud, float scale) {
	if(point_cloud.IsCpuTensor())
		drawPointCloudCPU(point_cloud, scale);
	else
		drawPointCloudGPU(point_cloud, scale);
}

void poser::DebugVisualizer::drawPointCloudCPU(const poser::TensorView<float4> &point_cloud, float scale) {
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
	CPUTensor2PCLPointCloud(point_cloud, *cloud, scale);
	DrawPointCloud(cloud);
}

void poser::DebugVisualizer::drawPointCloudGPU(const poser::TensorView<float4> &point_cloud, float scale) {
	LOG_ASSERT(point_cloud.IsGpuTensor());
	auto point_size = point_cloud.Size();
	TensorBlob cpu_blob;
	cpu_blob.Reset<float4>(point_size, MemoryContext::CpuMemory);
	auto cloud_cpu = cpu_blob.GetTypedTensorReadWrite<float4>();
	TensorCopyNoSync(point_cloud, cloud_cpu);
	drawPointCloudCPU(cloud_cpu, scale);
}

void poser::DebugVisualizer::CPUTensor2PCLPointCloud(
	const poser::TensorView<float4> &tensor,
	pcl::PointCloud<pcl::PointXYZ> &cloud,
	float scale
) {
	LOG_ASSERT(tensor.IsCpuTensor());
	cloud.clear();
	for(auto i = 0; i < tensor.FlattenSize(); i++) {
		auto point = tensor[i];
		pcl::PointXYZ pcl_point;
		pcl_point.x = point.x * scale;
		pcl_point.y = point.y * scale;
		pcl_point.z = point.z * scale;
		cloud.push_back(pcl_point);
	}
}

/* The method to draw point cloud with normal
 */
void poser::DebugVisualizer::DrawPointCloudWithNormal(
	const pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud,
	const pcl::PointCloud<pcl::Normal>::Ptr &normal_cloud
) {
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler(point_cloud, 0, 255, 0);
	viewer->addPointCloud<pcl::PointXYZ>(point_cloud, handler, "sample cloud");
	viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(point_cloud, normal_cloud, 30, 60.0f);
	while (!viewer->wasStopped()) {
		viewer->spinOnce(100);
	}
}

void poser::DebugVisualizer::DrawPointCloudWithNormal(
	const poser::TensorView<float4> &vertex,
	const poser::TensorView<float4> &normal,
	float vertex_scale
) {
	//Check the size
	LOG_ASSERT(vertex.FlattenSize() == normal.FlattenSize());
	LOG_ASSERT(vertex.GetMemoryContext() == normal.GetMemoryContext());
	
	//Dispatch on tensor type
	if(vertex.IsCpuTensor()) {
		drawPointCloudWithNormalCPU(vertex, normal, vertex_scale);
	} else {
		//Prepare the data
		TensorBlob vertex_cpu_blob; vertex_cpu_blob.Reset<float4>(vertex.DimensionalSize(), MemoryContext::CpuMemory);
		TensorBlob normal_cpu_blob; normal_cpu_blob.Reset<float4>(vertex.DimensionalSize(), MemoryContext::CpuMemory);
		auto vertex_cpu = vertex_cpu_blob.GetTypedTensorReadWrite<float4>();
		auto normal_cpu = normal_cpu_blob.GetTypedTensorReadWrite<float4>();
		
		//Copy it
		TensorCopyNoSync<float4>(vertex, vertex_cpu);
		TensorCopyNoSync<float4>(normal, normal_cpu);
		drawPointCloudWithNormalCPU(vertex_cpu, normal_cpu);
	}
}

void poser::DebugVisualizer::drawPointCloudWithNormalCPU(
	const poser::TensorView<float4> &vertex,
	const poser::TensorView<float4> &normal,
	float vertex_scale
) {
	pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::Normal>::Ptr normal_cloud(new pcl::PointCloud<pcl::Normal>);
	point_cloud->points.reserve(vertex.FlattenSize());
	normal_cloud->points.reserve(normal.FlattenSize());
	for(auto i = 0; i < vertex.FlattenSize(); i++) {
		pcl::PointXYZ pcl_point;
		const auto& point = vertex[i];
		//Scale point to mm
		pcl_point.x = point.x * vertex_scale;
		pcl_point.y = point.y * vertex_scale;
		pcl_point.z = point.z * vertex_scale;
		point_cloud->points.emplace_back(pcl_point);
		
		pcl::Normal pcl_normal;
		const auto& normal_in = normal[i];
		pcl_normal.normal_x = normal_in.x;
		pcl_normal.normal_y = normal_in.y;
		pcl_normal.normal_z = normal_in.z;
		normal_cloud->points.emplace_back(pcl_normal);
	}
	
	//Draw it
	DrawPointCloudWithNormal(point_cloud, normal_cloud);
}

//The method to draw the matched cloud pair
void poser::DebugVisualizer::DrawMatchedCloudPair(
	const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_1,
	const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_2
) {
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Matched Viewer"));
	viewer->setBackgroundColor(255, 255, 255);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_1(cloud_1, 255, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloud_1, handler_1, "cloud 1");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_2(cloud_2, 0, 0, 255);
	viewer->addPointCloud<pcl::PointXYZ>(cloud_2, handler_2, "cloud 2");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud 1");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud 2");
	
	//The position of the camera
	pcl::visualization::Camera camera;
	while (!viewer->wasStopped()) {
		viewer->spinOnce(100);
		viewer->getCameraParameters(camera);
	}
}


void poser::DebugVisualizer::DrawMatchedCloudPair(
	const poser::TensorView<float4> &cloud_1,
	const poser::TensorView<float4> &cloud_2
) {
	if(cloud_1.IsCpuTensor() && cloud_2.IsCpuTensor()) {
		drawMatchedCloudPairCPU(cloud_1, cloud_2);
	} else if(cloud_1.IsCpuTensor() && cloud_2.IsGpuTensor()) {
		TensorBlob cloud2_blob;
		cloud2_blob.Reset<float4>(cloud_2.DimensionalSize(), MemoryContext::CpuMemory);
		auto cpu_cloud2 = cloud2_blob.GetTypedTensorReadWrite<float4>();
		TensorCopyNoSync<float4>(cloud_2, cpu_cloud2);
		drawMatchedCloudPairCPU(cloud_1, cpu_cloud2);
	} else if(cloud_1.IsGpuTensor() && cloud_2.IsCpuTensor()) {
		DrawMatchedCloudPair(cloud_2, cloud_1);
	} else {
		//Download both cloud
		TensorBlob cloud2_blob;
		cloud2_blob.Reset<float4>(cloud_2.DimensionalSize(), MemoryContext::CpuMemory);
		auto cpu_cloud2 = cloud2_blob.GetTypedTensorReadWrite<float4>();
		TensorCopyNoSync<float4>(cloud_2, cpu_cloud2);
		drawMatchedCloudPairCPU(cloud_1, cpu_cloud2);
		
		TensorBlob cloud1_blob;
		cloud1_blob.Reset<float4>(cloud_1.DimensionalSize(), MemoryContext::CpuMemory);
		auto cpu_cloud1 = cloud2_blob.GetTypedTensorReadWrite<float4>();
		TensorCopyNoSync<float4>(cloud_1, cpu_cloud1);
		drawMatchedCloudPairCPU(cpu_cloud1, cpu_cloud2);
	}
}


void poser::DebugVisualizer::drawMatchedCloudPairCPU(
	const poser::TensorView<float4> &cloud_1,
	const poser::TensorView<float4> &cloud_2
) {
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud1(new pcl::PointCloud<pcl::PointXYZ>());
	CPUTensor2PCLPointCloud(cloud_1, *pcl_cloud1);
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud2(new pcl::PointCloud<pcl::PointXYZ>());
	CPUTensor2PCLPointCloud(cloud_2, *pcl_cloud2);
	DrawMatchedCloudPair(pcl_cloud1, pcl_cloud2);
}


//The method to draw visible point cloud
void poser::DebugVisualizer::DrawVisiblePointCloud(
	const poser::TensorView<float4> &cloud,
	const poser::TensorView<float> &visibility_score,
	float invisible_threshold
) {
	if(cloud.IsCpuTensor()) {
		LOG_ASSERT(visibility_score.IsCpuTensor());
		drawVisiblePointCloudCPU(cloud, visibility_score, invisible_threshold);
	} else {
		LOG_ASSERT(visibility_score.IsGpuTensor());
		LOG(FATAL) << "Not implemented yet";
	}
}


void poser::DebugVisualizer::drawVisiblePointCloudCPU(
	const TensorView<float4>& cloud, 
	const TensorView<float>& visibility_score, 
	float invisible_threshold
) {
	pcl::PointCloud<pcl::PointXYZ>::Ptr visible_cloud(new pcl::PointCloud<pcl::PointXYZ>());
	visible_cloud->points.reserve(cloud.Size());
	for (auto i = 0; i < cloud.Size(); i++) {
		const auto cloud_i = cloud[i];
		const auto visibility_score_i = visibility_score[i];
		if(visibility_score_i > invisible_threshold) {
			pcl::PointXYZ pcl_point;
			pcl_point.x = cloud_i.x;
			pcl_point.y = cloud_i.y;
			pcl_point.z = cloud_i.z;
			visible_cloud->push_back(pcl_point);
		}
	}

	LOG(INFO) << "The number of visible cloud is " << visible_cloud->points.size();
	DrawPointCloud(visible_cloud);
}


//The method to draw colored point cloud
void poser::DebugVisualizer::DrawColoredPointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud) {
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Color-Viewer"));
	viewer->setBackgroundColor(255, 255, 255);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> cloud_rgb(cloud);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, cloud_rgb, "cloud");
	
	//The size of the point
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud");
	while (!viewer->wasStopped()) {
		viewer->spinOnce();
	}
}

void poser::DebugVisualizer::DrawColoredPointCloud(
	const poser::TensorView<float4> &geometric,
	const poser::BlobView &color
) {
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
	GetColoredPointCloud(geometric, color, *cloud);
	DrawColoredPointCloud(cloud);
}

//The method to draw feature colored point cloud
void poser::DebugVisualizer::DrawMatchedColorCloudPair(
	const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &observation,
	const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &model
) {
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Match-Viewer"));
	viewer->setBackgroundColor(255, 255, 255);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> obs_rgb(observation);
	viewer->addPointCloud<pcl::PointXYZRGB>(observation, obs_rgb, "observation");
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> model_rgb(model);
	viewer->addPointCloud<pcl::PointXYZRGB>(model, model_rgb, "model");
	
	//The size of the point
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "observation");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "model");
	
	while (!viewer->wasStopped()) {
		viewer->spinOnce();
	}
}

void poser::DebugVisualizer::GetColoredPointCloud(
	const poser::TensorView<float4> &geometric,
	const poser::BlobView &color,
	pcl::PointCloud<pcl::PointXYZRGB> &colored_cloud
) {
	LOG_ASSERT(geometric.Size() == color.Size());
	LOG_ASSERT(color.ValidTypeByte() == sizeof(float) * 3);
	//Get the normalization for color
	float color_min[3] = { 1e5f,  1e5f,  1e5f};
	float color_max[3] = {-1e5f, -1e5f, -1e5f};
	for(auto i = 0; i < color.Size(); i++) {
		const auto color_i = color.ValidElemVectorAt<float>(i);
		for(auto j = 0; j < 3; j++) {
			if(color_i[j] < color_min[j])
				color_min[j] = color_i[j];
			if(color_i[j] > color_max[j])
				color_max[j] = color_i[j];
		}
	}
	
	//Get the normalized color
	auto get_normalized_color = [&](const BlobView::ElemVector<float>& color) -> uchar3 {
		uchar3 result = make_uchar3(0, 0, 0);
		unsigned char* result_ptr = (unsigned char*)(&result);
		for(auto j = 0; j < 3; j++) {
			auto float_value = 255.0f * (color[j] - color_min[j]) / (color_max[j] - color_min[j]);
			auto int_value = unsigned(float_value);
			result_ptr[j] = int_value;
		}
		return result;
	};
	
	//Resize the cloud
	colored_cloud.clear();
	for(auto i = 0; i < geometric.Size(); i++) {
		const auto& geometric_i = geometric[i];
		const auto& color_i = color.ValidElemVectorAt<float>(i);
		auto color_uchar3 = get_normalized_color(color_i);
		
		//Ignore zero
		if(is_zero_vertex(geometric_i))
			continue;
		
		pcl::PointXYZRGB point;
		point.x = geometric_i.x;
		point.y = geometric_i.y;
		point.z = geometric_i.z;
		point.r = color_uchar3.x;
		point.g = color_uchar3.y;
		point.b = color_uchar3.z;
		colored_cloud.push_back(point);
	}
}

void poser::DebugVisualizer::DrawMatchedColorCloudPair(
	const poser::TensorView<float4> &obs_geometric,
	const poser::BlobView &obs_color,
	const poser::TensorView<float4> &model_geometric,
	const poser::BlobView &model_color
) {
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr observation(new pcl::PointCloud<pcl::PointXYZRGB>());
	GetColoredPointCloud(obs_geometric, obs_color, *observation);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr model(new pcl::PointCloud<pcl::PointXYZRGB>());
	GetColoredPointCloud(model_geometric, model_color, *model);
	DrawMatchedColorCloudPair(observation, model);
}

