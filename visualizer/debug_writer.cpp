//
// Created by wei on 1/17/19.
//

#include "visualizer/debug_writer.h"
#include "visualizer/debug_visualizer.h"
#include <pcl/io/pcd_io.h>

void poser::DebugWriter::SavePointCloud(
	const pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud,
	const std::string &save_pcd_path
) {
	pcl::io::savePCDFile(save_pcd_path, *point_cloud);
}

void poser::DebugWriter::SavePointCloud(
	const poser::TensorView<float4> &point_cloud,
	const std::string &save_pcd_path,
	float scale
) {
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
	DebugVisualizer::CPUTensor2PCLPointCloud(point_cloud, *cloud, scale);
	SavePointCloud(cloud, save_pcd_path);
}