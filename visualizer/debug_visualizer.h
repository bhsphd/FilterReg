//
// Created by wei on 9/12/18.
//

#pragma once

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <memory>
#include <pcl/visualization/common/common.h>

#include "common/common_type.h"
#include "common/tensor_access.h"
#include "common/blob_access.h"

namespace poser {
	
	/* This class implements visualization for DEBUG purpose.
	 * The method is potentially very inefficient and should not
	 * be used in online visualization.
	 */
	class DebugVisualizer {
		/// The 3d xyz point cloud drawing method
	public:
		static void DrawPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr & point_cloud);
		static void DrawPointCloud(const TensorView<float4>& point_cloud, float scale = 1.0f);
		static void CPUTensor2PCLPointCloud(
			const TensorView<float4>& tensor,
			pcl::PointCloud<pcl::PointXYZ>& cloud,
			float scale = 1.0f);
	private:
		static void drawPointCloudCPU(const TensorView<float4>& point_cloud, float scale = 1.0f);
		static void drawPointCloudGPU(const TensorView<float4>& point_cloud, float scale = 1.0f);
		
		/// The normal visualizer, is usually combined with point cloud
	public:
		static void DrawPointCloudWithNormal(const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud,
		                                     const pcl::PointCloud<pcl::Normal>::Ptr& normal_cloud);
		static void DrawPointCloudWithNormal(const TensorView<float4>& vertex, const TensorView<float4>& normal, float vertex_scale = 1000.f);
	private:
		static void drawPointCloudWithNormalCPU(const TensorView<float4>& vertex, const TensorView<float4>& normal, float vertex_scale = 1000.f);
		
		
		/// The rgb image visualizer, note that the default format of
		/// opencv is BGR instead of RGB
	public:
		static void DrawRGBImage(const cv::Mat& cv_bgr_img);
		static void DrawRGBImage(const TensorView<uchar4>& rgb_img);
	private:
		static void drawRGBImageCPU(const TensorView<uchar4>& rgb_img);
		static void drawRGBImageGPU(const TensorView<uchar4>& rgb_img);
		
		
		/// The depth image visualizer
	public:
		static void DrawDepthImage(const cv::Mat& depth_img);
		static void DrawDepthImage(const TensorView<unsigned short>& depth_img);
		static void DrawForegroundMask(const TensorView<unsigned char>& mask);
	private:
		static void drawDepthImageCPU(const TensorView<unsigned short>& depth_img);
		static void drawDepthImageGPU(const TensorView<unsigned short>& depth_img);
		
		
		/// The visualizer for matched point cloud pair
	public:
		static void DrawMatchedCloudPair(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud_1,
		                                 const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud_2);
		static void DrawMatchedCloudPair(const TensorView<float4>& cloud_1,
		                                 const TensorView<float4>& cloud_2);
	private:
		static void drawMatchedCloudPairCPU(const TensorView<float4>& cloud_1,
		                                    const TensorView<float4>& cloud_2);
		
		/// The visualizer for colored point cloud
	public:
		static void DrawColoredPointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud);
		static void DrawColoredPointCloud(const poser::TensorView<float4> &geometric, const poser::BlobView &color);
		
		/// The visualizer for matched cloud pair with color
	public:
		static void DrawMatchedColorCloudPair(
			const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& observation,
			const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& model);
		static void GetColoredPointCloud(
			const poser::TensorView<float4> &geometric,
			const poser::BlobView &color,
			pcl::PointCloud<pcl::PointXYZRGB> &colored_cloud);
		static void DrawMatchedColorCloudPair(
			const TensorView<float4>& obs_geometric, const BlobView& obs_color,
			const TensorView<float4>& model_geometric, const BlobView& model_color);
		
		/// The visualizer for visible point cloud
	public:
		static void DrawVisiblePointCloud(
			const TensorView<float4>& cloud, 
			const TensorView<float>& visibility_score, 
			float invisible_threshold = 0.1f);
	private:
	    static void drawVisiblePointCloudCPU(
			const TensorView<float4>& cloud, 
			const TensorView<float>& visibility_score, 
			float invisible_threshold = 0.1f
		);
	};
	
}