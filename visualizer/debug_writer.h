//
// Created by wei on 1/17/19.
//

#pragma once

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <memory>

#include "common/common_type.h"
#include "common/tensor_access.h"
#include "common/blob_access.h"

namespace poser {
	
	
	/* This class implements the saving method for DEBUG purpose.
	 * The implementation is very similar to DebugVisualizer, and
	 * the method in this class can be very inefficient.
	 */
	class DebugWriter {
	public:
		static void SavePointCloud(
			const pcl::PointCloud<pcl::PointXYZ>::Ptr & point_cloud,
			const std::string& save_pcd_path);
		static void SavePointCloud(
			const TensorView<float4>& point_cloud,
			const std::string& save_pcd_path,
			float scale = 1.0f);
	};
}
