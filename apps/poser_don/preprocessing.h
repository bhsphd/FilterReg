//
// Created by wei on 12/9/18.
//

#pragma once
#include <opencv2/opencv.hpp>

#include "imgproc/imgproc.h"
#include "cloudproc/cloudproc.h"
#include "visualizer/debug_visualizer.h"

namespace poser {
	
	//The loader part
	void load_geometric_template(poser::FeatureMap& feature_map, const std::string& template_path);
	void load_depth_image(poser::FeatureMap& feature_map, const std::string& depth_path);
	void load_segment_mask(poser::FeatureMap& feature_map, const std::string& mask_path);
	poser::FeatureChannelType load_descriptor_image(poser::FeatureMap& feature_map, const std::string& npy_path);
	
	
	//The processing part
	void process_image(
		poser::FeatureMap& image_map,
		const poser::mat34& camera2world = poser::mat34::identity());
	void process_cloud(
		const poser::FeatureMap& image_map,
		const poser::FeatureChannelType& descriptor_channel,
		poser::FeatureMap& cloud_map
	);
}

