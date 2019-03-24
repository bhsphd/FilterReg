//
// Created by wei on 9/16/18.
//

#pragma once

#include "common/feature_map.h"

#include <vector_functions.h>

namespace poser {
	
	
	struct SubsamplerCommonOption {
		//The bounding box filtering
		float3 bounding_box_min;
		float3 bounding_box_max;
		
		float& x_max() { return bounding_box_max.x; } const float& x_max() const { return bounding_box_max.x; }
		float& x_min() { return bounding_box_min.x; } const float& x_min() const { return bounding_box_min.x; }
		float& y_max() { return bounding_box_max.y; } const float& y_max() const { return bounding_box_max.y; }
		float& y_min() { return bounding_box_min.y; } const float& y_min() const { return bounding_box_min.y; }
		float& z_max() { return bounding_box_max.z; } const float& z_max() const { return bounding_box_max.z; }
		float& z_min() { return bounding_box_min.z; } const float& z_min() const { return bounding_box_min.z; }
		
		//Use segmentation mask to filter the vertex, 0 is background and everything else is foreground
		FeatureChannelType foreground_mask; //An unsigned int map in the same size as input, can be segmentation mask
		
		//The default constructor
		SubsamplerCommonOption()
		: bounding_box_min({-10, -10, -10}),
		  bounding_box_max({ 10,  10,  10}) {}
	};
}
