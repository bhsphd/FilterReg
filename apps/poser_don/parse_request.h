//
// Created by wei on 12/9/18.
//

#pragma once
#include "common/common_type.h"
#include "geometry_utils/device_mat.h"

namespace poser {
	
	//The type for requrest
	struct PoserRequestYaml {
		//The name of requiest
		std::string request_name;
		
		//The path for images
		std::string template_path;
		std::string depth_img_path;
		std::string rgb_img_path;
		std::string foreground_mask_path;
		std::string descriptor_npy_path;
		
		//The transform from camera frame to world frame
		mat34 camera2world;
		
		//Do visualization or not
		bool visualize;
		
		//Save the processed depth cloud or not
		//If the path is empty, then don't save the cloud
		//Else save it, the path should end with pcd
		std::string save_world_observation_cloud_path;
		
		//Save the template as another format for further processing]
		std::string save_template_path;
	};
	
	//The parser for the request
	void parse_request(const std::string& yaml_path, vector<PoserRequestYaml>& request);
	void write_response(
		const std::string& yaml_path_in,
		const std::string& yaml_path_out,
		const std::vector<std::pair<poser::mat34, poser::mat34>>& estimated_pose);
}
