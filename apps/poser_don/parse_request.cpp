//
// Created by wei on 12/9/18.
//

#include "common/feature_map.h"
#include "geometry_utils/device2eigen.h"
#include "parse_request.h"

//The yaml header
#include <yaml-cpp/yaml.h>
#include <fstream>

//The Eigen header for geometry
#include <Eigen/Geometry>

void parse_request_instance(const YAML::Node& config, poser::PoserRequestYaml& request) {
	//The template
	request.template_path = config["template"].as<std::string>();
	
	//The later are on images
	const auto& image_node = config["image_1"];
	request.descriptor_npy_path = image_node["descriptor_img"].as<std::string>();
	request.rgb_img_path = image_node["rgb_img"].as<std::string>();
	request.depth_img_path = image_node["depth_img"].as<std::string>();
	
	//The mask is optional
	if(image_node["mask_img"])
		request.foreground_mask_path = image_node["mask_img"].as<std::string>();
	else
		request.foreground_mask_path = std::string();
	
	//The visualization is optional
	if(image_node["visualize"])
		request.visualize = image_node["visualize"].as<int>() != 0;
	else 
		request.visualize = false;
	
	//Save the processed depth cloud or not, and optional flag
	if(image_node["save_processed_cloud"])
		request.save_world_observation_cloud_path = image_node["save_processed_cloud"].as<std::string>();
	else
		request.save_world_observation_cloud_path = std::string();
	
	//Save the geometric template or not, and optional flag
	if(image_node["save_template"])
		request.save_template_path = image_node["save_template"].as<std::string>();
	else
		request.save_template_path = std::string();
	
	//The transformation to world
	const auto& transform_node = image_node["camera_to_world"];
	const auto& quaternion_node = transform_node["quaternion"];
	const auto& translation_node = transform_node["translation"];
	
	//Get the data
	Eigen::Quaterniond quaternion;
	quaternion.x() = quaternion_node["x"].as<double>();
	quaternion.y() = quaternion_node["y"].as<double>();
	quaternion.z() = quaternion_node["z"].as<double>();
	quaternion.w() = quaternion_node["w"].as<double>();
	Eigen::Vector3d translation;
	translation(0) = translation_node["x"].as<double>();
	translation(1) = translation_node["y"].as<double>();
	translation(2) = translation_node["z"].as<double>();
	
	//Do it
	Eigen::Isometry3f transform_eigen;
	transform_eigen.linear() = quaternion.normalized().toRotationMatrix().cast<float>();
	transform_eigen.translation() = translation.cast<float>();
	request.camera2world = poser::mat34(transform_eigen);
}

void poser::parse_request(const std::string &yaml_path, vector<PoserRequestYaml> &request) {
	YAML::Node config = YAML::LoadFile(yaml_path);
	LOG_ASSERT(config.IsMap());
	request.clear();
	
	//Load the template
	for(auto it = config.begin(); it != config.end(); it++) {
		const auto& key = it->first;
		const auto& value = it->second;
		
		//The name
		PoserRequestYaml request_instance;
		request_instance.request_name = key.as<std::string>();
		
		//Parse the other parts
		parse_request_instance(value, request_instance);
		request.emplace_back(request_instance);
	}
}

void poser::write_response(
	const std::string& yaml_path_in,
	const std::string& yaml_path_out,
	const std::vector<std::pair<poser::mat34, poser::mat34>> &estimated_pose
) {
	YAML::Node config = YAML::LoadFile(yaml_path_in);
	LOG_ASSERT(config.IsMap());
	
	//Load it
	int counter = 0;
	for(auto it = config.begin(); it != config.end(); it++) {
		auto& key = it->first;
		auto& value = it->second;
		
		//Get the pose
		const auto& pose_it = estimated_pose[counter];
		counter++;
		
		//The rigid part
		Eigen::Matrix4f rigid_pose_mat = to_eigen(pose_it.first);
		vector<float> data_colmajor; data_colmajor.resize(16);
		memcpy(data_colmajor.data(), rigid_pose_mat.data(), sizeof(float) * 16);
		value["rigid_transform"] = data_colmajor;
		
		//Add the pose to value
		Eigen::Matrix4f eigen_pose_it = to_eigen(pose_it.second);
		memcpy(data_colmajor.data(), eigen_pose_it.data(), sizeof(float) * 16);
		value["affine_transform"] = data_colmajor;
	}
	
	//Write it to output
	std::ofstream output_file(yaml_path_out);
	output_file << config;
	output_file.close();
}