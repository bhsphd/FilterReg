//
// Created by wei on 12/9/18.
//

#include <iostream>
#include <opencv2/opencv.hpp>

#include "imgproc/imgproc.h"
#include "cloudproc/cloudproc.h"
#include "visualizer/debug_visualizer.h"

//The subprocessor
#include "parse_request.h"
#include "preprocessing.h"
#include "estimation.h"
#include <chrono>

int main(int argc, char* argv[]) {
	using namespace poser;
	
	//Check the input
	if(argc < 2) {
		std::cout << "Usage: ./poser_don /path/to/request.yaml [/path/to/response.yaml]";
		exit(0);
	}
	
	//The request for poser
	std::string yaml_path(argv[1]);
	vector<PoserRequestYaml> request;
	parse_request(yaml_path, request);
	
	//Do estimation
	vector<std::pair<mat34, mat34>> estimated_transform; estimated_transform.resize(request.size());
	for(auto i = 0; i < request.size(); i++) {
		estimated_transform[i] = perform_estimation(request[i]);
	}
	
	//Write the result
	std::string response_path = "response.yaml";
	if(argc == 3)
		response_path = std::string(argv[2]);
	write_response(yaml_path, response_path, estimated_transform);
}