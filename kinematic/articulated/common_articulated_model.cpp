//
// Created by wei on 10/15/18.
//

#include "kinematic/articulated/common_articulated_model.h"

#include <drake/common/find_resource.h>
#include <drake/multibody/parsers/urdf_parser.h>
#include <drake/multibody/parsers/sdf_parser.h>

#include <fstream>

void poser::setupKukaIiwaTree(RigidBodyTree<double> &tree) {
	using namespace drake;
	std::string iiwa_urdf_path = "/home/wei/Documents/programs/poser/data/articulated/iiwa_model/iiwa_description/urdf/iiwa14_tracking.urdf";
	parsers::urdf::AddModelInstanceFromUrdfFileToWorld(
		iiwa_urdf_path, multibody::joints::kRollPitchYaw, &tree);
}

void poser::setupKukaIiwaWithGripperTree(RigidBodyTree<double> &tree) {
	using namespace drake;
	std::string iiwa_urdf_path = "/home/wei/Documents/programs/poser/data/articulated/iiwa_model/iiwa_description/urdf/iiwa14_tracking.urdf";
	parsers::urdf::AddModelInstanceFromUrdfFileToWorld(
		iiwa_urdf_path, multibody::joints::kRollPitchYaw, &tree);
	
	//Add the shunk
	std::string shunk_sdf_path = "/home/wei/Documents/programs/poser/data/articulated/iiwa_model/wsg_50_description/sdf/schunk_wsg_50.sdf";
	auto gripper_frame = tree.findFrame("iiwa_frame_ee");
	parsers::sdf::AddModelInstancesFromSdfFile(shunk_sdf_path, multibody::joints::kFixed, gripper_frame, &tree);
}

void poser::setupKukaIiwaModel(
	poser::FeatureMap &geometric_model,
	poser::ArticulatedKinematicModel &kinematic
) {
	//The json model
	using namespace poser;
	json j_model;
	std::ifstream model_in("/home/wei/Documents/programs/poser/tmp/iiwa_model.json");
	model_in >> j_model;
	model_in.close();
	
	//Load the model
	geometric_model = j_model.get<FeatureMap>();
	kinematic.CheckGeometricModelAndAllocateAttribute(geometric_model);
}

void poser::setupKukaIiwaWithGripperModel(
	poser::FeatureMap &geometric_model,
	poser::ArticulatedKinematicModel &kinematic
) {
	//The json model
	using namespace poser;
	json j_model;
	std::ifstream model_in("/home/wei/Documents/programs/poser/tmp/iiwa_with_shunk.json");
	model_in >> j_model;
	model_in.close();
	
	//Load the model
	geometric_model = j_model.get<FeatureMap>();
	kinematic.CheckGeometricModelAndAllocateAttribute(geometric_model);
}