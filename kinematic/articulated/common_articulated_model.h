//
// Created by wei on 10/15/18.
//

#pragma once
#include "common/feature_map.h"
#include "kinematic/articulated/articulated_kinematic_model.h"

namespace poser {
	
	//The kuka iiwa arm, with/without gripper
	void setupKukaIiwaTree(RigidBodyTree<double>& tree);
	void setupKukaIiwaWithGripperTree(RigidBodyTree<double>& tree);
	void setupKukaIiwaModel(
		FeatureMap& geometric_model,
		ArticulatedKinematicModel& kinematic);
	void setupKukaIiwaWithGripperModel(
		FeatureMap& geometric_model,
		ArticulatedKinematicModel& kinematic);
}
