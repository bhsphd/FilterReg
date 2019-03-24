//
// Created by wei on 9/18/18.
//

#pragma once

#include "kinematic/articulated/articulated_kinematic_model.h"

namespace poser {
	
	//The update method on cpu, assume after do kinematic
	void UpdateLiveVertexCPU(const ArticulatedKinematicModel& kinematic, FeatureMap& geometric_model);
}