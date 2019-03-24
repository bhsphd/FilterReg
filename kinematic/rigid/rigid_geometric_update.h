//
// Created by wei on 9/18/18.
//

#pragma once

#include "kinematic/rigid/rigid_kinematic_model.h"

namespace poser {
	
	//The update method on cpu
	void UpdateLiveVertexCPU(const RigidKinematicModel& kinematic, FeatureMap& geometric_model);
	void UpdateLiveVertexAndNormalCPU(const RigidKinematicModel& kinematic, FeatureMap& geometric_model);
	
	//The update method on gpu
	//void UpdateLiveVertexGPU(const RigidKinematicModel& kinematic, FeatureMap& geometric_model, cudaStream_t stream = 0);
	//void UpdateLiveVertexAndNormalGPU(const RigidKinematicModel& kinematic, FeatureMap& geometric_model, cudaStream_t stream = 0);
}
