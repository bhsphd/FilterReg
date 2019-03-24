//
// Created by wei on 11/29/18.
//

#pragma once

#include "kinematic/affine/affine_kinematic_model.h"

namespace poser {
	
	//The update method on CPU
	void UpdateLiveVertexCPU(const AffineKinematicModel& kinematic, FeatureMap& geometric_model);
	void UpdateLiveVertexAndNormalCPU(const AffineKinematicModel& kinematic, FeatureMap& geometric_model);
}
