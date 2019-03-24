//
// Created by wei on 1/14/19.
//

#pragma once

#include "common/feature_map.h"
#include "geometry_utils/device_mat.h"

namespace poser {
	//Generate uniform samples on the SO3 space
	//The translation of the transformation is set to identity
	//The sampled_transform should have n_samples elements
	void randomUniformSampleSO3Space(mat34* sampled_transform, unsigned n_samples);
}
