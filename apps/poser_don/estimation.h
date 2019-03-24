//
// Created by wei on 12/9/18.
//

#pragma once

#include "ransac/ransac_common.h"
#include "kinematic/rigid/rigid.h"
#include "kinematic/affine/affine.h"

#include "parse_request.h"

namespace poser {
	
	//The estimation interface using yaml
	std::pair<poser::mat34, poser::mat34> perform_estimation(const PoserRequestYaml& request);
}
