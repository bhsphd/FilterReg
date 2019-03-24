//
// Created by wei on 9/19/18.
//

#pragma once

#include "common/common_type.h"
#include "geometry_utils/vector_operations.hpp"
#include "geometry_utils/device_mat.h"

namespace poser {
	//Transfer the device vector/matrix to Eigen
	Eigen::Matrix3f to_eigen(const mat33 &rhs);
	Eigen::Matrix4f to_eigen(const mat34 &rhs);
	Eigen::Vector3f to_eigen(const float3 &rhs);
	Eigen::Vector4f to_eigen(const float4 &rhs);
	
	//For basic device vector
	float3 from_eigen(const Eigen::Vector3f &rhs);
	float4 from_eigen(const Eigen::Vector4f &rhs);
}
