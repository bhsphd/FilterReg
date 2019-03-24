//
// Created by wei on 9/11/18.
//

#pragma once

#include <vector>
#include <string>
#include <unordered_map>

//nvcc is not happy with json
#ifndef __CUDACC__
#include <nlohmann/json.hpp>
#else
namespace nlohmann {
	//Placeholder
	class json;
}
#endif

//Do not use eigen on cuda
#ifndef EIGEN_NO_CUDA
#define EIGEN_NO_CUDA
#endif
#include <Eigen/Eigen>

namespace poser {
	//These are in poser namespace
	using std::vector;
	using std::string;
	using std::unordered_map;
	using std::unordered_multimap;
	using std::pair;
	using nlohmann::json;
	
	//A very commonly used method
	static inline int div_up(int total, int grain) { return (total + grain - 1) / grain; }
}


//Some eigen types
namespace Eigen {
	using Matrix6f = Matrix<float, 6, 6>;
	using Vector6f = Matrix<float, 6, 1>;
	
	//The twist type, the rotation is in (0, 1, 2), while translation is in (3, 4, 5)
	using Twist6f = Matrix<float, 6, 1>;
}