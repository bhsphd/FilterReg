//
// Created by wei on 1/14/19.
//

#include "ransac/so3_sample.h"
#include <chrono>
#include <random>
#include <math.h>
#include <Eigen/Geometry>

void poser::randomUniformSampleSO3Space(poser::mat34 *sampled_transform, unsigned n_samples) {
	long seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine r_engine(seed);
	std::uniform_real_distribution<float> distribution(0, 1.0f);
	
	//Do sampling
	for(auto i = 0; i < n_samples; i++) {
		//First generate quaternion
		float u_1 = distribution(r_engine);
		float u_2 = distribution(r_engine);
		float u_3 = distribution(r_engine);
		
		Eigen::Quaternionf quat;
		quat.x() = std::sqrt(1.0f - u_1) * std::sin(2.0f * M_PI * u_2);
		quat.y() = std::sqrt(1.0f - u_1) * std::cos(2.0f * M_PI * u_2);
		quat.z() = std::sqrt(u_1) * std::sin(2.0f * M_PI * u_3);
		quat.w() = std::sqrt(u_1) * std::cos(2.0f * M_PI * u_3);
		
		//To rotation matrix
		Eigen::Isometry3f transform; transform.setIdentity();
		transform.linear() = quat.toRotationMatrix();
		sampled_transform[i] = mat34(transform);
	}
}