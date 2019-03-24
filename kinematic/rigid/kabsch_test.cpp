//
// Created by wei on 12/1/18.
//

#include <gtest/gtest.h>

#include "geometry_utils/vector_operations.hpp"
#include "geometry_utils/device2eigen.h"
#include "kinematic/rigid/rigid_point2point_kabsch.h"

class KabschTest : public ::testing::Test {
protected:
	void SetUp() override {
		//The init for SE3
		const auto angle = 0.87f;
		Eigen::Vector3f axis;
		axis.setRandom(); axis.normalize();
		Eigen::Isometry3f eigen_rand_init_SE3(Eigen::AngleAxisf(angle, axis));
		axis.setRandom(); axis.normalize();
		eigen_rand_init_SE3.translation() = 1.0f * axis;
		rand_init_SE3 = poser::mat34(eigen_rand_init_SE3);
	}
	
	poser::mat34 rand_init_SE3;
};

TEST_F(KabschTest, BasicTest) {
	//Generate some random point
	using namespace poser;
	float4 point_from[3];
	float4 point_to[3];
	
	Eigen::Vector4f rand_eigen;
	rand_eigen.setRandom(); point_from[0] = from_eigen(rand_eigen);
	rand_eigen.setRandom(); point_from[1] = from_eigen(rand_eigen);
	rand_eigen.setRandom(); point_from[2] = from_eigen(rand_eigen);
	
	//Transfer it
	for(auto i = 0; i < 3; i++) {
		auto& from_i = point_from[i];
		auto& to_i = point_to[i];
		float3& to_i_float3 = *(float3*)(&to_i);
		
		//Do transform and set weight to 1
		to_i_float3 = rand_init_SE3.rotation() * from_i + rand_init_SE3.translation;
		to_i.w = from_i.w = 1.0f;
	}
	
	//Compute the transform
	float4 centeralized_from[3];
	float4 centeralized_to[3];
	mat34 transform;
	RigidPoint2PointKabsch::ComputeTransformBetweenClouds(
		3,
		point_from, point_to,
		centeralized_from, centeralized_to,
		transform);
	
	//Check transform and actual value
	auto num_float = sizeof(transform) / sizeof(float);
	float* computed = (float*)(&transform);
	float* gt = (float*)(&rand_init_SE3);
	for(auto i = 0; i < num_float; i++) {
		EXPECT_NEAR(computed[i], gt[i], 1e-3f);
	}
}

TEST_F(KabschTest, RandomTest) {
	const auto test_samples = 1000;
	std::srand((unsigned)std::time(0));
	using namespace poser;
	
	//The test loop
	for(auto k = 0; k < test_samples; k++) {
		float4 point_from[3];
		float4 point_to[3];
		
		//Random point
		Eigen::Vector4f rand_eigen;
		rand_eigen.setRandom(); point_from[0] = from_eigen(rand_eigen);
		rand_eigen.setRandom(); point_from[1] = from_eigen(rand_eigen);
		rand_eigen.setRandom(); point_from[2] = from_eigen(rand_eigen);
		
		//Random transformation
		//The init for SE3
		const auto angle = 1.87f;
		Eigen::Vector3f axis;
		axis.setRandom(); axis.normalize();
		Eigen::Isometry3f eigen_rand_init_SE3(Eigen::AngleAxisf(angle, axis));
		axis.setRandom(); axis.normalize();
		eigen_rand_init_SE3.translation() = 1.0f * axis;
		mat34 rand_SE3 = poser::mat34(eigen_rand_init_SE3);
		
		//Transform the point
		for(auto i = 0; i < 3; i++) {
			auto& from_i = point_from[i];
			auto& to_i = point_to[i];
			float3& to_i_float3 = *(float3*)(&to_i);
			
			//Do transform and set weight to 1
			to_i_float3 = rand_SE3.rotation() * from_i + rand_SE3.translation;
			to_i.w = from_i.w = 1.0f;
		}
		
		//Compute the transform
		float4 centeralized_from[3];
		float4 centeralized_to[3];
		mat34 transform;
		RigidPoint2PointKabsch::ComputeTransformBetweenClouds(
			3,
			point_from, point_to,
			centeralized_from, centeralized_to,
			transform);
		
		//Check transform and actual value
		auto num_float = sizeof(transform) / sizeof(float);
		float* computed = (float*)(&transform);
		float* gt = (float*)(&rand_SE3);
		for(auto i = 0; i < num_float; i++) {
			EXPECT_NEAR(computed[i], gt[i], 1e-3f);
		}
	}
}