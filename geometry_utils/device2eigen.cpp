//
// Created by wei on 9/19/18.
//

#include "geometry_utils/device2eigen.h"

Eigen::Matrix3f poser::to_eigen(const mat33 &rhs) {
	Eigen::Matrix3f lhs;
	lhs(0, 0) = rhs.m00();
	lhs(0, 1) = rhs.m01();
	lhs(0, 2) = rhs.m02();
	lhs(1, 0) = rhs.m10();
	lhs(1, 1) = rhs.m11();
	lhs(1, 2) = rhs.m12();
	lhs(2, 0) = rhs.m20();
	lhs(2, 1) = rhs.m21();
	lhs(2, 2) = rhs.m22();
	return lhs;
}

Eigen::Matrix4f poser::to_eigen(const mat34 &rhs) {
	Eigen::Matrix4f lhs;
	lhs.setIdentity();
	//The rotational part
	lhs(0, 0) = rhs.linear.m00();
	lhs(0, 1) = rhs.linear.m01();
	lhs(0, 2) = rhs.linear.m02();
	lhs(1, 0) = rhs.linear.m10();
	lhs(1, 1) = rhs.linear.m11();
	lhs(1, 2) = rhs.linear.m12();
	lhs(2, 0) = rhs.linear.m20();
	lhs(2, 1) = rhs.linear.m21();
	lhs(2, 2) = rhs.linear.m22();
	//The translation part
	lhs.block<3, 1>(0, 3) = to_eigen(rhs.translation);
	return lhs;
}


Eigen::Vector3f poser::to_eigen(const float3 &rhs) {
	Eigen::Vector3f lhs;
	lhs(0) = rhs.x;
	lhs(1) = rhs.y;
	lhs(2) = rhs.z;
	return lhs;
}

Eigen::Vector4f poser::to_eigen(const float4 &rhs) {
	Eigen::Vector4f lhs;
	lhs(0) = rhs.x;
	lhs(1) = rhs.y;
	lhs(2) = rhs.z;
	lhs(3) = rhs.w;
	return lhs;
}

float3 poser::from_eigen(const Eigen::Vector3f &rhs) {
	float3 lhs;
	lhs.x = rhs(0);
	lhs.y = rhs(1);
	lhs.z = rhs(2);
	return lhs;
}

float4 poser::from_eigen(const Eigen::Vector4f &rhs) {
	float4 lhs;
	lhs.x = rhs(0);
	lhs.y = rhs(1);
	lhs.z = rhs(2);
	lhs.w = rhs(3);
	return lhs;
}