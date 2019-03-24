//
// Created by wei on 9/19/18.
//

#include "geometry_utils/device_mat.h"
#include "geometry_utils/device2eigen.h"

poser::mat33::mat33(const Eigen::Matrix3f &rhs) {
	cols[0] = make_float3(rhs(0, 0), rhs(1, 0), rhs(2, 0));
	cols[1] = make_float3(rhs(0, 1), rhs(1, 1), rhs(2, 1));
	cols[2] = make_float3(rhs(0, 2), rhs(1, 2), rhs(2, 2));
}

poser::mat33& poser::mat33::operator=(const Eigen::Matrix3f &rhs) {
	cols[0] = make_float3(rhs(0, 0), rhs(1, 0), rhs(2, 0));
	cols[1] = make_float3(rhs(0, 1), rhs(1, 1), rhs(2, 1));
	cols[2] = make_float3(rhs(0, 2), rhs(1, 2), rhs(2, 2));
	return *this;
}

poser::mat34::mat34(const Eigen::Isometry3f &se3) : linear(se3.linear().matrix()) {
	Eigen::Vector3f eigen_translation = se3.translation();
	translation = from_eigen(eigen_translation);
}

poser::mat34::mat34(const Eigen::Matrix4f &rhs) : linear(rhs.block<3, 3>(0, 0)) {
	Eigen::Vector3f eigen_trans = rhs.block<3, 1>(0, 3);
	translation = from_eigen(eigen_trans);
}

poser::mat34::mat34(const float3 &twist_rot, const float3 &twist_trans) {
	if (fabsf_sum(twist_rot) < 1e-4f) {
		linear.set_identity();
	}
	else {
		float angle = ::poser::norm(twist_rot);
		float3 axis = (1.0f / angle) * twist_rot;
		
		float c = cosf(angle);
		float s = sinf(angle);
		float t = 1.0f - c;
		
		linear.m00() = t*axis.x*axis.x + c;
		linear.m01() = t*axis.x*axis.y - axis.z*s;
		linear.m02() = t*axis.x*axis.z + axis.y*s;
		
		linear.m10() = t*axis.x*axis.y + axis.z*s;
		linear.m11() = t*axis.y*axis.y + c;
		linear.m12() = t*axis.y*axis.z - axis.x*s;
		
		linear.m20() = t*axis.x*axis.z - axis.y*s;
		linear.m21() = t*axis.y*axis.z + axis.x*s;
		linear.m22() = t*axis.z*axis.z + c;
	}
	
	//The translation part
	translation = twist_trans;
}