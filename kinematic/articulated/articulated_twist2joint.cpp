//
// Created by wei on 9/30/18.
//

#include "kinematic/articulated/articulated_twist2joint.h"

#include <glog/logging.h>


void poser::ArticulatedJacobianTwist2Joint::AssembleJointSpaceJacobian(
	const poser::ArticulatedKinematicModel &kinematic,
	const float *body_jtj, const float *body_jte,
	Eigen::Ref<Eigen::MatrixXf> joint_jtj,
	Eigen::Ref<Eigen::VectorXf> joint_jte
) {
	//Fetch info
	const auto& body_map = kinematic.GetBody2GeometricMap();
	const auto& kinematic_tree = kinematic.GetKinematicTree();
	const auto& cache = kinematic.GetKinematicCache();
	
	//Simple sanity check and zero-init
	LOG_ASSERT(kinematic_tree.get_num_positions() == joint_jtj.rows());
	LOG_ASSERT(kinematic_tree.get_num_positions() == joint_jtj.cols());
	LOG_ASSERT(kinematic_tree.get_num_positions() == joint_jte.rows());
	joint_jte.setZero();
	joint_jtj.setZero();
	
	//The index for geometric jacobian
	std::vector<int> geometric_jacobian_idx;
	geometric_jacobian_idx.reserve(kinematic_tree.get_num_positions());
	Eigen::MatrixXf full_geometric_jacobian;
	full_geometric_jacobian.resize(6, kinematic_tree.get_num_positions());
	
	//Construct the full geometric jacobian from partial jacobian
	auto fill_geometric_jacobian = [&](const Eigen::Ref<const Eigen::MatrixXd>& geometric_jacobian) -> void {
		LOG_ASSERT(geometric_jacobian.cols() == geometric_jacobian_idx.size());
		full_geometric_jacobian.setZero();
		for(auto col_idx = 0; col_idx < geometric_jacobian.cols(); col_idx++) {
			for(auto r_idx = 0; r_idx < 6; r_idx++)
				full_geometric_jacobian(r_idx, geometric_jacobian_idx[col_idx]) = static_cast<float>(geometric_jacobian(r_idx, col_idx));
		}
	};

	//Iterate over each body
	for (auto body_i = 0; body_i < body_map.size(); body_i++) {
		const auto body_idx = body_map[body_i].body_index;

		//Construct the eigen map
		const Eigen::Map<const Eigen::Matrix6f> twist_jtj(body_jtj + 36 * body_i);
		const Eigen::Map<const Eigen::Vector6f> twist_jte(body_jte + 6 * body_i);

		//Get the geometric jacobian
		geometric_jacobian_idx.clear();
		const Eigen::MatrixXd geometric_jacobian = kinematic_tree.geometricJacobian(
			cache, 0, body_idx, 0,
			true,
			&geometric_jacobian_idx);
		fill_geometric_jacobian(geometric_jacobian);
		
		//Write to joint space jtj
		joint_jtj += full_geometric_jacobian.transpose() * (twist_jtj * full_geometric_jacobian);
		joint_jte += full_geometric_jacobian.transpose() * twist_jte;
	}
}