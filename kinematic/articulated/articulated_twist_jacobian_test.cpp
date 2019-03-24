//
// Created by wei on 9/30/18.
//

#include <gtest/gtest.h>

#include "kinematic/articulated/articulated_twist2joint.h"

#include <drake/multibody/rigid_body_tree.h>
#include <drake/common/find_resource.h>
#include <drake/multibody/parsers/urdf_parser.h>

class TwistJacobianTest : public ::testing::Test {
protected:
	void SetUp() override {
		using namespace drake;
		const char kIiwaUrdf[] = "drake/manipulation/models/iiwa_description/urdf/iiwa14_polytope_collision.urdf";
		auto urdf_path = drake::FindResourceOrThrow(kIiwaUrdf);
		drake::parsers::urdf::AddModelInstanceFromUrdfFileToWorld(
			urdf_path, drake::multibody::joints::kRollPitchYaw, &iiwa_tree_);
		
		//setup q
		q_.resize(iiwa_tree_.get_num_positions());
		q_.setRandom();
		
		//setup kinematic cache'
		cache_.reset(new KinematicsCache<double>(iiwa_tree_.CreateKinematicsCacheWithType<double>()));
		cache_->initialize(q_);
		iiwa_tree_.doKinematics(*cache_);
		
		//Must use one that depends on all the joint position
		test_body_idx_ = 10;
	}
	
	RigidBodyTree<double> iiwa_tree_;
	Eigen::VectorXd q_;
	std::unique_ptr<KinematicsCache<double>> cache_;
	int test_body_idx_;
};

TEST_F(TwistJacobianTest, RawJacobianTest) {
	Eigen::Vector3d point; point.setZero();
	auto world_point = iiwa_tree_.transformPoints(*cache_, point, test_body_idx_, 0);
	auto point_jacobian = iiwa_tree_.transformPointsJacobian(*cache_, point, test_body_idx_, 0, true);
	EXPECT_EQ(point_jacobian.rows(), 3);
	
	//Init the geometric jacobian
	auto geometric_jacobian = iiwa_tree_.geometricJacobian(*cache_, 0, test_body_idx_, 0, true);
	EXPECT_EQ(point_jacobian.cols(), geometric_jacobian.cols());
	LOG(INFO) << geometric_jacobian.cols();
	
	//Init the point jacobian from geometric jacobian
	Eigen::MatrixXd point_jacobian_from_twist;
	point_jacobian_from_twist.resize(3, geometric_jacobian.cols());
	point_jacobian_from_twist.setZero();
	
	//Test of twist jacobian
	{
		drake::Vector6<double> dx_dtwist; dx_dtwist.setZero();
		dx_dtwist(0) = 0.0; dx_dtwist(1) = world_point(2); dx_dtwist(2) = - world_point(1);
		dx_dtwist(3) = 1.0; dx_dtwist(4) = 0.0; dx_dtwist(5) = 0.0;
		Eigen::MatrixXd dx_dq = dx_dtwist.transpose() * geometric_jacobian;
		point_jacobian_from_twist.row(0) = dx_dq;
		
		drake::Vector6<double> dy_dtwist; dy_dtwist.setZero();
		dy_dtwist(0) = - world_point(2); dy_dtwist(1) = 0; dy_dtwist(2) = world_point(0);
		dy_dtwist(3) = 0.0; dy_dtwist(4) = 1.0; dy_dtwist(5) = 0.0;
		Eigen::MatrixXd dy_dq = dy_dtwist.transpose() * geometric_jacobian;
		point_jacobian_from_twist.row(1) = dy_dq;
		
		drake::Vector6<double> dz_dtwist; dz_dtwist.setZero();
		dz_dtwist(0) = world_point(1); dz_dtwist(1) = - world_point(0); dz_dtwist(2) = 0.0;
		dz_dtwist(3) = 0.0; dz_dtwist(4) = 0.0; dz_dtwist(5) = 1.0;
		Eigen::MatrixXd dz_dq = dz_dtwist.transpose() * geometric_jacobian;
		point_jacobian_from_twist.row(2) = dz_dq;
	}
	
	//Compare the result
	for(auto r_idx = 0; r_idx < 3; r_idx++) {
		for(auto c_idx = 0; c_idx < point_jacobian_from_twist.cols(); c_idx++)
			EXPECT_NEAR(point_jacobian_from_twist(r_idx, c_idx), point_jacobian(r_idx, c_idx), 1e-4f);
	}
}

TEST_F(TwistJacobianTest, JtJTest) {
	//Compute jtj by raw point jacobian
	Eigen::Vector3d point; point.setZero();
	auto world_point = iiwa_tree_.transformPoints(*cache_, point, test_body_idx_, 0);
	auto point_jacobian = iiwa_tree_.transformPointsJacobian(*cache_, point, test_body_idx_, 0, true);
	EXPECT_EQ(point_jacobian.rows(), 3);
	Eigen::MatrixXd point_jtj = point_jacobian.transpose() * point_jacobian;
	EXPECT_EQ(point_jtj.rows(), iiwa_tree_.get_num_positions());
	EXPECT_EQ(point_jtj.cols(), iiwa_tree_.get_num_positions());

	//Init the geometric jacobian
	auto geometric_jacobian = iiwa_tree_.geometricJacobian(*cache_, 0, test_body_idx_, 0, true);
	EXPECT_EQ(point_jacobian.cols(), geometric_jacobian.cols());
	EXPECT_EQ(geometric_jacobian.rows(), 6);

	//The twist jacobian term
	Eigen::Matrix<double, 6, 6> twist_jtj;
	twist_jtj.setZero();
	{
		drake::Vector6<double> dx_dtwist; dx_dtwist.setZero();
		dx_dtwist(0) = 0.0; dx_dtwist(1) = world_point(2); dx_dtwist(2) = - world_point(1);
		dx_dtwist(3) = 1.0; dx_dtwist(4) = 0.0; dx_dtwist(5) = 0.0;
		twist_jtj += dx_dtwist * dx_dtwist.transpose();

		drake::Vector6<double> dy_dtwist; dy_dtwist.setZero();
		dy_dtwist(0) = - world_point(2); dy_dtwist(1) = 0; dy_dtwist(2) = world_point(0);
		dy_dtwist(3) = 0.0; dy_dtwist(4) = 1.0; dy_dtwist(5) = 0.0;
		twist_jtj += dy_dtwist * dy_dtwist.transpose();

		drake::Vector6<double> dz_dtwist; dz_dtwist.setZero();
		dz_dtwist(0) = world_point(1); dz_dtwist(1) = - world_point(0); dz_dtwist(2) = 0.0;
		dz_dtwist(3) = 0.0; dz_dtwist(4) = 0.0; dz_dtwist(5) = 1.0;
		twist_jtj += dz_dtwist * dz_dtwist.transpose();
	}

	//Compute and compare the result
	Eigen::MatrixXd point_jtj_from_twist = geometric_jacobian.transpose() * twist_jtj * geometric_jacobian;
	for (auto r_idx = 0; r_idx < iiwa_tree_.get_num_positions(); r_idx++) {
		for (auto c_idx = 0; c_idx < iiwa_tree_.get_num_positions(); c_idx++) {
			EXPECT_NEAR(point_jtj(r_idx, c_idx), point_jtj_from_twist(r_idx, c_idx), 1e-4f);
		}
	}
}


TEST_F(TwistJacobianTest, JtErrorTest) {
	//Compute jtj by raw point jacobian
	Eigen::Vector3d point; point.setZero();
	auto world_point = iiwa_tree_.transformPoints(*cache_, point, test_body_idx_, 0);
	auto point_jacobian = iiwa_tree_.transformPointsJacobian(*cache_, point, test_body_idx_, 0, true);
	EXPECT_EQ(point_jacobian.rows(), 3);

	//Some stupid error...
	Eigen::Vector3d residual = world_point - point;
	Eigen::MatrixXd point_jte = - residual.transpose() * point_jacobian;
	EXPECT_EQ(point_jte.rows(), 1);
	EXPECT_EQ(point_jte.cols(), iiwa_tree_.get_num_positions());

	//Init the geometric jacobian
	auto geometric_jacobian = iiwa_tree_.geometricJacobian(*cache_, 0, test_body_idx_, 0, true);
	EXPECT_EQ(point_jacobian.cols(), geometric_jacobian.cols());
	EXPECT_EQ(geometric_jacobian.rows(), 6);

	//The twist jacobian term
	Eigen::Matrix<double, 6, 1> twist_jte;
	twist_jte.setZero();
	{
		drake::Vector6<double> dx_dtwist; dx_dtwist.setZero();
		dx_dtwist(0) = 0.0; dx_dtwist(1) = world_point(2); dx_dtwist(2) = - world_point(1);
		dx_dtwist(3) = 1.0; dx_dtwist(4) = 0.0; dx_dtwist(5) = 0.0;
		twist_jte += - residual(0) * dx_dtwist;

		drake::Vector6<double> dy_dtwist; dy_dtwist.setZero();
		dy_dtwist(0) = - world_point(2); dy_dtwist(1) = 0; dy_dtwist(2) = world_point(0);
		dy_dtwist(3) = 0.0; dy_dtwist(4) = 1.0; dy_dtwist(5) = 0.0;
		twist_jte += - residual(1) * dy_dtwist;

		drake::Vector6<double> dz_dtwist; dz_dtwist.setZero();
		dz_dtwist(0) = world_point(1); dz_dtwist(1) = - world_point(0); dz_dtwist(2) = 0.0;
		dz_dtwist(3) = 0.0; dz_dtwist(4) = 0.0; dz_dtwist(5) = 1.0;
		twist_jte += - residual(2) * dz_dtwist;
	}

	Eigen::MatrixXd point_jte_from_twist = (twist_jte.transpose() * geometric_jacobian);
	EXPECT_EQ(point_jte_from_twist.rows(), 1);
	EXPECT_EQ(point_jte_from_twist.cols(), iiwa_tree_.get_num_positions());
	for (auto r_idx = 0; r_idx < 1; r_idx++) {
		for (auto c_idx = 0; c_idx < iiwa_tree_.get_num_positions(); c_idx++) {
			EXPECT_NEAR(point_jte(r_idx, c_idx), point_jte_from_twist(r_idx, c_idx), 1e-4f);
		}
	}
}