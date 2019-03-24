//
// Created by wei on 9/30/18.
//

#include <gtest/gtest.h>

#include "common/geometric_target_interface.h"
#include "kinematic/articulated/articulated_twist2joint.h"
#include "kinematic/articulated/articulated_point2point_cpu.h"
#include "kinematic/articulated/articulated_geometric_update.h"

#include <drake/multibody/rigid_body_tree.h>
#include <drake/common/find_resource.h>
#include <drake/multibody/parsers/urdf_parser.h>

#include <fstream>

class TwistPoint2PointTest : public ::testing::Test {
protected:
	void SetUp() override {
		using namespace drake;
		//Load the rigid boyd tree
		RigidBodyTree<double> tree;
		const char kIiwaUrdf[] = "drake/manipulation/models/iiwa_description/urdf/iiwa14_polytope_collision.urdf";
		auto urdf_path = FindResourceOrThrow(kIiwaUrdf);
		parsers::urdf::AddModelInstanceFromUrdfFileToWorld(
			urdf_path, multibody::joints::kRollPitchYaw, &tree);
		
		using namespace poser;
		kinematic_model_ = std::make_unique<ArticulatedKinematicModel>(MemoryContext::CpuMemory, tree);
		
		//Load the geometric model
		json j_model;
		std::ifstream model_in("/home/wei/Documents/programs/poser/tmp/iiwa_model.json");
		model_in >> j_model;
		model_in.close();
		
		//Load the model
		geometric_model_ = j_model.get<FeatureMap>();
		kinematic_model_->CheckGeometricModelAndAllocateAttribute(geometric_model_);
		
		//Do simple kinematic
		Eigen::VectorXd q; q.resize(tree.get_num_positions());
		q.setZero();
		kinematic_model_->SetMotionParameter(q);
		kinematic_model_->DoKinematicAndUpdateBodyPoseNoSync();
		
		//Update the live vertex
		UpdateLiveVertexCPU(*kinematic_model_, geometric_model_);
		
		//Allocate the target
		dense_target_.AllocateDenseTargetForModel(geometric_model_, MemoryContext::CpuMemory);
	}
	
	std::unique_ptr<poser::ArticulatedKinematicModel> kinematic_model_;
	poser::FeatureMap geometric_model_;
	poser::GeometricTarget dense_target_;
};

TEST_F(TwistPoint2PointTest, JtJTest) {
	using namespace poser;
	const auto& kinematic_tree = kinematic_model_->GetKinematicTree();
	const auto& cache = kinematic_model_->GetKinematicCache();
	auto live_vertex_channel = kinematic_model_->LiveVertexChannel();
	auto live_vertex = geometric_model_.GetTypedFeatureValueReadOnly<float4>(live_vertex_channel, MemoryContext::CpuMemory);
	auto ref_vertex_channel = kinematic_model_->ReferenceVertexChannel();
	auto ref_vertex = geometric_model_.GetTypedFeatureValueReadOnly<float4>(ref_vertex_channel, MemoryContext::CpuMemory);
	
	//Check the size of target
	auto target_vertex = dense_target_.GetTargetVertexReadWrite();
	EXPECT_EQ(target_vertex.Size(), live_vertex.Size());
	
	//Some stupid target
	float offset = 0.01f;
	for(auto i = 0; i < live_vertex.Size(); i++) {
		auto& target_i = target_vertex[i];
		const auto live_vertex_i = live_vertex[i];
		target_i = live_vertex_i;
		target_i.x += offset;
		target_i.w = 1.0f;
	}
	
	//The assembler
	ArticulatedPoint2PointTermAssemblerCPU assembler;
	assembler.CheckAndAllocate(geometric_model_, *kinematic_model_, dense_target_);
	
	//The output by assembler
	Eigen::MatrixXf JtJ;
	JtJ.resize(kinematic_tree.get_num_positions(), kinematic_tree.get_num_positions());
	Eigen::VectorXf JtError;
	JtError.resize(kinematic_tree.get_num_positions());
	
	//Do it
	assembler.ProcessAssembleDenseTerm(
		geometric_model_,
		*kinematic_model_,
		dense_target_,
		JtJ, JtError);
	
	//Compute it using point jacobian
	Eigen::MatrixXd JtJ_point;
	JtJ_point.resizeLike(JtJ);
	JtJ_point.setZero();

	const auto &body_map = kinematic_model_->GetBody2GeometricMap();
	for(const auto& body : body_map) {
		auto body_idx = body.body_index;
		auto start_idx = body.geometry_start;
		auto end_idx = body.geometry_end;

		for (auto vert_idx = start_idx; vert_idx < end_idx; vert_idx++) {
			//Fetch the point
			const auto body_vertex = ref_vertex[vert_idx];
			const auto target_vert = target_vertex[vert_idx];
			Eigen::Vector3d point;
			point(0) = body_vertex.x;
			point(1) = body_vertex.y;
			point(2) = body_vertex.z;
			
			auto point_jacobian = kinematic_tree.transformPointsJacobian(cache, point, body_idx, 0, true);
			JtJ_point += target_vert.w * point_jacobian.transpose() * point_jacobian;
		}
	}

	//Compare it
	for (auto r_idx = 0; r_idx < JtJ.rows(); r_idx++) {
		for (auto c_idx = 0; c_idx < JtJ.cols(); c_idx++) {
			const auto point_value = static_cast<float>(JtJ_point(r_idx, c_idx));
			const auto twist_value = JtJ(r_idx, c_idx);
			const auto relative_error = std::abs((point_value - twist_value) / (std::abs(twist_value) + 1e-4f));
			EXPECT_TRUE(relative_error < 1e-2f)
			<< "The value from point is " << point_value << " while from twist is " << twist_value;
		}
	}
}


TEST_F(TwistPoint2PointTest, JtErrorTest) {
	using namespace poser;
	const auto& kinematic_tree = kinematic_model_->GetKinematicTree();
	const auto& cache = kinematic_model_->GetKinematicCache();
	auto live_vertex_channel = kinematic_model_->LiveVertexChannel();
	auto live_vertex = geometric_model_.GetTypedFeatureValueReadOnly<float4>(live_vertex_channel, MemoryContext::CpuMemory);
	auto ref_vertex_channel = kinematic_model_->ReferenceVertexChannel();
	auto ref_vertex = geometric_model_.GetTypedFeatureValueReadOnly<float4>(ref_vertex_channel, MemoryContext::CpuMemory);
	
	//Check the size of target
	auto target_vertex = dense_target_.GetTargetVertexReadWrite();
	EXPECT_EQ(target_vertex.Size(), live_vertex.Size());
	
	//Some stupid target
	float offset = 0.01f;
	for(auto i = 0; i < live_vertex.Size(); i++) {
		auto& target_i = target_vertex[i];
		const auto live_vertex_i = live_vertex[i];
		target_i = live_vertex_i;
		target_i.x += offset;
		target_i.w = 1.0f;
	}
	
	//The assembler
	ArticulatedPoint2PointTermAssemblerCPU assembler;
	assembler.CheckAndAllocate(geometric_model_, *kinematic_model_, dense_target_);
	
	//The output by assembler
	Eigen::MatrixXf JtJ;
	JtJ.resize(kinematic_tree.get_num_positions(), kinematic_tree.get_num_positions());
	Eigen::VectorXf JtError;
	JtError.resize(kinematic_tree.get_num_positions());
	
	//Do it
	assembler.ProcessAssembleDenseTerm(
		geometric_model_,
		*kinematic_model_,
		dense_target_,
		JtJ, JtError);

	//Construct using point jacobian
	Eigen::VectorXd JtError_point;
	JtError_point.resizeLike(JtError);
	JtError_point.setZero();

	const auto &body_map = kinematic_model_->GetBody2GeometricMap();
	for(const auto& body : body_map) {
		auto body_idx = body.body_index;
		auto start_idx = body.geometry_start;
		auto end_idx = body.geometry_end;

		for (auto vert_idx = start_idx; vert_idx < end_idx; vert_idx++) {
			//Fetch the point
			const auto body_vertex = ref_vertex[vert_idx];
			const auto target_vert = target_vertex[vert_idx];
			Eigen::Vector3d point, residual;
			point(0) = body_vertex.x;
			point(1) = body_vertex.y;
			point(2) = body_vertex.z;
			residual(0) = live_vertex[vert_idx].x - target_vert.x;
			residual(1) = live_vertex[vert_idx].y - target_vert.y;
			residual(2) = live_vertex[vert_idx].z - target_vert.z;

			auto point_jacobian = kinematic_tree.transformPointsJacobian(cache, point, body_idx, 0, true);
			JtError_point += (-target_vert.w) * point_jacobian.transpose() * residual;
		}
	}


	//Compare it
	for (auto r_idx = 0; r_idx < JtError.rows(); r_idx++) {
		for (auto c_idx = 0; c_idx < JtError.cols(); c_idx++) {
			const auto point_value = static_cast<float>(JtError_point(r_idx, c_idx));
			const auto twist_value = JtError(r_idx, c_idx);
			const auto relative_error = std::abs((point_value - twist_value) / (std::abs(twist_value) + 1e-4f));
			EXPECT_TRUE(relative_error < 1e-2f) << "For row " << r_idx
			<< " the value from point is " << point_value << " while from twist is " << twist_value;
		}
	}
}