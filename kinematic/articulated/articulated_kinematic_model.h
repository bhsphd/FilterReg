//
// Created by wei on 9/18/18.
//

#pragma once

#include "common/feature_channel_type.h"
#include "common/feature_map.h"
#include "common/common_type.h"
#include "geometry_utils/device_mat.h"
#include "kinematic/kinematic_model_base.h"

#include <drake/multibody/rigid_body_tree.h>

namespace poser {
	
	
	class ArticulatedKinematicModel : public KinematicModelBase {
	private:
		void resetBufferForTree();
		void releaseBuffer();
		void copyIndexAndTransform(const ArticulatedKinematicModel& rhs);
	public:
		explicit ArticulatedKinematicModel(
			MemoryContext context,
			const RigidBodyTree<double>& kinematic_tree,
			FeatureChannelType reference_vertex = FeatureChannelType(),
			FeatureChannelType live_vertex = FeatureChannelType(),
			FeatureChannelType reference_normal = FeatureChannelType(),
			FeatureChannelType live_normal = FeatureChannelType(),
			FeatureChannelType body_index = FeatureChannelType()
		);
		~ArticulatedKinematicModel() override;
		ArticulatedKinematicModel(const ArticulatedKinematicModel& rhs);
		ArticulatedKinematicModel& operator=(const ArticulatedKinematicModel& rhs);
		
		/* The member and method to maintain the map from
		 * body to all the vertex that belongs to that body
		 */
	private:
		struct RigidBodyElementMapType {
			int body_index;
			unsigned geometry_start;
			unsigned geometry_end;
		};
		vector<RigidBodyElementMapType> body2vertex_interval_map_;
		FeatureChannelType rigidbody_index_channel_;
		void buildBody2VertexMap(
			const FeatureMap& geometric_model,
			const TensorView<int>& body_index_view);
	public:
		void CheckGeometricModelAndAllocateAttribute(FeatureMap& geometric_model) override;
		const vector<RigidBodyElementMapType>& GetBody2GeometricMap() const { return body2vertex_interval_map_; }
		
		/* The member and method to maintain the kinematic
		 * of the robot and the pose of each rigid body.
		 * The method can only be invoked after body2vertex map is ready.
		 */
	private:
		//The actual kinematic information
		std::unique_ptr<RigidBodyTree<double>> kinematic_tree_; //This tree has no collision/visual elements
		KinematicsCache<double> kinematic_cache_;
		Eigen::VectorXd q_; //Generalized coordinate
		
		//The struct for rigid transformation for each body
		struct {
			mat34* body_transform_pagelocked;
			mat34* body_transform_dev;
			vector<Eigen::Isometry3f> body2world_isometry;
		} body_transformation_;
	public:
		//The method to update the motion parameter
		void SetMotionParameter(const Eigen::Ref<const Eigen::VectorXd>& q);
		void UpdateWithDeltaQ(const Eigen::Ref<const Eigen::VectorXf>& dq, cudaStream_t stream = 0);
		void DoKinematicAndUpdateBodyPoseNoSync(cudaStream_t stream = 0);
		
		//The accessing method
		TensorView<mat34> GetBody2WorldTransformCPU() const;
		TensorView<mat34> GetBody2WorldTransformGPU() const;
		const RigidBodyTree<double>& GetKinematicTree() const { return *kinematic_tree_; }
		const KinematicsCache<double>& GetKinematicCache() const { return kinematic_cache_; }
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};
}
