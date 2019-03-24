
//
// Created by wei on 1/16/19.
//

#pragma once

#include "common/feature_channel_type.h"
#include "common/feature_map.h"
#include "common/common_type.h"
#include "geometry_utils/device_mat.h"
#include "geometry_utils/kdtree_flann.h"
#include "kinematic/kinematic_model_base.h"

#include <Eigen/Dense>


namespace poser {
	
	
	class CoherentPointDriftKinematic : public KinematicModelBase {
	public:
		//The constructor
		explicit CoherentPointDriftKinematic(
			float regularizer_sigma,
			FeatureChannelType reference_vertex = FeatureChannelType(),
			FeatureChannelType live_vertex = FeatureChannelType());
		
		//Need to construct G_ and zero-initialize w in this method
		void CheckGeometricModelAndAllocateAttribute(FeatureMap& geometric_model) override;
		
		//The reference point in the size of (n_points, 3)
		//Either a direct copy or subsample of reference_vertex
	protected:
		TensorBlob reference_points_;
		KDTreeKNN reference_points_kdtree_;
	public:
		TensorView<float4> GetReferencePoints() const { return reference_points_.GetTypedTensorReadOnly<float4>(); }
		

	  protected:
		//The G matrix in the size of (n_points, n_points)
		//The matrix should be constructed only once
		//Considering using a sparse matrix instead
		Eigen::MatrixXf G_;
		float beta_; //G_(i, j) = GaussKernel(xi; 0, beta^2)
		void buildRegularizerMatrixFromReferencePoints();
	public:
		//Compute the DENSE regularizer matrix from ref-point
		static void ComputeRegularizerMatrixDense(
			Eigen::Ref<Eigen::MatrixXf> G,
			const BlobView& reference_points,
			float beta);
		
		
		//Method and members used for motion parameter
	protected:
		//The motion parameter, should be in the size of (n_points, 3)
		Eigen::MatrixXf w_;
		
		//The actual motion field in the size of (n_points, 3): v = G w
		Eigen::MatrixXf v_;
	
	public:
		//Update the kinematic model given w
		void SetMotionParameterW(const Eigen::Ref<const Eigen::MatrixXf>& w);
		void UpdateKinematicModelGivenMotionParameter();
		
		//The getter for members
		const Eigen::MatrixXf& GetMotionRegularizerMatrix() const { return G_; }
		const Eigen::MatrixXf& GetMotionVectorField() const { return v_; }
	};
}
