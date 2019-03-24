
//
// Created by wei on 1/16/19.
//

#include "kinematic/cpd/cpd_kinematic_model.h"

poser::CoherentPointDriftKinematic::CoherentPointDriftKinematic(
	float regularizer_sigma,
	poser::FeatureChannelType reference_vertex,
	poser::FeatureChannelType live_vertex
) : KinematicModelBase(
		MemoryContext::CpuMemory,
		std::move(reference_vertex),
		std::move(live_vertex)),
	beta_(regularizer_sigma)
{
	//Nothing here
}

void poser::CoherentPointDriftKinematic::CheckGeometricModelAndAllocateAttribute(poser::FeatureMap &geometric_model) {
	//Check it using base method
	KinematicModelBase::CheckGeometricModelAndAllocateAttribute(geometric_model);
	
	//Do further allocation
	const auto& ref_points = static_cast<const FeatureMap&>(geometric_model).GetDenseFeatureRawBlob(
		reference_vertex_channel_.get_name_key(), MemoryContext::CpuMemory);
	ref_points.CloneTo(reference_points_);
	
	//Build kdtree
	reference_points_kdtree_.ResetInputData(reference_points_);
	
	//Allocate the output
	const auto n_points = ref_points.TensorFlattenSize();
	w_.resize(n_points, 3); w_.setZero();
	v_.resize(n_points, 3); v_.setZero();
	
	//Build the G matrix
	G_.resize(n_points, n_points);
	buildRegularizerMatrixFromReferencePoints();
}

//The method to build the G matrix
void poser::CoherentPointDriftKinematic::buildRegularizerMatrixFromReferencePoints() {
	const auto ref_points = reference_points_.GetTensorReadOnly();
	const auto n_points = ref_points.FlattenSize();
	LOG_ASSERT(n_points > 0);
	
	//Invoke the method
	G_.resize(n_points, n_points);
	ComputeRegularizerMatrixDense(G_, ref_points, beta_);
}

void poser::CoherentPointDriftKinematic::ComputeRegularizerMatrixDense(
	Eigen::Ref<Eigen::MatrixXf> G,
	const poser::BlobView &reference_points,
	float beta
) {
	//Check the size of G
	const auto n_points = reference_points.FlattenSize();
	LOG_ASSERT(G.rows() == n_points);
	LOG_ASSERT(G.cols() == n_points);
	
	//Check the size of reference_points
	LOG_ASSERT(reference_points.ValidTypeByte() == sizeof(float3));
	const float inv_sigma_square = 1.0f / (beta * beta);
	
	//The iteration on points
	for(auto i = 0; i < n_points; i++) {
		const auto point_i = reference_points.ValidElemVectorAt<float>(i);
		for(auto j = i; j < n_points; j++) {
			//Retreive the point j
			const auto point_j = reference_points.ValidElemVectorAt<float>(j);
			
			//Compute the squared distance
			float dist_ij = 0;
			for(auto k = 0; k < point_i.typed_size; k++) {
				dist_ij += (point_i[k] - point_j[k]) * (point_i[k] - point_j[k]);
			}
			
			//Assign the matrix
			float g_ij = expf(-0.5f * dist_ij * inv_sigma_square);
			G(i, j) = g_ij;
		}
	}
	
	//The other half
	for(auto i = 0; i < n_points; i++) {
		for(auto j = i + 1; j < n_points; j++) {
			G(j, i) = G(i, j);
		}
	}
}

//Update of kinematic model
void poser::CoherentPointDriftKinematic::SetMotionParameterW(const Eigen::Ref<const Eigen::MatrixXf> &w) {
	//Check the size of input
	const auto n_point = reference_points_.TensorFlattenSize();
	LOG_ASSERT(w.rows() == n_point) << "Motion parameter w size mismatch";
	LOG_ASSERT(w.cols() == 3) << "Motion parameter w size mismatch";
	
	w_ = w;
	UpdateKinematicModelGivenMotionParameter();
}

void poser::CoherentPointDriftKinematic::UpdateKinematicModelGivenMotionParameter() {
	v_ = G_ * w_;
}