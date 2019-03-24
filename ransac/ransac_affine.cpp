//
// Created by wei on 12/3/18.
//

#include "ransac/ransac_affine.h"
#include "ransac/ransac_hypothesis.h"
#include "ransac/hypothesis_evaluate.h"
#include "geometry_utils/device2eigen.h"

void poser::RansacAffine::generateTransformHypoethesis(
	const poser::FeatureMap &observation,
	const poser::FeatureMap &model
) {
	//Get the output
	const auto& selected_model_point = selected_model_point_.GetTypedTensorReadOnly<float4>();
	const auto& model_corr_point = model_corresponded_point_.GetTypedTensorReadOnly<float4>();
	auto generated_hypothesis = transform_hypothesis_.GetTypedTensorReadWrite<mat34>();
	void* local_buffer = model_nn_index_.RawPtr();
	
	//Check the size
	LOG_ASSERT(selected_model_point.Size() == model_corr_point.Size());
	LOG_ASSERT(generated_hypothesis.Size() == ransac_parameter_.n_samples);
	LOG_ASSERT(selected_model_point.Size() == ransac_parameter_.n_point_per_sample * generated_hypothesis.Size());
	
	//Invoke the method
	computeRansacHypothesisAffine(
		selected_model_point, model_corr_point,
		ransac_parameter_.n_samples, ransac_parameter_.n_point_per_sample,
		generated_hypothesis, local_buffer);
}

void poser::RansacAffine::CheckAndAllocate(const poser::FeatureMap &observation, const poser::FeatureMap &model) {
	//The basic checking method
	RansacBase::CheckAndAllocate(observation, model);
	
	//Allocate the model point buffer
	transform_hypothesis_.Reset<mat34>(ransac_parameter_.n_samples);
	model.CloneTo(mutable_model_);
	kinematic_.CheckGeometricModelAndAllocateAttribute(mutable_model_);
}

void poser::RansacAffine::evaluateHypothesisAffine() {
	const auto& pose_hypothesis = transform_hypothesis_.GetTypedTensorReadOnly<mat34>();
	auto hypothesis_fitness = hypothesis_fitness_.GetTypedTensorReadWrite<FitnessEvaluator::Result>();
	::poser::evaluteRansacHypothesisRigidAndAffine<AffineKinematicModel>(
		fitness_evaluator_,
		mutable_model_, kinematic_,
		pose_hypothesis,
		//The output
		hypothesis_fitness, hypothesis_flag_);
}

void poser::RansacAffine::Compute(const poser::FeatureMap &observation, const poser::FeatureMap &model) {
	//The pre-set methods
	std::fill(hypothesis_flag_.begin(), hypothesis_flag_.end(), true);
	
	//These are efficient
	RansacBase::generateHypothesisCorrespondence(observation, model);
	generateTransformHypoethesis(observation, model);
	
	//This can be very slow
	evaluateHypothesisAffine();
}

Eigen::Matrix4f poser::RansacAffine::GetBestHypothesisAffineTransform() const {
	const auto& pose_hypothesis = transform_hypothesis_.GetTypedTensorReadOnly<mat34>();
	const auto& hypothesis_fitness = hypothesis_fitness_.GetTypedTensorReadOnly<FitnessEvaluator::Result>();
	LOG_ASSERT(pose_hypothesis.Size() == hypothesis_fitness.Size());
	
	//Just iterate through the hypothesis
	int best_idx = -1;
	float best_fitness = -1.0f;
	for(auto i = 0; i < pose_hypothesis.Size(); i++) {
		if(hypothesis_fitness[i].fitness_score > best_fitness) {
			best_fitness = hypothesis_fitness[i].fitness_score;
			best_idx = i;
		}
	}
	
	//Return the pose at best index
	return to_eigen(pose_hypothesis[best_idx]);
}