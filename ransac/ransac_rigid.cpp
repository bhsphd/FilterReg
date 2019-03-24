
//
// Created by wei on 1/16/19.
//

#include "ransac/ransac_rigid.h"
#include "ransac/ransac_hypothesis.h"
#include "ransac/hypothesis_evaluate.h"
#include "ransac/so3_sample.h"


void poser::RansacRigidBase::CheckAndAllocate(const poser::FeatureMap &observation, const poser::FeatureMap &model) {
	//The basic checking method
	RansacBase::CheckAndAllocate(observation, model);
	sorted_fitness_.reserve(ransac_parameter_.n_samples);
	
	//Allocate the model point buffer
	pose_hypothesis_.Reset<mat34>(ransac_parameter_.n_samples);
	model.CloneTo(mutable_model_);
	kinematic_.CheckGeometricModelAndAllocateAttribute(mutable_model_);
}

void poser::RansacRigidBase::evaluateHypothesisRigid() {
	const auto& pose_hypothesis = pose_hypothesis_.GetTypedTensorReadOnly<mat34>();
	auto hypothesis_fitness = hypothesis_fitness_.GetTypedTensorReadWrite<FitnessEvaluator::Result>();
	::poser::evaluteRansacHypothesisRigidAndAffine<RigidKinematicModel>(
		fitness_evaluator_,
		mutable_model_, kinematic_,
		pose_hypothesis,
		//The output
		hypothesis_fitness, hypothesis_flag_);
}

void poser::RansacRigidBase::EvaluateHypothesisRigid(
	const poser::TensorView<poser::mat34> &pose_in,
	poser::TensorSlice<poser::FitnessEvaluator::Result> result
) {
	LOG_ASSERT(pose_in.Size() == result.Size());
	::poser::evaluteRansacHypothesisRigidAndAffine<RigidKinematicModel>(
		fitness_evaluator_,
		mutable_model_, kinematic_,
		pose_in,
		//The output
		result, std::vector<bool>());
}

void poser::RansacRigidBase::Compute(const poser::FeatureMap &observation, const poser::FeatureMap &model) {
	//The pre-set methods
	std::fill(hypothesis_flag_.begin(), hypothesis_flag_.end(), true);
	
	//These is efficient (although virtual)
	generatePoseHypoethesis(observation, model);
	
	//This can be very slow
	evaluateHypothesisRigid();
	
	//Sort the evaluated hypothesis
	sortEvaluatedHypothesis();
}

void poser::RansacRigidBase::sortEvaluatedHypothesis() {
	//Get and check the input
	const auto& pose_hypothesis = pose_hypothesis_.GetTypedTensorReadOnly<mat34>();
	const auto& hypothesis_fitness = hypothesis_fitness_.GetTypedTensorReadOnly<FitnessEvaluator::Result>();
	LOG_ASSERT(pose_hypothesis.Size() == hypothesis_fitness.Size());
	
	//Copy the value
	sorted_fitness_.resize(hypothesis_fitness.Size());
	for(auto i = 0; i < sorted_fitness_.size(); i++) {
		sorted_fitness_[i].score = hypothesis_fitness[i].fitness_score;
		sorted_fitness_[i].index = i;
	}
	
	//Get it
	std::sort(sorted_fitness_.begin(), sorted_fitness_.end());
}

poser::mat34 poser::RansacRigidBase::GetBestHypothesisPose() const {
	const auto& pose_hypothesis = pose_hypothesis_.GetTypedTensorReadOnly<mat34>();
	const auto index = sorted_fitness_[sorted_fitness_.size() - 1].index;
	return pose_hypothesis[index];
}

void poser::RansacRigidBase::GetTopKHypothesisPose(int k, poser::TensorSlice<poser::mat34> pose_out) const {
	const auto& pose_hypothesis = pose_hypothesis_.GetTypedTensorReadOnly<mat34>();
	LOG_ASSERT(k < pose_hypothesis.Size());
	LOG_ASSERT(k == pose_out.Size());
	
	//Write to output
	for(auto i = 0; i < k; i++) {
		const auto index = sorted_fitness_[sorted_fitness_.size() - 1 - i].index;
		pose_out[i] = pose_hypothesis[index];
	}
}

//For the featured version
void poser::RansacRigidFeature::generatePoseHypoethesis(
	const poser::FeatureMap &observation,
	const poser::FeatureMap &model
) {
	//First build the feature correspondence
	RansacBase::generateHypothesisCorrespondence(observation, model);
	
	//Get the output
	const auto& selected_model_point = selected_model_point_.GetTypedTensorReadOnly<float4>();
	const auto& model_corr_point = model_corresponded_point_.GetTypedTensorReadOnly<float4>();
	auto generated_hypothesis = pose_hypothesis_.GetTypedTensorReadWrite<mat34>();
	void* local_buffer = (void*) (model_nn_index_.RawPtr());
	
	//Check the size
	LOG_ASSERT(selected_model_point.Size() == model_corr_point.Size());
	LOG_ASSERT(generated_hypothesis.Size() == ransac_parameter_.n_samples);
	LOG_ASSERT(selected_model_point.Size() == ransac_parameter_.n_point_per_sample * generated_hypothesis.Size());
	
	//Invoke the method
	computeRansacHypothesisRigid(
		selected_model_point, model_corr_point,
		ransac_parameter_.n_samples, ransac_parameter_.n_point_per_sample,
		generated_hypothesis, local_buffer);
}

//For the random version
void poser::RansacRigidRandom::generatePoseHypoethesis(
	const poser::FeatureMap &observation,
	const poser::FeatureMap &model
) {
	auto pose_hypothesis = pose_hypothesis_.GetTypedTensorReadWrite<mat34>();
	LOG_ASSERT(pose_hypothesis.Size() == ransac_parameter_.n_samples);
	randomUniformSampleSO3Space(pose_hypothesis.RawPtr(), ransac_parameter_.n_samples);
}