#pragma once

#include "ransac/hypothesis_evaluate.h"

template<typename KinematicT>
void poser::evaluteRansacHypothesisRigidAndAffine(
	const poser::FitnessEvaluator &evaluator,
	poser::FeatureMap &model, KinematicT &kinematic,
	const poser::TensorView<poser::mat34> &pose_hypothesis,
	poser::TensorSlice<poser::FitnessEvaluator::Result> hypothesis_fitness,
	const std::vector<bool> &flag
) {
	//Sanity check
	LOG_ASSERT(pose_hypothesis.Size() == hypothesis_fitness.Size());
	LOG_ASSERT(flag.empty() || flag.size() == hypothesis_fitness.Size());
	using Result = FitnessEvaluator::Result;
	
	//Do it
	for(auto i = 0; i < pose_hypothesis.Size(); i++) {
		//If the flag is ready
		if(!flag.empty() && !flag[i]) {
			Result result;
			result.fitness_score = std::numeric_limits<float>::max();
			hypothesis_fitness[i] = result;
			continue;
		}
		
		//Pass the flag
		kinematic.SetMotionParameter(pose_hypothesis[i]);
		UpdateLiveVertexCPU(kinematic, model);
		auto result = evaluator.Evaluate(model);
		hypothesis_fitness[i] = result;
	}
}