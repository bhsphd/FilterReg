//
// Created by wei on 12/2/18.
//

#include <limits>
#include "ransac/hypothesis_evaluate.h"


void poser::refineAndEvaluateHypothesis(
	const poser::FitnessEvaluator &evaluator,
	poser::FeatureMap &model, poser::RigidKinematicModel &kinematic,
	const poser::TensorView<poser::mat34> &pose_hypothesis,
	const poser::RigidPoseRefiner &refiner,
	poser::TensorSlice<poser::mat34> refined_pose,
	poser::TensorSlice<poser::FitnessEvaluator::Result> refined_fitness
) {

}