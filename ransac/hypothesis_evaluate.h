//
// Created by wei on 12/2/18.
//

#pragma once

#include "kinematic/rigid/rigid.h"
#include "ransac/fitness_base.h"
#include "geometry_utils/device_mat.h"

namespace poser {
	
	
	/* The method to evaluate the hypothesis. The most expensive step.
	 * Potentially support multi-thread/GPU parallelism. This one is
	 * single-threaded, multi-thread one built on top of this.
	 */
	template<typename KinematicT>
	void evaluteRansacHypothesisRigidAndAffine(
		const FitnessEvaluator& evaluator,
		FeatureMap& model, KinematicT& kinematic,
		const TensorView<mat34>& pose_hypothesis,
		TensorSlice<FitnessEvaluator::Result> hypothesis_fitness,
		const std::vector<bool>& flag = std::vector<bool>());
	
	/* The method to refine the pose hypothesis. Usually used to refine the
	 * best k hypothesis from the raw evaluation.
	 */
	using RigidPoseRefiner = std::function<void(
		FeatureMap& model,
		RigidKinematicModel& kinematic,
		const mat34& initial_pose,
		mat34& refined_pose)>;
	void refineAndEvaluateHypothesis(
		const FitnessEvaluator& evaluator,
		FeatureMap& model, RigidKinematicModel& kinematic,
		const TensorView<mat34>& pose_hypothesis, const RigidPoseRefiner& refiner,
		TensorSlice<mat34> refined_pose, TensorSlice<FitnessEvaluator::Result> refined_fitness);
}


#include "ransac/hypothesis_evaluate.hpp"