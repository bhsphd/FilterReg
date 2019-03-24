
//
// Created by wei on 1/16/19.
//

#pragma once

#include "ransac/ransac_common.h"
#include "kinematic/rigid/rigid.h"

namespace poser {
	
	
	class RansacRigidBase : public RansacBase {
	public:
		//The constructor group
		RansacRigidBase(
			RansacParameters parameters,
			RigidKinematicModel kinematic,
			const FitnessEvaluator& fitness
		) : RansacBase(std::move(parameters), fitness),
		    kinematic_(std::move(kinematic))
		{ LOG_ASSERT(ransac_parameter_.n_point_per_sample >= 3); };
		~RansacRigidBase() override = default;
	
		
		//The group of method to generate the hypothesis used for ransac
		//Depends on actual implementation, might be different
	protected:
		TensorBlob pose_hypothesis_;
		virtual void generatePoseHypoethesis(const FeatureMap& observation, const FeatureMap& model) = 0;
	public:
		void CheckAndAllocate(const FeatureMap& observation, const FeatureMap& model) override;
		
		
		//The group of method to evaluate the generated hypotheis
		//This is shared for rigid ransac
	protected:
		RigidKinematicModel kinematic_;
		FeatureMap mutable_model_;
		virtual void evaluateHypothesisRigid();
	public:
		void Compute(const FeatureMap& observation, const FeatureMap& model);
		//The evaluator interface use the internal buffer
		void EvaluateHypothesisRigid(
			const TensorView<mat34>& pose_in,
			TensorSlice<FitnessEvaluator::Result> result);
	
		
		//The group of the method used to get the best-k
		//fitness, shared for rigid ransac
	protected:
		struct ScoreAndIndex {
			float score;
			int index;
			//The comparator
			bool operator < (const ScoreAndIndex &rhs) const {
				return score < rhs.score;
			}
		};
		mutable std::vector<ScoreAndIndex> sorted_fitness_;
		void sortEvaluatedHypothesis();
	public:
		mat34 GetBestHypothesisPose() const;
		void GetTopKHypothesisPose(int k, TensorSlice<mat34> pose_out) const;
	};
	
	
	//The version using feature to establish hypothesis
	class RansacRigidFeature : public RansacRigidBase {
	public:
		RansacRigidFeature(
			RansacParameters parameters,
			RigidKinematicModel kinematic,
			const FitnessEvaluator& fitness
		) : RansacRigidBase(std::move(parameters),
		                    std::move(kinematic), fitness)
		{
			LOG_ASSERT(ransac_parameter_.obs_feature.is_valid());
			LOG_ASSERT(ransac_parameter_.model_feature.is_valid());
		}
		~RansacRigidFeature() override = default;
	
	protected:
		void generatePoseHypoethesis(const FeatureMap& observation, const FeatureMap& model) override;
	};
	
	
	//The version using random initialized hypothesis
	class RansacRigidRandom : public RansacRigidBase {
	public:
		RansacRigidRandom(
			RansacParameters parameters,
			RigidKinematicModel kinematic,
			const FitnessEvaluator& fitness
		) : RansacRigidBase(std::move(parameters), std::move(kinematic), fitness) {}
		~RansacRigidRandom() override = default;
		
		//The method to generate hypothesis
	protected:
		void generatePoseHypoethesis(const FeatureMap& observation, const FeatureMap& model) override;
	};
}
