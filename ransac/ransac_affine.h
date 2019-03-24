//
// Created by wei on 12/3/18.
//

#pragma once

#include "ransac/ransac_common.h"
#include "kinematic/affine/affine.h"

namespace poser {
	
	class RansacAffine : public RansacBase {
	public:
		//The constructor group
		RansacAffine(
			RansacParameters parameters,
			AffineKinematicModel kinematic,
			const FitnessEvaluator& fitness
		) : RansacBase(std::move(parameters), fitness),
		    kinematic_(std::move(kinematic))
		{ LOG_ASSERT(ransac_parameter_.n_point_per_sample >= 4); };
		~RansacAffine() override = default;
		
		//The internal buffer with pose hypothesis
	protected:
		TensorBlob transform_hypothesis_;
		void generateTransformHypoethesis(const FeatureMap& observation, const FeatureMap& model);
	public:
		void CheckAndAllocate(const FeatureMap& observation, const FeatureMap& model) override;
		
		//Evaluate the geometric model
	protected:
		AffineKinematicModel kinematic_;
		FeatureMap mutable_model_;
		void evaluateHypothesisAffine();
	public:
		void Compute(const FeatureMap& observation, const FeatureMap& model);
		Eigen::Matrix4f GetBestHypothesisAffineTransform() const;
	};
	
}
