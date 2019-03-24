//
// Created by wei on 11/29/18.
//

#pragma once

#include "common/feature_map.h"
#include "common/geometric_target_interface.h"
#include "kinematic/kinematic_model_base.h"

namespace poser {
	
	
	class TargetReweight {
		/* The point2point reweight functor and the general processing method.
		 * It is actually an approximate form as each channel should be reweighted.
		 */
	private:
		using ReweightFunctor = std::function<float(const float4&, const float4&)>;
		static void densePoint2PointReweight(
			const FeatureMap& model,
			const KinematicModelBase& kinematic,
			DenseGeometricTarget& target, const ReweightFunctor& functor);
		static void sparsePoint2PointReweight(
			const FeatureMap& model,
			const KinematicModelBase& kinematic,
			SparseGeometricTarget& target, const ReweightFunctor& functor);
		static void point2pointReweight(
			const FeatureMap& model,
			const KinematicModelBase& kinematic,
			GeometricTargetBase& target, const ReweightFunctor& functor);
	public:
		static void BlackRangarajanReweightPt2Pt(
			const FeatureMap& model,
			const KinematicModelBase& kinematic,
			GeometricTargetBase& target, float mu = 1.0f);
		static void HuberReweightPt2Pt(
			float residual_boundary,
			const FeatureMap& model,
			const KinematicModelBase& kinematic,
			GeometricTargetBase& target);
		static void BisquareReweightPt2Pt(
			float residual_boundary,
			const FeatureMap& model,
			const KinematicModelBase& kinematic,
			GeometricTargetBase& target);
	};
}
