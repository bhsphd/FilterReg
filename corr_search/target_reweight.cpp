//
// Created by wei on 11/29/18.
//

#include "corr_search/target_reweight.h"
#include "geometry_utils/vector_operations.hpp"

void poser::TargetReweight::densePoint2PointReweight(
	const poser::FeatureMap &model,
	const poser::KinematicModelBase &kinematic,
	poser::DenseGeometricTarget &target,
	const poser::TargetReweight::ReweightFunctor &functor
) {
	const auto model_v = model.GetTypedFeatureValueReadOnly<float4>(kinematic.LiveVertexChannel(), MemoryContext::CpuMemory);
	auto target_v = target.GetTargetVertexReadWrite();
	for(auto i = 0; i < target.GetTargetFlattenSize(); i++) {
		const auto& model_i = model_v[i];
		auto& target_i = target_v[i];
		float new_weight = functor(model_i, target_i);
		target_i.w = new_weight;
	}
}

void poser::TargetReweight::sparsePoint2PointReweight(
	const poser::FeatureMap &model,
	const poser::KinematicModelBase &kinematic,
	poser::SparseGeometricTarget &target,
	const poser::TargetReweight::ReweightFunctor &functor
) {
	const auto model_v = model.GetTypedFeatureValueReadOnly<float4>(kinematic.LiveVertexChannel(), MemoryContext::CpuMemory);
	const auto model_idx = target.GetTargetModelIndexReadOnly();
	auto target_v = target.GetTargetVertexReadWrite();
	for(auto i = 0; i < target.GetTargetFlattenSize(); i++) {
		const auto& model_i = model_v[model_idx[i]];
		auto& target_i = target_v[i];
		float new_weight = functor(model_i, target_i);
		target_i.w = new_weight;
	}
}


void poser::TargetReweight::point2pointReweight(
	const poser::FeatureMap &model,
	const poser::KinematicModelBase &kinematic,
	poser::GeometricTargetBase &target,
	const poser::TargetReweight::ReweightFunctor &functor
) {
	if(target.IsDenseTarget()) {
		densePoint2PointReweight(model, kinematic,
			static_cast<DenseGeometricTarget&>(target), functor);
	} else {
		sparsePoint2PointReweight(model, kinematic,
			static_cast<SparseGeometricTarget&>(target), functor);
	}
}

void poser::TargetReweight::BlackRangarajanReweightPt2Pt(
	const poser::FeatureMap &model,
	const poser::KinematicModelBase &kinematic,
	poser::GeometricTargetBase &target, float mu
) {
	auto reweight_functor = [&](const float4& model_v, const float4& target_v) -> float {
		//The original weight is zero
		if(target_v.w < 1e-6f)
			return 0.0f;
		
		//There exist match
		float weight = mu / (squared_norm_xyz(model_v - target_v) + mu);
		if(weight < 2e-6f) weight = 2e-6f;
		return weight * target_v.w;
	};
	
	//Do it
	point2pointReweight(model, kinematic, target, reweight_functor);
}

void poser::TargetReweight::HuberReweightPt2Pt(
	float residual_boundary,
	const poser::FeatureMap &model,
	const poser::KinematicModelBase &kinematic,
	poser::GeometricTargetBase &target
) {
	auto reweight_functor = [&](const float4& model_v, const float4& target_v) -> float {
		float4 diverge = model_v - target_v;
		float l1_div = std::max(std::abs(diverge.x), std::abs(diverge.y));
		l1_div = std::max(std::abs(diverge.z), l1_div);
		if(l1_div < residual_boundary) {
			return target_v.w;
		} else {
			return target_v.w * (residual_boundary / l1_div);
		}
	};
	
	//Do it
	point2pointReweight(model, kinematic, target, reweight_functor);
}

void poser::TargetReweight::BisquareReweightPt2Pt(
	float residual_boundary,
	const poser::FeatureMap &model,
	const poser::KinematicModelBase &kinematic,
	poser::GeometricTargetBase &target
) {
	auto reweight_functor = [&](const float4& model_v, const float4& target_v) -> float {
		float4 diverge = model_v - target_v;
		float l1_div = std::max(std::abs(diverge.x), std::abs(diverge.y));
		l1_div = std::max(std::abs(diverge.z), l1_div);
		if(l1_div < residual_boundary) {
			float ratio = l1_div / residual_boundary;
			float w_tmp = 1 - ratio * ratio;
			return target_v.w * w_tmp * w_tmp;
		} else {
			return 0.0f;
		}
	};
	
	//Do it
	point2pointReweight(model, kinematic, target, reweight_functor);
}

