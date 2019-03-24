//
// Created by wei on 9/21/18.
//

#include "corr_search/target_computer_base.h"
#include "target_computer_base.h"


poser::SingleFeatureTargetComputerBase::SingleFeatureTargetComputerBase(
	poser::MemoryContext context,
	poser::FeatureChannelType observation_world_vertex,
	poser::FeatureChannelType model_feature,
	poser::FeatureChannelType observation_feature
) : context_(context),
    observation_world_vertex_(std::move(observation_world_vertex)),
    model_feature_channel_(std::move(model_feature)),
    observation_feature_channel_(std::move(observation_feature)),
    model_visibility_score_(),
    finalized_(false)
{
	if(!observation_feature_channel_.is_valid())
		observation_feature_channel_ = observation_world_vertex_;
}

/* The method that build that check the input and build internal index
 */
void poser::SingleFeatureTargetComputerBase::CheckAndAllocateTarget(
	const poser::FeatureMap &observation,
	const poser::FeatureMap &model,
	poser::GeometricTargetBase &target
) {
	//Check it
	checkModelAndObservationBasic(observation, model);
	
	//Depends on type
	if(target.IsDenseTarget()) {
		auto& dense_target = static_cast<DenseGeometricTarget&>(target);
		checkAndAllocateDenseTarget(observation, model, dense_target);
	} else {
		auto& sparse_target = static_cast<SparseGeometricTarget&>(target);
		checkAndAllocateSparseTarget(observation, model, sparse_target);
	}
	
	//Mark the flag
	finalized_ = true;
}

void poser::SingleFeatureTargetComputerBase::checkModelAndObservationBasic(
	const FeatureMap& observation,
	const FeatureMap& model
) {
	//Check the size and feature
	LOG_ASSERT(observation_feature_channel_.valid_type_byte() == model_feature_channel_.valid_type_byte());
	LOG_ASSERT(observation.ExistFeature(observation_feature_channel_, context_));
	LOG_ASSERT(observation.ExistFeature(observation_world_vertex_, context_));
	LOG_ASSERT(model.ExistFeature(model_feature_channel_, context_));
	
	//Check the existence of visibility score
	if(model_visibility_score_.is_valid()) {
		if(!model.ExistFeature(model_visibility_score_, context_)) {
			//Make it invalid here
			LOG(WARNING) << "Cannot find the weight for model points";
			model_visibility_score_ = FeatureChannelType();
		}
	}
}

//The specific procsssing method
void poser::SingleFeatureTargetComputerBase::checkAndAllocateDenseTarget(
	const poser::FeatureMap &observation,
	const poser::FeatureMap &model,
	poser::DenseGeometricTarget &target
) {
	LOG_ASSERT(model_feature_channel_.is_dense());
	target.AllocateTargetForModel(model, context_);
}

void poser::SingleFeatureTargetComputerBase::checkAndAllocateSparseTarget(
	const poser::FeatureMap &observation,
	const poser::FeatureMap &model,
	poser::SparseGeometricTarget &target
) {
	LOG_ASSERT(model_feature_channel_.is_sparse());
	target.AllocateForSparseFeature(model, model_feature_channel_);
}


