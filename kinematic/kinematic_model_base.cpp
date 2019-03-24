//
// Created by wei on 9/20/18.
//

#include "kinematic/kinematic_model_base.h"

poser::KinematicModelBase::KinematicModelBase(
	poser::MemoryContext context,
	poser::FeatureChannelType reference_vertex,
	poser::FeatureChannelType live_vertex,
	poser::FeatureChannelType reference_normal,
	poser::FeatureChannelType live_normal) : context_(context),
                                             reference_vertex_channel_(std::move(reference_vertex)),
                                             live_vertex_channel_(std::move(live_vertex)),
                                             reference_normal_channel_(std::move(reference_normal)),
                                             live_normal_channel_(std::move(live_normal))
{
	//Set to default channel type
	if(!reference_vertex_channel_.is_valid())
		reference_vertex_channel_ = CommonFeatureChannelKey::ReferenceVertex();
	if(!reference_normal_channel_.is_valid())
		reference_normal_channel_ = CommonFeatureChannelKey::ReferenceNormal();
	if(!live_vertex_channel_.is_valid())
		live_vertex_channel_ = CommonFeatureChannelKey::LiveVertex();
	if(!live_normal_channel_.is_valid())
		live_normal_channel_ = CommonFeatureChannelKey::LiveNormal();
	
	//Check the channel
	LOG_ASSERT(reference_vertex_channel_.is_dense()); LOG_ASSERT(reference_vertex_channel_.type_size_matched<float4>());
	LOG_ASSERT(reference_normal_channel_.is_dense()); LOG_ASSERT(reference_normal_channel_.type_size_matched<float4>());
	LOG_ASSERT(live_vertex_channel_.is_dense()); LOG_ASSERT(live_vertex_channel_.type_size_matched<float4>());
	LOG_ASSERT(live_normal_channel_.is_dense()); LOG_ASSERT(live_normal_channel_.type_size_matched<float4>());
}


void poser::KinematicModelBase::CheckGeometricModelAndAllocateAttribute(poser::FeatureMap &geometric_model) {
	//The cpu reference vertex must present
	LOG_ASSERT(geometric_model.ExistFeature(reference_vertex_channel_, MemoryContext::CpuMemory));
	
	//Check the customized version (might on gpu).
	//Allocate it if not exist
	if(!geometric_model.ExistFeature(reference_vertex_channel_, context_)) {
		const auto cpu_ref_vertex = geometric_model.GetTypedFeatureValueReadOnly<float4>(reference_vertex_channel_, MemoryContext::CpuMemory);
		LOG_ASSERT(geometric_model.InsertDenseFeature<float4>(reference_vertex_channel_, cpu_ref_vertex, context_));
	}
	
	//Allocate the live vertex, if not exist on the given context
	if(!geometric_model.ExistFeature(live_vertex_channel_, context_)) {
		geometric_model.AllocateDenseFeature(live_vertex_channel_, context_);
	}
	
	//Check the normal, if exist on cpu, move it to gpu
	if(geometric_model.ExistFeature(reference_normal_channel_, MemoryContext::CpuMemory)) {
		if(!geometric_model.ExistFeature(reference_normal_channel_, context_)) {
			const auto cpu_ref_normal = geometric_model.GetTypedFeatureValueReadOnly<float4>(reference_normal_channel_, MemoryContext::CpuMemory);
			geometric_model.InsertDenseFeature<float4>(reference_normal_channel_, cpu_ref_normal, context_);
		}
	}
	
	if(geometric_model.ExistFeature(reference_normal_channel_, context_)) {
		//Allocate live normal
		if(!geometric_model.ExistFeature(live_normal_channel_, context_)) {
			const auto ref_normal = geometric_model.GetTypedFeatureValueReadOnly<float4>(reference_normal_channel_, context_);
			geometric_model.InsertDenseFeature<float4>(live_normal_channel_, ref_normal, context_);
		}
	} else {
		//There is no normal
		reference_normal_channel_ = FeatureChannelType();
		live_normal_channel_ = FeatureChannelType();
	}
}