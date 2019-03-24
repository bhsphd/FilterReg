//
// Created by wei on 10/16/18.
//

#include <glog/logging.h>
#include "cloudproc/cloud_rigid_transform.h"

poser::CloudRigidTransformer::CloudRigidTransformer(
	poser::MemoryContext context,
	poser::FeatureChannelType vertex_from,
	poser::FeatureChannelType vertex_to,
	poser::FeatureChannelType normal_from,
	poser::FeatureChannelType normal_to
) : context_(context),
    vertex_from_(std::move(vertex_from)),
    vertex_to_(std::move(vertex_to)),
    normal_from_(std::move(normal_from)),
    normal_to_(std::move(normal_to)),
    finalized_(false)
{
	//Process the default case
	if(!vertex_from_.is_valid())
		vertex_from_ = CommonFeatureChannelKey::ObservationVertexCamera();
	if(!vertex_to_.is_valid())
		vertex_to_ = CommonFeatureChannelKey::ObservationVertexWorld();
	if(!normal_from_.is_valid())
		normal_from_ = CommonFeatureChannelKey::ObservationNormalCamera();
	if(!normal_to_.is_valid())
		normal_to_ = CommonFeatureChannelKey::ObservationNormalWorld();
	
	//The transform is initialized to indentity
	rigid_transform_ = mat34::identity();
}

void poser::CloudRigidTransformer::CheckAndAllocate(poser::FeatureMap &cloud_map) {
	//The vertex input, compulsory
	LOG_ASSERT(cloud_map.ExistFeature(vertex_from_, context_));
	if(!cloud_map.ExistFeature(vertex_to_, context_)) {
		LOG_ASSERT(vertex_to_.is_dense());
		cloud_map.AllocateDenseFeature(vertex_to_, context_);
	}
	
	//The process for normal
	if(normal_from_.is_valid() && cloud_map.ExistFeature(normal_from_, context_)) {
		if(!cloud_map.ExistFeature(normal_to_, context_)) {
			LOG_ASSERT(normal_to_.is_dense());
			cloud_map.AllocateDenseFeature(normal_to_, context_);
		}
	} else {
		//LOG(WARNING) << "Only the vertex will be transformed";
		normal_from_ = FeatureChannelType();
		normal_to_ = FeatureChannelType();
	}
	
	//OK
	finalized_ = true;
}

void poser::CloudRigidTransformer::Process(poser::FeatureMap &cloud_map) {
	LOG_ASSERT(finalized_) << "Please invoke CheckAndAllocate before actual processing";
	
	if(context_ == MemoryContext::CpuMemory) {
		if(normal_from_.is_valid())
			processVertexNormalCpu(cloud_map);
		else
			processVertexCpu(cloud_map);
	} else {
		LOG(FATAL) << "Not implemented yet";
	}
}

//The actual processing on CPU
void poser::CloudRigidTransformer::processVertexCpu(poser::FeatureMap &cloud_map) {
	const auto vertex_from = cloud_map.GetTypedFeatureValueReadOnly<float4>(vertex_from_, context_);
	auto vertex_to = cloud_map.GetTypedFeatureValueReadWrite<float4>(vertex_to_, context_);
	LOG_ASSERT(vertex_from.Size() == vertex_to.Size());
	
	//The iteration
	for(auto i = 0; i < vertex_from.Size(); i++) {
		const auto vertex_from_i = vertex_from[i];
		auto& vertex_to_i = vertex_to[i];
		float3& vertex_to_i_float3 = *((float3*)(&vertex_to_i));
		vertex_to_i_float3 = rigid_transform_.rotation() * vertex_from_i + rigid_transform_.translation;
		vertex_to_i.w = vertex_from_i.w;
	}
}

void poser::CloudRigidTransformer::processVertexNormalCpu(poser::FeatureMap &cloud_map) {
	const auto vertex_from = cloud_map.GetTypedFeatureValueReadOnly<float4>(vertex_from_, context_);
	auto vertex_to = cloud_map.GetTypedFeatureValueReadWrite<float4>(vertex_to_, context_);
	const auto normal_from = cloud_map.GetTypedFeatureValueReadOnly<float4>(normal_from_, context_);
	auto normal_to = cloud_map.GetTypedFeatureValueReadWrite<float4>(normal_to_, context_);
	
	//The iteration
	for(auto i = 0; i < vertex_from.Size(); i++) {
		const auto vertex_from_i = vertex_from[i];
		auto& vertex_to_i = vertex_to[i];
		float3& vertex_to_i_float3 = *((float3*)(&vertex_to_i));
		const auto normal_from_i = normal_from[i];
		auto& normal_to_i = normal_to[i];
		float3& normal_to_i_float3 = *((float3*)(&normal_to_i));
		
		//Actual computation
		vertex_to_i_float3 = rigid_transform_.rotation() * vertex_from_i + rigid_transform_.translation;
		vertex_to_i.w = vertex_from_i.w;
		normal_to_i_float3 = rigid_transform_.rotation() * normal_from_i;
		normal_to_i.w = normal_from_i.w;
	}
}