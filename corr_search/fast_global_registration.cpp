//
// Created by wei on 11/28/18.
//

#include "corr_search/fast_global_registration.h"
#include "geometry_utils/vector_operations.hpp"
#include "geometry_utils/kdtree_flann.h"


poser::FastGlobalRegistration::FastGlobalRegistration(
	poser::FeatureChannelType observation_world_vertex,
	poser::FeatureChannelType model_live_vertex,
	poser::FeatureChannelType feature_channel
) : SingleFeatureTargetComputerBase(
	    MemoryContext::CpuMemory,
	    std::move(observation_world_vertex),
	    feature_channel,
	    feature_channel),
	model_live_vertex_(std::move(model_live_vertex))
{
	if(!model_live_vertex_.is_valid())
		model_live_vertex_ = CommonFeatureChannelKey::LiveVertex();
}

void poser::FastGlobalRegistration::CheckAndAllocateTarget(
	const poser::FeatureMap &observation,
	const poser::FeatureMap &model,
	poser::GeometricTargetBase &target
) {
	SingleFeatureTargetComputerBase::CheckAndAllocateTarget(observation, model, target);
	LOG_ASSERT(model.ExistFeature(model_live_vertex_, MemoryContext::CpuMemory));
	
	//Allocate the normalized value
	normalized_obs_vertex_.reserve(observation.GetDenseFeatureCapacity());
	normalized_model_vertex_.reserve(model.GetDenseFeatureCapacity());
	
	//Allocate for the reciprocity correspondence
	const auto capacity = std::max(model.GetDenseFeatureCapacity(), observation.GetDenseFeatureCapacity());
	reciprocity_pairs_0_.Reserve<unsigned>(capacity);
	reciprocity_pairs_1_.Reserve<unsigned>(capacity);
	feature_0_nn_.reserve(capacity);
	feature_1_nn_.reserve(capacity);
	distance_buffer_.reserve(capacity);
}

void poser::FastGlobalRegistration::checkAndAllocateSparseTarget(
	const poser::FeatureMap &observation,
	const poser::FeatureMap &model,
	poser::SparseGeometricTarget &target
) {
	const auto model_size = model.GetDenseFeatureCapacity();
	const auto allocate_size = model_size;
	target.AllocateTargetForModel(allocate_size);
}

void poser::FastGlobalRegistration::buildNormalizedCloud(
	const poser::FeatureMap &observation,
	const poser::FeatureMap &model
) {
	//Get the vertex
	const auto obs_vertex = observation.GetTypedFeatureValueReadOnly<float4>(observation_world_vertex_, MemoryContext::CpuMemory);
	const auto model_vertex = model.GetTypedFeatureValueReadOnly<float4>(model_live_vertex_, MemoryContext::CpuMemory);
	
	//shift the cloud
	auto shift_cloud = [&](const TensorView<float4>& in_cloud, vector<float4>& shifted_cloud) {
		float4 mean_point = make_float4(0, 0, 0, 0);
		const auto num_point = in_cloud.Size();
		shifted_cloud.resize(num_point);
		
		//Compute the mean
		for(auto i = 0; i < num_point; i++) {
			const auto& cloud_i = in_cloud[i];
			mean_point.x += cloud_i.x;
			mean_point.y += cloud_i.y;
			mean_point.z += cloud_i.z;
		}
		mean_point.x = mean_point.x / num_point;
		mean_point.y = mean_point.y / num_point;
		mean_point.z = mean_point.z / num_point;
		
		//Shift it
		for(auto i = 0; i < num_point; i++) {
			auto& cloud_i = shifted_cloud[i];
			const auto& in_cloud_i = in_cloud[i];
			cloud_i.x = in_cloud_i.x - mean_point.x;
			cloud_i.y = in_cloud_i.y - mean_point.y;
			cloud_i.z = in_cloud_i.z - mean_point.z;
			cloud_i.w = 1.0f;
		}
	};
	
	//Do it
	shift_cloud(obs_vertex, normalized_obs_vertex_);
	shift_cloud(model_vertex, normalized_model_vertex_);
	
	//Compute the scale
	float global_scale = 0.0f;
	auto update_scale = [&](const vector<float4>& cloud) -> void {
		for(const auto& cloud_i : cloud) {
			const auto scale_i = std::sqrt(squared_norm_xyz(cloud_i));
			if(scale_i > global_scale)
				global_scale = scale_i;
		}
	};
	update_scale(normalized_obs_vertex_);
	update_scale(normalized_model_vertex_);
	
	
	//Scale the cloud
	auto scale_cloud = [&](vector<float4>& cloud) -> void {
		for(auto i = 0; i < cloud.size(); i++) {
			auto& cloud_i = cloud[i];
			cloud_i.x /= global_scale;
			cloud_i.y /= global_scale;
			cloud_i.z /= global_scale;
			cloud_i.w = 1.0f;
		}
	};
	scale_cloud(normalized_model_vertex_);
	scale_cloud(normalized_obs_vertex_);
}

unsigned poser::FastGlobalRegistration::buildReciprocityCorrespondence(
	const poser::BlobView &feature_0, const poser::BlobView &feature_1,
	unsigned* corr_0, unsigned* corr_1
) {
	//Allocate the result
	LOG_ASSERT(feature_0.ValidTypeByte() == feature_1.ValidTypeByte());
	feature_0_nn_.resize(feature_0.Size());
	feature_1_nn_.resize(feature_1.Size());
	distance_buffer_.resize(std::max(feature_0.Size(), feature_1.Size()));
	KDTreeSingleNN feature_tree_0(feature_0);
	KDTreeSingleNN feature_tree_1(feature_1);
	feature_tree_1.SearchNN(feature_0, feature_0_nn_.data(), distance_buffer_.data());
	feature_tree_0.SearchNN(feature_1, feature_1_nn_.data(), distance_buffer_.data());
	
	//Cross check the result
	unsigned offset = 0;
	for(auto i = 0; i < feature_0.Size(); i++) {
		const auto nn_in_feature_1 = feature_0_nn_[i];
		if(feature_1_nn_[nn_in_feature_1] != i)
			feature_0_nn_[i] = -1;
		else {
			corr_0[offset] = i;
			corr_1[offset] = nn_in_feature_1;
			offset++;
		}
	}
	return offset;
}

int poser::FastGlobalRegistration::buildTupleTestedCorrespondence(
	unsigned* in_corr_0, unsigned* in_corr_1, unsigned num_in_corr,
	const float4 *point_0, const float4 *point_1,
	unsigned* updated_corr_0, unsigned* updated_corr_1
) {
	//Check the data
	LOG_ASSERT(num_in_corr > 3);
	LOG_ASSERT(num_in_corr > 3);
	
	//Tuple constraint
	std::srand((unsigned)std::time(0));
	int rand0, rand1, rand2;
	int idi0, idi1, idi2, idj0, idj1, idj2;
	int number_of_trial = num_in_corr * 100;
	const float scale = 0.8f;
	
	//Do it
	int offset = 0;
	for(auto i = 0; i < number_of_trial; i++) {
		rand0 = rand() % num_in_corr;
		rand1 = rand() % num_in_corr;
		rand2 = rand() % num_in_corr;
		idi0 = in_corr_0[rand0];
		idj0 = in_corr_1[rand0];
		idi1 = in_corr_0[rand1];
		idj1 = in_corr_1[rand1];
		idi2 = in_corr_0[rand2];
		idj2 = in_corr_1[rand2];
		
		// collect 3 points from i-th fragment
		auto pti0 = point_0[idi0];
		auto pti1 = point_0[idi1];
		auto pti2 = point_0[idi2];
		float li0 = std::sqrt(squared_norm_xyz(pti0 - pti1));
		float li1 = std::sqrt(squared_norm_xyz(pti1 - pti2));
		float li2 = std::sqrt(squared_norm_xyz(pti2 - pti0));
		
		// collect 3 points from i-th fragment
		auto ptj0 = point_1[idi0];
		auto ptj1 = point_1[idi1];
		auto ptj2 = point_1[idi2];
		float lj0 = std::sqrt(squared_norm_xyz(ptj0 - ptj1));
		float lj1 = std::sqrt(squared_norm_xyz(ptj1 - ptj2));
		float lj2 = std::sqrt(squared_norm_xyz(ptj2 - ptj0));
		
		if ((li0 * scale < lj0) && (lj0 < li0 / scale) &&
		    (li1 * scale < lj1) && (lj1 < li1 / scale) &&
		    (li2 * scale < lj2) && (lj2 < li2 / scale)) {
			updated_corr_0[offset + 0] = idi0;
			updated_corr_1[offset + 0] = idj0;
			updated_corr_0[offset + 1] = idi1;
			updated_corr_1[offset + 1] = idj1;
			updated_corr_0[offset + 2] = idi2;
			updated_corr_1[offset + 2] = idj2;
			offset += 3;
		}
		
		//Break if too much
		if(offset + 3 > num_in_corr)
			break;
	}
	
	//The size of correspondence
	return offset;
}

void poser::FastGlobalRegistration::ComputeTarget(
	const poser::FeatureMap &observation,
	const poser::FeatureMap &model,
	poser::SparseGeometricTarget &target
) {
	//Get the feature
	const auto obs_feature = observation.GetFeatureValueReadOnly(observation_feature_channel_, MemoryContext::CpuMemory);
	const auto obs_vertex = observation.GetTypedFeatureValueReadOnly<float4>(observation_world_vertex_, MemoryContext::CpuMemory);
	const auto model_feature = model.GetFeatureValueReadOnly(model_feature_channel_, MemoryContext::CpuMemory);
	
	//Do it
	buildNormalizedCloud(observation, model);
	auto num_corr = buildReciprocityCorrespondence(
		model_feature, obs_feature,
		(unsigned*)reciprocity_pairs_0_.RawPtr(), (unsigned*)reciprocity_pairs_1_.RawPtr());
	reciprocity_pairs_0_.ResizeOrException(num_corr);
	reciprocity_pairs_1_.ResizeOrException(num_corr);
	
	//Get the final one
	auto model_correspondence_idx = target.GetTargetModelIndexReadWrite();
	auto obs_correspondence_idx = target.GetObservationIndexReadWrite();
	num_corr = buildTupleTestedCorrespondence(
		(unsigned*) reciprocity_pairs_0_.RawPtr(), (unsigned*) reciprocity_pairs_1_.RawPtr(), reciprocity_pairs_0_.TensorFlattenSize(),
		normalized_model_vertex_.data(), normalized_obs_vertex_.data(),
		model_correspondence_idx.RawPtr(), obs_correspondence_idx.RawPtr());
	target.ResizeSparseTarget(num_corr);
	
	
	//Write to result
	auto target_vertex = target.GetTargetVertexReadWrite();
	memset(target_vertex.RawPtr(), 0, sizeof(float4) * target_vertex.Size());
	for(auto i = 0; i < num_corr; i++) {
		const auto obs_idx = obs_correspondence_idx[i];
		auto& target_i = target_vertex[i];
		target_i = obs_vertex[obs_idx];
		target_i.w = 1.0f;
	}
}
