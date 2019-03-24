//
// Created by wei on 10/9/18.
//

#include "cloudproc/cloud_visibility_projection.h"

poser::CloudVisibilityProjection::CloudVisibilityProjection(
	poser::Intrinsic camera_intrinsic,
	poser::FeatureChannelType vertex_channel,
	poser::FeatureChannelType visibility_score_channel,
	int subsample_rate,
	float tolerance_depth,
	int img_rows, int img_cols
) : raw_intrinsic_(camera_intrinsic),
    vertex_channel_(std::move(vertex_channel)),
    visibility_score_channel_(std::move(visibility_score_channel)),
    image_rows_(img_rows), image_cols_(img_cols),
    subsample_rate_(subsample_rate), tolerance_depth_diff_(tolerance_depth),
    tolerance_viewangle_cos_(0.6f) //Not used
{
	//Check the compatibility of rows and subsample rate
	const unsigned subsampled_rows = image_rows_ / subsample_rate_;
	const unsigned subsampled_cols = image_cols_ / subsample_rate_;
	subsampled_z_map_.Reset<float>(subsampled_rows, subsampled_cols, MemoryContext::CpuMemory);
}

poser::CloudVisibilityProjection::CloudVisibilityProjection(
	poser::Intrinsic camera_intrinsic,
	poser::FeatureChannelType vertex_channel,
	poser::FeatureChannelType normal_channel,
	poser::FeatureChannelType visibility_score_channel,
	int subsample_rate,
	float tolerance_depth,
	float tolerance_viewangle_cos,
	int img_rows, int img_cols
) : raw_intrinsic_(camera_intrinsic),
    vertex_channel_(std::move(vertex_channel)),
    normal_channel_(std::move(normal_channel)),
    visibility_score_channel_(std::move(visibility_score_channel)),
    image_rows_(img_rows), image_cols_(img_cols),
    subsample_rate_(subsample_rate),
    tolerance_depth_diff_(tolerance_depth), tolerance_viewangle_cos_(tolerance_viewangle_cos)
{
	//Check the compatibility of rows and subsample rate
	const unsigned subsampled_rows = image_rows_ / subsample_rate_;
	const unsigned subsampled_cols = image_cols_ / subsample_rate_;
	subsampled_z_map_.Reset<float>(subsampled_rows, subsampled_cols, MemoryContext::CpuMemory);
}

void poser::CloudVisibilityProjection::SetCamera2World(const poser::mat34 &camera2world) {
	world2camera_ = camera2world.inverse_as_rigid();
}

void poser::CloudVisibilityProjection::SetCamera2World(const Eigen::Isometry3f& camera2world) {
	mat34 camera2world_mat(camera2world);
	SetCamera2World(camera2world_mat);
}

void poser::CloudVisibilityProjection::CheckAndAllocate(poser::FeatureMap &feature_map) {
	LOG_ASSERT(feature_map.ExistFeature(vertex_channel_, MemoryContext::CpuMemory));
	if(!feature_map.ExistFeature(visibility_score_channel_, MemoryContext::CpuMemory))
		feature_map.AllocateDenseFeature(visibility_score_channel_, MemoryContext::CpuMemory);
	
	//Check the normal
	if(normal_channel_.is_valid())
		LOG_ASSERT(feature_map.ExistFeature(normal_channel_, MemoryContext::CpuMemory));
}

void poser::CloudVisibilityProjection::PredictVisibility(poser::FeatureMap &feature_map) {
	if(normal_channel_.is_valid())
		predictVisibilityVertexNormal(feature_map);
	else
		predictVisibilityVertexOnly(feature_map);
}


void poser::CloudVisibilityProjection::ReweightTarget(const FeatureMap& geometric_model, poser::GeometricTargetBase &target) {
	//Check the context
	LOG_ASSERT(target.IsCpuTarget());
	LOG_ASSERT(target.IsDenseTarget());
	
	const auto visibility_score = geometric_model.GetTypedFeatureValueReadOnly<float>(visibility_score_channel_, MemoryContext::CpuMemory);
	auto target_vertex = target.GetTargetVertexReadWrite();
	LOG_ASSERT(target_vertex.Size() == visibility_score.Size());
	for(auto i = 0; i < target_vertex.Size(); i++) {
		target_vertex[i].w *= visibility_score[i];
	}
}

void poser::CloudVisibilityProjection::buildDepthMap(const poser::FeatureMap &feature_map) {
	//Get the vertex
	const auto world_vertex = feature_map.GetTypedFeatureValueReadOnly<float4>(vertex_channel_, MemoryContext::CpuMemory);
	
	//Reset the z_map
	const unsigned subsampled_rows = image_rows_ / subsample_rate_;
	const unsigned subsampled_cols = image_cols_ / subsample_rate_;
	auto z_map = subsampled_z_map_.GetTypedTensorReadWrite<float>();
	for(auto r_idx = 0; r_idx < subsampled_rows; r_idx++)
		for (auto c_idx = 0; c_idx < subsampled_cols; c_idx++)
			z_map(r_idx, c_idx) = -1.0f; //An invalid depth
	
	//Project the vertex into the z_map
	for(auto i = 0; i < world_vertex.Size(); i++) {
		const float4 world_vertex_i = world_vertex[i];
		const float3 camera_vertex_i = world2camera_.rotation() * world_vertex_i + world2camera_.translation;
		
		//Into image plane
		const int x = (((camera_vertex_i.x / (camera_vertex_i.z + 1e-10f)) * raw_intrinsic_.focal_x) + raw_intrinsic_.principal_x);
		const int y = (((camera_vertex_i.y / (camera_vertex_i.z + 1e-10f)) * raw_intrinsic_.focal_y) + raw_intrinsic_.principal_y);
		const int subsampled_x = x / subsample_rate_;
		const int subsampled_y = y / subsample_rate_;
		
		//Update the z_value
		if(subsampled_y >= 0 && subsampled_y < z_map.Rows() && subsampled_x >= 0 && subsampled_x < z_map.Cols()) {
			const float curr_z_value = camera_vertex_i.z;
			if(curr_z_value > 0.0f) {
				float& map_z_value = z_map(subsampled_y, subsampled_x);
				if(map_z_value < 0.0f || curr_z_value < map_z_value)
					map_z_value = curr_z_value;
			}
		}
	}
}

void poser::CloudVisibilityProjection::predictVisibilityVertexOnly(poser::FeatureMap &feature_map) {
	//Get the vertex
	const auto world_vertex = feature_map.GetTypedFeatureValueReadOnly<float4>(vertex_channel_, MemoryContext::CpuMemory);
	auto visibility_score = feature_map.GetTypedFeatureValueReadWrite<float>(visibility_score_channel_, MemoryContext::CpuMemory);
	LOG_ASSERT(world_vertex.Size() == visibility_score.Size());
	
	//Reset and build the z_map
	buildDepthMap(feature_map);
	const auto z_map = subsampled_z_map_.GetTypedTensorReadOnly<float>();
	
	//The second loop, predict the visibility score
	for(auto i = 0; i < world_vertex.Size(); i++) {
		const float4 world_vertex_i = world_vertex[i];
		const float3 camera_vertex_i = world2camera_.rotation() * world_vertex_i + world2camera_.translation;
		
		//Into image plane
		const int x = (((camera_vertex_i.x / (camera_vertex_i.z + 1e-10f)) * raw_intrinsic_.focal_x) + raw_intrinsic_.principal_x);
		const int y = (((camera_vertex_i.y / (camera_vertex_i.z + 1e-10f)) * raw_intrinsic_.focal_y) + raw_intrinsic_.principal_y);
		const int subsampled_x = x / subsample_rate_;
		const int subsampled_y = y / subsample_rate_;
		
		//The final result
		float visibility_i = 0.0f;
		const float inv_sigma_square = 9.0f / (tolerance_depth_diff_ * tolerance_depth_diff_);
		
		//Update the z_value
		if(subsampled_y >= 0 && subsampled_y < z_map.Rows() && subsampled_x >= 0 && subsampled_x < z_map.Cols()) {
			const float curr_z_value = camera_vertex_i.z;
			const float map_z_value = z_map(subsampled_y, subsampled_x);
			const float diff_z = curr_z_value - map_z_value; //Should be greater than 0
			if(curr_z_value > 0.0f && diff_z < tolerance_depth_diff_)
				visibility_i = expf(-0.5 * diff_z * diff_z * inv_sigma_square);
		}
		
		//Write to result
		visibility_score[i] = visibility_i;
	}
}

void poser::CloudVisibilityProjection::predictVisibilityVertexNormal(poser::FeatureMap &feature_map) {
	//Get the vertex
	const auto world_vertex = feature_map.GetTypedFeatureValueReadOnly<float4>(vertex_channel_, MemoryContext::CpuMemory);
	const auto world_normal = feature_map.GetTypedFeatureValueReadOnly<float4>(normal_channel_, MemoryContext::CpuMemory);
	auto visibility_score = feature_map.GetTypedFeatureValueReadWrite<float>(visibility_score_channel_, MemoryContext::CpuMemory);
	LOG_ASSERT(world_vertex.Size() == visibility_score.Size());
	
	//Reset and build the z_map
	buildDepthMap(feature_map);
	const auto z_map = subsampled_z_map_.GetTypedTensorReadOnly<float>();
	
	//The second loop, predict the visibility score
	for(auto i = 0; i < world_vertex.Size(); i++) {
		const float4 world_vertex_i = world_vertex[i];
		const float4 world_normal_i = world_normal[i];
		const float3 camera_vertex_i = world2camera_.rotation() * world_vertex_i + world2camera_.translation;
		const float3 camera_normal_i = world2camera_.rotation() * world_normal_i;
		
		//Check the view angle
		auto viewangle_cos = - dot(normalized(camera_vertex_i), camera_normal_i);
		if(viewangle_cos < tolerance_viewangle_cos_) {
			visibility_score[i] = 0.0f;
			continue;
		}
		
		//Into image plane
		const int x = (((camera_vertex_i.x / (camera_vertex_i.z + 1e-10f)) * raw_intrinsic_.focal_x) + raw_intrinsic_.principal_x);
		const int y = (((camera_vertex_i.y / (camera_vertex_i.z + 1e-10f)) * raw_intrinsic_.focal_y) + raw_intrinsic_.principal_y);
		const int subsampled_x = x / subsample_rate_;
		const int subsampled_y = y / subsample_rate_;
		
		//The final result
		float visibility_i = 0.0f;
		const float inv_sigma_square = 9.0f / (tolerance_depth_diff_ * tolerance_depth_diff_);
		
		//Update the z_value
		if(subsampled_y >= 0 && subsampled_y < z_map.Rows() && subsampled_x >= 0 && subsampled_x < z_map.Cols()) {
			const float curr_z_value = camera_vertex_i.z;
			const float map_z_value = z_map(subsampled_y, subsampled_x);
			const float diff_z = curr_z_value - map_z_value; //Should be greater than 0
			if(curr_z_value > 0.0f && diff_z < tolerance_depth_diff_)
				visibility_i = expf(-0.5 * diff_z * diff_z * inv_sigma_square);
		}
		
		//Write to result
		visibility_score[i] = visibility_i;
	}
}