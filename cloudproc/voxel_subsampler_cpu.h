//
// Created by wei on 10/16/18.
//

#pragma once

#include "cloudproc/subsampler_base.h"
#include "cloudproc/subsampler_common.h"

namespace poser {
	
	/* Susample the point cloud using voxel grid.
	 */
	class VoxelGridSubsamplerCPU : public SubsampleProcessorBase {
	public:
		explicit VoxelGridSubsamplerCPU(
			FeatureChannelType vertex = FeatureChannelType(),
			FeatureChannelType gather_index = FeatureChannelType(),
			SubsamplerCommonOption option = SubsamplerCommonOption());
		void SetVoxelGridLeafSize(const float3& leaf_size);
		void SetVoxelGridLeafSize(float leaf_size) { SetVoxelGridLeafSize(make_float3(leaf_size, leaf_size, leaf_size)); }
		void CheckAndAllocate(const FeatureMap& in_cloud_map, FeatureMap& out_cloud_map) override;
		
		//The processing method
	protected:
		using PointFilter = std::function<bool(int idx_in_input, const float4& vertex)>;
		void processInternal(const FeatureMap& in_cloud_map, FeatureMap& out_cloud_map, const PointFilter& filter);
	public:
		void Process(const FeatureMap& in_cloud_map, FeatureMap& out_cloud_map);
		
	private:
		//The option used to subsample the point cloud
		float3 inv_leaf_size_;
		
		//The map from voxel to element and index in this voxel
		struct VoxelInfo {
			float4 weighted_vertex; //x, y, z is weighted vertex, w is the weight; x/w is true vertex
			unsigned gather_index;
		};
		struct VoxelCoordHasher {
			std::size_t operator()(const int3& voxel) const {
				return std::hash<int>()(voxel.x)^std::hash<int>()(voxel.y)^std::hash<int>()(voxel.z);
			}
		};
		struct VoxelCoordComparator {
			bool operator()(const int3& lhs, const int3& rhs) const {
				return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
			}
		};
		unordered_map<int3, VoxelInfo, VoxelCoordHasher, VoxelCoordComparator> voxel_map_;
	};
}
