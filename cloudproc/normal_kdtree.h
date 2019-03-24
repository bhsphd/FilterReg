//
// Created by wei on 12/5/18.
//

#pragma once

#include "common/feature_map.h"
#include "geometry_utils/kdtree_flann.h"

namespace poser {
	
	/* Compute normal directly use nearest neighbour from kdtree.
	 * The direction of normal vector is determined with +/- unless
	 * a camera location or a direction is specified.
	 * The unoriented normal is enough for point-to-plane ICP.
	 */
	class NormalEstimationKDTree {
	private:
		//The channel for input vertex
		FeatureChannelType vertex_channel_;
		FeatureChannelType normal_channel_;
		
		//The buffer for kdtree search and normal estimation
		KDTreeKNN kdtree_;
		TensorBlob knn_index_;
		TensorBlob knn_dist_;
		int max_knn_ = 30;
		float search_radius_ = 1e6f;
	public:
		explicit NormalEstimationKDTree(
			FeatureChannelType vertex = FeatureChannelType(),
			FeatureChannelType normal = FeatureChannelType());
		
		//Update of parameter
		void SetMaximumKNN(int max_knn) { max_knn_ = max_knn; }
		void SetSearchRadius(float radius) { search_radius_ = radius; }
		
		void CheckAndAllocate(FeatureMap& cloud_feature_map);
		void Process(FeatureMap& cloud_feature_map);
	
	protected:
		static float3 ComputeNormal(
			const TensorView<float4>& cloud,
			const float4& vertex,
			const BlobView::ElemVector<int>& knn);
	};
}
