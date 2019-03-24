//
// Created by wei on 9/20/18.
//

#pragma once

#include "common/feature_map.h"
#include "common/blob_access.h"

#include <flann/flann.hpp>

namespace poser {
	
	class KDTreeBase {
	protected:
		//The index for kdtree
		std::unique_ptr<flann::Index<flann::L2<float>>> kdtree_index_;
		TensorBlob flann_data_;
		void buildIndexInternal(const flann::IndexParams& index_params);
		virtual void buildIndex() = 0;
	public:
		KDTreeBase() = default;
		virtual ~KDTreeBase() = default;
		
		//The method to reset the input data and build the index
		void ResetInputData(const BlobView& data_in);
		void ResetInputData(const TensorBlob& data_in);
	};
	
	/* The kdtree that only perform 1-NN search.
	 * It needs a lot of different search params and interfaces
	 */
	class KDTreeSingleNN : public KDTreeBase {
	protected:
		//The build index method
		void buildIndex() override { buildIndexInternal(flann::KDTreeSingleIndexParams(15, true)); }
	public:
		KDTreeSingleNN() = default;
		explicit KDTreeSingleNN(const BlobView& data_in) { ResetInputData(data_in); }
		explicit KDTreeSingleNN(const TensorBlob& data_in) { ResetInputData(data_in); }
		
		//The method for multiple query
		void SearchNN(const BlobView& query, int* result_index, float* dist_square) const;
		void SearchRadius(
			const BlobView& query,
			float radius,
			int* result_index, float* dist_square) const;
	};
	
	
	/* The kdtree that performs k-NN search
	 * Again, different parameters and interfaces
	 */
	class KDTreeKNN : public KDTreeBase {
	protected:
		//The method to build the index, multiple kdtree
		void buildIndex() override { buildIndexInternal(flann::KDTreeIndexParams()); }
		
		//The general search interface
		using SearchFunctor = std::function<void(
			const flann::Matrix<float>& query,
			flann::Matrix<int>& result_index,
			flann::Matrix<float>& result_dist)>;
		void searchKNNInternal(
			const BlobView& query,
			BlobSlice result_index,
			BlobSlice dist_square,
			int knn_k,
			const SearchFunctor & search_functor) const;
	public:
		KDTreeKNN() = default;
		explicit KDTreeKNN(const BlobView& data_in) { ResetInputData(data_in); }
		explicit KDTreeKNN(const TensorBlob& data_in) { ResetInputData(data_in); }
		
		//The search interface
		void SearchKNN(const BlobView& query, int knn_k, BlobSlice result_index, BlobSlice dist_square) const;
		void SearchRadiusKNN(
			const BlobView& query,
			float radius, int max_knn_k,
			BlobSlice result_index, BlobSlice dist_square) const;
		void SearchRadius(
			const BlobView& query,
			float radius,
			std::vector<std::vector<int>>& indices,
			std::vector<std::vector<float>>& dist_square) const;
	};
}
