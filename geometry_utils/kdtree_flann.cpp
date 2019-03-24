//
// Created by wei on 9/20/18.
//

#include "geometry_utils/kdtree_flann.h"


//The implementation of base class
void poser::KDTreeBase::buildIndexInternal(const flann::IndexParams &index_params) {
	//Check the data
	LOG_ASSERT(flann_data_.TypeByte() % sizeof(float) == 0);
	LOG_ASSERT(flann_data_.ValidTypeByte() % sizeof(float) == 0);
	const auto valid_float_channels = flann_data_.ValidTypeByte() / sizeof(float);
	
	//Build index
	flann::Matrix<float> data((float*)flann_data_.RawPtr(), flann_data_.TensorFlattenSize(), valid_float_channels, flann_data_.TypeByte());
	kdtree_index_.reset(new flann::Index<flann::L2<float>>(data, index_params));
	kdtree_index_->buildIndex();
}

void poser::KDTreeBase::ResetInputData(const poser::BlobView &data_in) {
	LOG_ASSERT(data_in.IsCpuTensor()) << "KDTree is only implemented on CPU";
	flann_data_.Reset(data_in.DimensionalSize(), data_in.TypeByte(), MemoryContext::CpuMemory, data_in.ValidTypeByte());
	BlobCopyNoSync(data_in, flann_data_.GetTensorReadWrite());
	buildIndex();
}

void poser::KDTreeBase::ResetInputData(const poser::TensorBlob &data_in) {
	data_in.CloneTo(flann_data_);
	buildIndex();
}

//The search interface for 1-NN kdtree
void poser::KDTreeSingleNN::SearchNN(const poser::BlobView &query_in, int *result_idx, float *dist_square) const {
	//Construct the result
	flann::Matrix<int> result_index(result_idx, query_in.Size(), 1);
	flann::Matrix<float> result_dist(dist_square, query_in.Size(), 1);
	
	//Construct the query for flann
	LOG_ASSERT(query_in.ValidTypeByte() == flann_data_.ValidTypeByte());
	const auto valid_channel_size = query_in.ValidTypeByte() / sizeof(float);
	flann::Matrix<float> query((float*)query_in.RawPtr(), query_in.FlattenSize(), valid_channel_size, query_in.TypeByte());
	
	//Search it
	kdtree_index_->knnSearch(query, result_index, result_dist, 1, flann::SearchParams(10, 0.0));
}

void poser::KDTreeSingleNN::SearchRadius(
	const poser::BlobView &query_in, 
	float radius, 
	int *result_idx, float *dist_square
) const {
	//Construct the result
	flann::Matrix<int> result_index(result_idx, query_in.Size(), 1);
	flann::Matrix<float> result_dist(dist_square, query_in.Size(), 1);
	
	//Construct the query for flann
	LOG_ASSERT(query_in.ValidTypeByte() == flann_data_.ValidTypeByte());
	const auto valid_channel_size = query_in.ValidTypeByte() / sizeof(float);
	flann::Matrix<float> query((float*)query_in.RawPtr(), query_in.FlattenSize(), valid_channel_size, query_in.TypeByte());
	
	//The search method
	auto search_params = flann::SearchParams(10, 0.0f);
	search_params.max_neighbors = 1;
	kdtree_index_->radiusSearch(query, result_index, result_dist, radius * radius, search_params);
}


//The search interface for k-NN kdtree
void poser::KDTreeKNN::searchKNNInternal(
	const poser::BlobView &query_in,
	poser::BlobSlice result_index, poser::BlobSlice dist_square, int knn_k,
	const SearchFunctor& search_functor
) const {
	//Check the size of result
	LOG_ASSERT(result_index.ValidTypeByte() == knn_k * sizeof(int));
	LOG_ASSERT(dist_square.ValidTypeByte() == knn_k * sizeof(float));
	LOG_ASSERT(query_in.ValidTypeByte() == flann_data_.ValidTypeByte());
	
	//Create the query
	const auto valid_channel_size = query_in.ValidTypeByte() / sizeof(float);
	flann::Matrix<float> query((float*)query_in.RawPtr(), query_in.FlattenSize(), valid_channel_size, query_in.TypeByte());
	flann::Matrix<int> result_idx((int*)result_index.RawPtr(), result_index.FlattenSize(), knn_k, result_index.TypeByte());
	flann::Matrix<float> result_dist((float*)dist_square.RawPtr(), dist_square.FlattenSize(), knn_k, dist_square.TypeByte());
	
	//Do it
	search_functor(query, result_idx, result_dist);
}


void poser::KDTreeKNN::SearchKNN(
	const poser::BlobView &query_in,
	int knn_k,
	poser::BlobSlice result_index,
	poser::BlobSlice dist_square
) const {
	//The functor
	auto functor = [&](
		const flann::Matrix<float>& query,
		flann::Matrix<int>& result_index,
		flann::Matrix<float>& result_dist
	) {
		kdtree_index_->knnSearch(query, result_index, result_dist, knn_k, flann::SearchParams(15, 0.0));
	};
	
	//Do it
	searchKNNInternal(query_in, result_index, dist_square, knn_k, functor);
}

void poser::KDTreeKNN::SearchRadiusKNN(
	const poser::BlobView &query_in,
	float radius, int max_knn_k,
	poser::BlobSlice result_index, poser::BlobSlice dist_square
) const {
	//The functor
	auto functor = [&](
		const flann::Matrix<float>& query,
		flann::Matrix<int>& result_index,
		flann::Matrix<float>& result_dist
	) {
		auto search_params = flann::SearchParams(15, 0.0f);
		search_params.max_neighbors = max_knn_k;
		kdtree_index_->radiusSearch(query, result_index, result_dist, radius * radius, search_params);
	};
	
	//Do it
	searchKNNInternal(query_in, result_index, dist_square, max_knn_k, functor);
}

void poser::KDTreeKNN::SearchRadius(
	const poser::BlobView &query_in,
	float radius,
	std::vector<std::vector<int>> &indices,
	std::vector<std::vector<float>> &dist_square
) const {
	//Construct the query for flann
	LOG_ASSERT(query_in.ValidTypeByte() == flann_data_.ValidTypeByte());
	const auto valid_channel_size = query_in.ValidTypeByte() / sizeof(float);
	flann::Matrix<float> query((float*)query_in.RawPtr(), query_in.FlattenSize(), valid_channel_size, query_in.TypeByte());
	
	//Note that the radius should be squared here
	kdtree_index_->radiusSearch(query, indices, dist_square, radius * radius, flann::SearchParams(10, 0.0));
}
