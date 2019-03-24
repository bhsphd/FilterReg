//
// Created by wei on 9/20/18.
//

#include <flann/flann.hpp>
#include <gtest/gtest.h>
#include <glog/logging.h>

#include <vector_types.h>
#include <chrono>
#include <random>

#include "geometry_utils/kdtree_flann.h"
#include "geometry_utils/vector_operations.hpp"

class KDTreeTest : public ::testing::Test {
protected:
	void SetUp() override {
		//Random distribution
		std::uniform_real_distribution<float> uniform;
		std::default_random_engine re;
		
		//Do it
		test_data.resize(test_size);
		for(auto& vec : test_data) {
			vec.x = uniform(re);
			vec.y = uniform(re);
			vec.z = uniform(re);
			vec.w = uniform(re);
		}
		
		query_vec.resize(test_size);
		for(auto& vec : query_vec) {
			vec.x = uniform(re);
			vec.y = uniform(re);
			vec.z = uniform(re);
			vec.w = uniform(re);
		}
	}
	
	//The data for testing
	std::vector<float4> test_data;
	std::vector<float4> query_vec;
	static constexpr auto test_size = 4000;
};


TEST_F(KDTreeTest, FlannTest) {
	using namespace std::chrono;
	const auto knn_k = 1;
	
	//Build the kd tree index
	flann::Matrix<float> dataset((float*)test_data.data(), test_size, 3, sizeof(float) * 4);
	flann::Index<flann::L2<float>> index(dataset, flann::KDTreeIndexParams(4));
	
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	index.buildIndex();
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	auto delta_t_ms = duration_cast<milliseconds>(t2 - t1);
	//LOG(INFO) << "The time is ms is " << delta_t_ms.count();
	
	
	//The distance
	std::vector<int> queried_index; queried_index.resize(knn_k * test_size);
	std::vector<float> queried_dist; queried_dist.resize(knn_k * test_size);
	
	//Construct flann type
	flann::Matrix<float> query((float*) query_vec.data(), test_size, 3, sizeof(float) * 4);
	flann::Matrix<int> indices(queried_index.data(), query.rows, knn_k);
	flann::Matrix<float> distance(queried_dist.data(), query.rows, knn_k);
	
	t1 = high_resolution_clock::now();
	index.knnSearch(query, indices, distance, knn_k, flann::SearchParams());
	t2 = high_resolution_clock::now();
	delta_t_ms = duration_cast<milliseconds>(t2 - t1);
	//LOG(INFO) << "The time is ms is " << delta_t_ms.count();
}

TEST_F(KDTreeTest, BasicTest) {
	using namespace poser;
	KDTreeSingleNN kdtree;
	BlobView data_in((const char*)test_data.data(), test_data.size(), sizeof(float4), MemoryContext::CpuMemory);
	kdtree.ResetInputData(data_in);
	
	//Just search the data_in, the distance should be zero
	std::vector<int> result_index; result_index.resize(data_in.Size());
	std::vector<float> result_dist_square; result_dist_square.resize(data_in.Size());
	kdtree.SearchNN(data_in, result_index.data(), result_dist_square.data());
	
	//Do check
	for(auto i = 0; i < result_index.size(); i++) {
		EXPECT_EQ(result_index[i], i);
		EXPECT_NEAR(result_dist_square[i], 0.0f, 1e-5f);
	}
}

TEST_F(KDTreeTest, SingleRadiusTest) {
	using namespace poser;
	KDTreeSingleNN kdtree;
	BlobView data_in((const char*)test_data.data(), test_data.size(), sizeof(float4), MemoryContext::CpuMemory);
	kdtree.ResetInputData(data_in);
	
	//Just search the data_in, the distance should be zero
	std::vector<int> result_index; result_index.resize(data_in.Size());
	std::vector<float> result_dist_square; result_dist_square.resize(data_in.Size());
	float radius = 0.3f;
	kdtree.SearchRadius(data_in, radius, result_index.data(), result_dist_square.data());
	
	//Do check
	for(auto i = 0; i < result_index.size(); i++) {
		EXPECT_EQ(result_index[i], i);
		EXPECT_NEAR(result_dist_square[i], 0.0f, 1e-5f);
	}
}

TEST_F(KDTreeTest, RadiusTest) {
	using namespace poser;
	KDTreeKNN kdtree;
	BlobView data_in((const char*)test_data.data(), test_data.size(), sizeof(float4), MemoryContext::CpuMemory);
	kdtree.ResetInputData(data_in);
	
	//Search the query
	BlobView query((const char*)query_vec.data(), query_vec.size(), sizeof(float4), MemoryContext::CpuMemory);
	std::vector<std::vector<int>> result_index; result_index.resize(query.Size());
	std::vector<std::vector<float>> dist_square; dist_square.resize(query.Size());
	
	const float radius = 0.2f;
	kdtree.SearchRadius(query, radius, result_index, dist_square);
	for(auto i = 0; i < query.Size(); i++) {
		const auto& radius_nn_i = result_index[i];
		const auto& dist_square_nn_i = dist_square[i];
		const auto& query_i = query_vec[i];
		
		//Do it
		for(auto k = 0; k < radius_nn_i.size(); k++) {
			const auto nn_idx = radius_nn_i[k];
			const auto& input_i = test_data[nn_idx];
			auto dist_square_direct = squared_norm(input_i - query_i);
			//Note that the distance is squared
			EXPECT_NEAR(dist_square_direct, dist_square_nn_i[k], 1e-6f);
			EXPECT_LE(dist_square_direct, radius * radius);
		}
	}
}

TEST_F(KDTreeTest, KNNTest) {
	using namespace poser;
	KDTreeKNN kdtree;
	BlobView data_in((const char*)test_data.data(), test_data.size(), sizeof(float4), MemoryContext::CpuMemory);
	kdtree.ResetInputData(data_in);
	
	//Search the query: just the input data
	BlobView query((const char*)test_data.data(), test_data.size(), sizeof(float4), MemoryContext::CpuMemory);
	const auto knn = 4;
	std::vector<int> knn_idx; knn_idx.resize(query_vec.size() * knn);
	std::vector<float> knn_dist; knn_dist.resize(query_vec.size() * knn);
	BlobSlice knn_idx_slice((char*)knn_idx.data(), query.Size(), sizeof(int) * 4, MemoryContext::CpuMemory);
	BlobSlice knn_dist_slice((char*)knn_dist.data(), query.Size(), sizeof(float) * 4, MemoryContext::CpuMemory);
	kdtree.SearchKNN(query, knn, knn_idx_slice, knn_dist_slice);
	
	//Check it
	for(auto i = 0; i < query.Size(); i++) {
		const auto& query_i = test_data[i];
		const auto& knn_i = knn_idx_slice.ElemVectorAt<int>(i);
		const auto& knn_dist_i = knn_dist_slice.ElemVectorAt<float>(i);
		for(auto knn_k = 0; knn_k < knn_i.typed_size; knn_k++) {
			const auto& neighbour_k = test_data[knn_i[knn_k]];
			const auto dist_square_direct = squared_norm(neighbour_k - query_i);
			EXPECT_NEAR(dist_square_direct, knn_dist_i[knn_k], 1e-6f);
			
			//The nearest should be exactly zero
			if(knn_k == 0) {
				EXPECT_NEAR(dist_square_direct, 0.0, 1e-6f);
			}
		}
	}
}

TEST_F(KDTreeTest, RadiusKNNTest) {
	using namespace poser;
	KDTreeKNN kdtree;
	BlobView data_in((const char*)test_data.data(), test_data.size(), sizeof(float4), MemoryContext::CpuMemory);
	kdtree.ResetInputData(data_in);
	
	//Search the query: just the input data
	BlobView query((const char*)test_data.data(), test_data.size(), sizeof(float4), MemoryContext::CpuMemory);
	const auto knn = 4;
	std::vector<int> knn_idx; knn_idx.resize(query_vec.size() * knn);
	std::vector<float> knn_dist; knn_dist.resize(query_vec.size() * knn);
	BlobSlice knn_idx_slice((char*)knn_idx.data(), query.Size(), sizeof(int) * 4, MemoryContext::CpuMemory);
	BlobSlice knn_dist_slice((char*)knn_dist.data(), query.Size(), sizeof(float) * 4, MemoryContext::CpuMemory);
	
	//Do it
	const auto radius = 0.4f;
	kdtree.SearchRadiusKNN(query, radius, knn, knn_idx_slice, knn_dist_slice);
	
	//Check it
	for(auto i = 0; i < query.Size(); i++) {
		const auto& query_i = test_data[i];
		const auto& knn_i = knn_idx_slice.ElemVectorAt<int>(i);
		const auto& knn_dist_i = knn_dist_slice.ElemVectorAt<float>(i);
		for(auto knn_k = 0; knn_k < knn_i.typed_size; knn_k++) {
			const auto knn_k_query = knn_i[knn_k];
			const auto& neighbour_k = test_data[knn_k_query];
			const auto dist_square_direct = squared_norm(neighbour_k - query_i);
			EXPECT_NEAR(dist_square_direct, knn_dist_i[knn_k], 1e-6f);
			EXPECT_LE(dist_square_direct, radius * radius);
			
			//The nearest should be exactly zero
			if(knn_k == 0) {
				EXPECT_NEAR(dist_square_direct, 0.0, 1e-6f);
			}
		}
	}
}