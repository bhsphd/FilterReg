//
// Created by wei on 9/11/18.
//

#include "common/tensor_utils.h"
#include "common/feature_channel_type.h"
#include "common/tensor_blob.h"
#include "common/feature_map.h"

#include <vector_functions.h>
#include <gtest/gtest.h>

TEST(SeralizeTest, TensorUtilTest) {
	using namespace poser;
	MemoryContext context = MemoryContext::GpuMemory;
	TensorDim tensor_dim{480, 640};
	
	//As a array
	std::vector<MemoryContext> context_arr;
	std::vector<TensorDim> dim_arr;
	const auto test_size = 100;
	for(auto i = 0; i < test_size; i++) {
		context_arr.emplace_back(context);
		dim_arr.emplace_back(tensor_dim);
	}
	
	//Save it
	json j;
	j["context_vec"] = context_arr;
	j["dim_vec"] = dim_arr;
	
	//To and load from string
	std::string j_str = j.dump();
	std::stringstream ss(j_str);
	json load_j;
	ss >> load_j;
	
	//Check the element in load node
	auto load_context = load_j["context_vec"].get<std::vector<MemoryContext>>();
	auto load_dim = load_j["dim_vec"].get<std::vector<TensorDim>>();
	EXPECT_EQ(load_dim.size(), test_size);
	EXPECT_EQ(load_context.size(), test_size);
	for(auto i = 0; i < test_size; i++) {
		EXPECT_TRUE(load_context[i] == MemoryContext::GpuMemory);
		EXPECT_EQ(load_dim[i].rows(), 480);
		EXPECT_EQ(load_dim[i].cols(), 640);
	}
}

TEST(SeralizeTest, FeatureChannelTypeMapTest) {
	using namespace poser;
	
	//First test simple type
	{
		auto dense_depth = CommonFeatureChannelKey::ObservationVertexCamera();
		
		//Save it
		json j;
		j["dense_depth_type"] = dense_depth;
		
		//To and load from string
		std::string j_str = j.dump();
		std::stringstream ss(j_str);
		json load_j;
		ss >> load_j;
		
		//Load it
		auto load_dense_depth = load_j["dense_depth_type"].get<FeatureChannelType>();
		EXPECT_EQ(load_dense_depth, dense_depth);
		EXPECT_TRUE(load_dense_depth.is_valid());
		EXPECT_FALSE(load_dense_depth.is_sparse());
		EXPECT_EQ(load_dense_depth.type_byte(), dense_depth.type_byte());
		EXPECT_EQ(dense_depth.valid_type_byte(), load_dense_depth.valid_type_byte());
	}
	
	//The multimap version
	{
		FeatureMultiMap<int> map;
		auto dense_depth = CommonFeatureChannelKey::RawDepthImage();
		auto dense_rgb = CommonFeatureChannelKey::RawRGBImage();
		map.emplace(dense_depth.get_name_key(), 1);
		map.emplace(dense_depth.get_name_key(), 2);
		map.emplace(dense_rgb.get_name_key(), 3);
		
		//Save it
		json j;
		j["map"] = map;
		
		//To and load from string
		std::string j_str = j.dump();
		std::stringstream ss(j_str);
		json load_j;
		ss >> load_j;
		
		//load the map
		auto load_map = load_j["map"].get<FeatureMultiMap<int>>();
		//load_j["map"].get<std::unordered_multimap<int, int>>();
		//FeatureChannelTypeMultiMap<int> load_map;
		EXPECT_EQ(load_map.size(), 3);
		
		//Check it
		auto iter = load_map.find(dense_rgb.get_name_key());
		EXPECT_FALSE(iter == load_map.end());
		EXPECT_EQ(iter->second, 3);
		
		//Check multi key
		EXPECT_EQ(load_map.count(dense_depth.get_name_key()), 2);
		auto equal_range = load_map.equal_range(dense_depth.get_name_key());
		for(iter = equal_range.first; iter != equal_range.second; iter++) {
			EXPECT_TRUE(iter->second == 1 || iter->second == 2) << iter->second;
		}
	}
}


TEST(SeralizeTest, BlobTest) {
	using namespace poser;
	TensorBlob tensor;
	tensor.Reserve<float4>(1000, MemoryContext::CpuMemory, sizeof(float3));
	EXPECT_EQ(tensor.TensorFlattenSize(), 0);
	EXPECT_EQ(tensor.ValidTypeByte(), sizeof(float3));
	EXPECT_FALSE(tensor.IsVector());
	
	//Resize it
	tensor.Reset<float4>(20);
	EXPECT_EQ(tensor.TensorFlattenSize(), 20);
	EXPECT_TRUE(tensor.IsVector());
	
	auto slice = tensor.GetTypedTensorReadWrite<float4>();
	EXPECT_TRUE(slice.Size() == 20);
	for(auto i = 0; i < slice.Size(); i++)
		slice[i] = make_float4(1, 2, 3, 4);
	
	auto* data = (const float4*) tensor.RawPtr();
	for(auto i = 0; i < 20; i++) {
		EXPECT_NEAR(data[i].x, 1, 1e-6);
		EXPECT_NEAR(data[i].y, 2, 1e-6);
		EXPECT_NEAR(data[i].z, 3, 1e-6);
		EXPECT_NEAR(data[i].w, 4, 1e-6);
	}
	
	//Save it
	json j;
	j["tensor"] = tensor;
	
	//To and load from string
	std::string j_str = j.dump();
	std::stringstream ss(j_str);
	json load_j;
	ss >> load_j;
	
	auto load_tensor = load_j["tensor"].get<TensorBlob>();
	EXPECT_TRUE(load_tensor.TypeSizeMatched<float4>());
	EXPECT_EQ(load_tensor.ByteCapacity(), tensor.ByteCapacity());
	EXPECT_NE(load_tensor.RawPtr(), tensor.RawPtr());
	EXPECT_EQ(load_tensor.TensorFlattenSize(), tensor.TensorFlattenSize());
	EXPECT_EQ(tensor.ValidTypeByte(), load_tensor.ValidTypeByte());
	data = (const float4*) load_tensor.RawPtr();
	for(auto i = 0; i < 20; i++) {
		EXPECT_NEAR(data[i].x, 1, 1e-6);
		EXPECT_NEAR(data[i].y, 2, 1e-6);
		EXPECT_NEAR(data[i].z, 3, 1e-6);
		EXPECT_NEAR(data[i].w, 4, 1e-6);
	}
}


TEST(SeralizeTest, BlobGPUTest) {
	using namespace poser;
	TensorBlob tensor;
	//First test an vector
	TensorDim dim(1000);
	tensor.Reset<char>(dim, MemoryContext::GpuMemory);
	
	//Check it
	EXPECT_TRUE(tensor.TypeSizeMatched<char>());
	EXPECT_FALSE(tensor.TypeSizeMatched<double>());
	
	//Check the capacity
	EXPECT_EQ(tensor.ByteCapacity(), sizeof(char) * 1000);
	
	//Set the meomry
	auto* ptr = tensor.RawPtr();
	cudaSafeCall(cudaMemset(ptr, 1, sizeof(char) * 1000));
	
	//Copy to host
	std::vector<char> host_mem;
	host_mem.resize(1000);
	cudaSafeCall(cudaMemcpy(host_mem.data(), ptr, sizeof(char) * 1000, cudaMemcpyDeviceToHost));
	for(const auto& elem : host_mem)
		EXPECT_EQ(elem, 1);
	
	//Save it
	json j;
	j["tensor"] = tensor;
	
	//To and load from string
	std::string j_str = j.dump();
	std::stringstream ss(j_str);
	json load_j;
	ss >> load_j;
	
	//Load it
	auto load_tensor = load_j["tensor"].get<TensorBlob>();
	
	//Check it
	ptr = load_tensor.RawPtr();
	cudaSafeCall(cudaMemcpy(host_mem.data(), ptr, sizeof(char) * 1000, cudaMemcpyDeviceToHost));
	for(const auto& elem : host_mem)
		EXPECT_EQ(elem, 1);
}

TEST(SeralizeTest, FeatureMapTest) {
	using namespace poser;
	FeatureMap feature_map;
	
	//Do allocation
	auto dense_depth = CommonFeatureChannelKey::RawDepthImage();
	EXPECT_TRUE(feature_map.AllocateDenseFeature<unsigned short>(dense_depth, TensorDim(480, 640)));
	EXPECT_FALSE(feature_map.AllocateDenseFeature<unsigned short>(dense_depth, TensorDim(480, 640)));
	
	//Assign some
	auto dense_depth_map = feature_map.GetTypedDenseFeatureReadWrite<unsigned short>(dense_depth.get_name_key(), MemoryContext::CpuMemory);
	for(auto r_idx = 0; r_idx < dense_depth_map.Rows(); r_idx++) {
		for(auto c_idx = 0; c_idx < dense_depth_map.Cols(); c_idx++)
			dense_depth_map(r_idx, c_idx) = c_idx;
	}
	
	//On gpu
	EXPECT_TRUE(feature_map.AllocateDenseFeature<unsigned short>(dense_depth, TensorDim(480, 640), MemoryContext::GpuMemory));
	EXPECT_FALSE(feature_map.AllocateDenseFeature<unsigned short>(dense_depth, TensorDim(480, 640), MemoryContext::GpuMemory));
	
	//Check the existence
	EXPECT_TRUE(feature_map.ExistDenseFeature(dense_depth.get_name_key(), MemoryContext::CpuMemory));
	EXPECT_TRUE(feature_map.ExistDenseFeature(dense_depth.get_name_key(), MemoryContext::GpuMemory));
	
	//Allocate the sparse feature
	EXPECT_TRUE(feature_map.AllocateSparseFeature<unsigned short>(dense_depth.get_name_key(), 4000, MemoryContext::CpuMemory));
	EXPECT_FALSE(feature_map.AllocateSparseFeature<unsigned short>(dense_depth.get_name_key(), 4000, MemoryContext::CpuMemory));
	
	//The gpu version
	EXPECT_TRUE(feature_map.AllocateSparseFeature<unsigned short>(dense_depth.get_name_key(), 5000, MemoryContext::GpuMemory));
	//auto& blob = feature_map.GetSparseFeatureValueBlob(dense_depth.get_name_key(), MemoryContext::GpuMemory);
	//blob.ResetNoAllocate<unsigned short>(TensorDim(5000));
	feature_map.ResizeSparseFeatureOrException(dense_depth.get_name_key(), 5000, MemoryContext::GpuMemory);
	auto feature_slice = feature_map.GetTypedSparseFeatureValueReadWrite<unsigned short>(dense_depth.get_name_key(), MemoryContext::GpuMemory);
	EXPECT_EQ(feature_slice.Size(), 5000);
	
	//Save it
	json j;
	j["map"] = feature_map;
	
	//To and load from string
	std::string j_str = j.dump();
	std::stringstream ss(j_str);
	json load_j;
	ss >> load_j;
	
	//Load it
	{
		//The sparse feature
		auto load_feature_map = load_j["map"].get<FeatureMap>();
		auto load_feature_slice = load_feature_map.GetTypedSparseFeatureValueReadWrite<unsigned short>(dense_depth.get_name_key(), MemoryContext::GpuMemory);
		EXPECT_EQ(load_feature_slice.Size(), 5000);
		
		//The dense map
		auto load_dense_depth = load_feature_map.GetTypedDenseFeatureReadOnly<unsigned short>(dense_depth.get_name_key(), MemoryContext::CpuMemory);
		for(auto r_idx = 0; r_idx < dense_depth_map.Rows(); r_idx++) {
			for(auto c_idx = 0; c_idx < dense_depth_map.Cols(); c_idx++)
				EXPECT_EQ(load_dense_depth(r_idx, c_idx), c_idx);
		}
		
		//Check the dim and capacity
		EXPECT_TRUE(feature_map.GetDenseFeatureCapacity() == load_feature_map.GetDenseFeatureCapacity());
		EXPECT_TRUE(feature_map.GetDenseFeatureDim() == load_feature_map.GetDenseFeatureDim());
	}
}