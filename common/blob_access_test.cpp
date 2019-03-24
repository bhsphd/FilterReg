//
// Created by wei on 9/17/18.
//

#include "common/feature_map.h"
#include "common/feature_channel_type.h"

#include <gtest/gtest.h>
#include <vector_functions.h>

TEST(BlobAccessTest, BlobViewBasicTest) {
	using namespace poser;
	
	//Allocate the feature map
	FeatureMap feature_map;
	feature_map.AllocateDenseFeature(CommonFeatureChannelKey::ObservationVertexCamera(), TensorDim(480, 640));
	
	//Get the map
	auto blob_view = feature_map.GetFeatureValueReadOnly(CommonFeatureChannelKey::ObservationVertexCamera(), MemoryContext::CpuMemory);
	EXPECT_EQ(blob_view.FlattenSize(), 480 * 640);
	
	//Assign with float4
	auto tensor = feature_map.GetTypedFeatureValueReadWrite<float4>(CommonFeatureChannelKey::ObservationVertexCamera(), MemoryContext::CpuMemory);
	for(auto i = 0; i < tensor.FlattenSize(); i++)
		tensor[i] = make_float4(0, 1, 2, 3);
	
	//Get it with float x
	for(auto i = 0; i < blob_view.FlattenSize(); i++) {
		auto elem_vec = blob_view.ElemVectorAt<float>(i);
		EXPECT_EQ(elem_vec.typed_size, 4);
		for(auto j = 0; j < elem_vec.typed_size; j++)
			EXPECT_NEAR(elem_vec[j], j, 1e-6);
	}
	
	//Another access
	for(auto r_idx = 0; r_idx < blob_view.Rows(); r_idx++) {
		for(auto c_idx = 0; c_idx < blob_view.Cols(); c_idx++) {
			auto elem_vec = blob_view.ElemVectorAt<float>(r_idx, c_idx);
			EXPECT_EQ(elem_vec.typed_size, 4);
			for(auto j = 0; j < elem_vec.typed_size; j++)
				EXPECT_NEAR(elem_vec[j], j, 1e-6);
		}
	}
}


TEST(BlobAccessTest, BlobSliceWriteTest) {
	using namespace poser;
	
	//Allocate the feature map
	FeatureMap feature_map;
	feature_map.AllocateDenseFeature(CommonFeatureChannelKey::ObservationVertexCamera(), TensorDim(480, 640));
	
	//Get the map
	auto blob_slice = feature_map.GetFeatureValueReadWrite(CommonFeatureChannelKey::ObservationVertexCamera(), MemoryContext::CpuMemory);
	EXPECT_EQ(blob_slice.Rows(), 480);
	EXPECT_EQ(blob_slice.Cols(), 640);
	for(auto r_idx = 0; r_idx < blob_slice.Rows(); r_idx++) {
		for(auto c_idx = 0; c_idx < blob_slice.Cols(); c_idx++) {
			auto elem_vec = blob_slice.ElemVectorAt<float>(r_idx, c_idx);
			EXPECT_EQ(elem_vec.typed_size, 4);
			for(auto j = 0; j < elem_vec.typed_size; j++)
				elem_vec[j] = float(j);
		}
	}
	
	//Get it with float x
	auto blob_view = feature_map.GetFeatureValueReadOnly(CommonFeatureChannelKey::ObservationVertexCamera(), MemoryContext::CpuMemory);
	for(auto i = 0; i < blob_view.FlattenSize(); i++) {
		auto elem_vec = blob_view.ElemVectorAt<float>(i);
		EXPECT_EQ(elem_vec.typed_size, 4);
		for(auto j = 0; j < elem_vec.typed_size; j++)
			EXPECT_NEAR(elem_vec[j], j, 1e-6);
	}
	
	//Another access
	for(auto r_idx = 0; r_idx < blob_view.Rows(); r_idx++) {
		for(auto c_idx = 0; c_idx < blob_view.Cols(); c_idx++) {
			auto elem_vec = blob_view.ElemVectorAt<float>(r_idx, c_idx);
			EXPECT_EQ(elem_vec.typed_size, 4);
			for(auto j = 0; j < elem_vec.typed_size; j++)
				EXPECT_NEAR(elem_vec[j], j, 1e-6);
		}
	}
}