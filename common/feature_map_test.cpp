//
// Created by wei on 9/10/18.
//

#include "common/feature_map.h"
#include "common/feature_channel_type.h"

#include <vector_functions.h>
#include <gtest/gtest.h>

TEST(FeatureMapTest, BasicTest) {
	using namespace poser;
	FeatureMap feature_map;
	
	//The dense depth
	auto dense_depth = CommonFeatureChannelKey::RawDepthImage();
	EXPECT_TRUE(feature_map.AllocateDenseFeature<unsigned short>(dense_depth, TensorDim(480, 640)));
	EXPECT_FALSE(feature_map.AllocateDenseFeature<unsigned short>(dense_depth, TensorDim(480, 640)));
	
	//Get it
	auto depth_map = feature_map.GetTypedDenseFeatureReadWrite<unsigned short>(dense_depth.get_name_key());
	auto size = depth_map.DimensionalSize();
	EXPECT_TRUE(size.rows() == 480);
	EXPECT_TRUE(size.cols() == 640);
	
	//Write some values
	for(auto i = 0; i < depth_map.Size(); i++) {
		depth_map[i] = i % size.cols();
	}
	
	//Iterate by row and col
	auto depth_map_view = feature_map.GetTypedDenseFeatureReadOnly<unsigned short>(dense_depth.get_name_key());
	for(auto row_idx = 0; row_idx < size.rows(); row_idx++) {
		for(auto col_idx = 0; col_idx < size.cols(); col_idx++) {
			EXPECT_EQ(depth_map_view(row_idx, col_idx), col_idx);
		}
	}
}

TEST(FeatureMapTest, BlobViewTest) {
	using namespace poser;
	FeatureMap feature_map;
	
	//The dense depth
	auto dense_depth = CommonFeatureChannelKey::RawDepthImage();
	EXPECT_TRUE(feature_map.AllocateDenseFeature<unsigned short>(dense_depth, TensorDim(480, 640)));
	EXPECT_FALSE(feature_map.AllocateDenseFeature<unsigned short>(dense_depth, TensorDim(480, 640)));
	
	//Get it
	auto depth_map = feature_map.GetTypedDenseFeatureReadWrite<unsigned short>(dense_depth.get_name_key());
	auto size = depth_map.DimensionalSize();
	EXPECT_TRUE(size.rows() == 480);
	EXPECT_TRUE(size.cols() == 640);
	
	//Write some values
	for(auto i = 0; i < depth_map.Size(); i++) {
		depth_map[i] = i % size.cols();
	}
	
	//Query using blob view
	auto depth_view = feature_map.GetDenseFeatureReadOnly(dense_depth.get_name_key(), MemoryContext::CpuMemory);
	EXPECT_EQ(depth_view.Rows(), 480);
	EXPECT_EQ(depth_view.Cols(), 640);
	for(auto row_idx = 0; row_idx < size.rows(); row_idx++) {
		for(auto col_idx = 0; col_idx < size.cols(); col_idx++) {
			EXPECT_EQ(depth_view.At<unsigned short>(row_idx, col_idx), col_idx);
		}
	}
}

TEST(FeatureMapTest, ManyFeatureTest) {
	using namespace poser;
	FeatureMap feature_map;
	
	//Prepare a lot of types
	const auto test_size = 40;
	std::vector<FeatureChannelType> type_vec;
	for(auto i = 0; i < test_size; i++) {
		bool is_sparse = (i % 2 == 0);
		type_vec.emplace_back(FeatureChannelType("feature" + std::to_string(i), is_sparse));
	}
	
	//Do it
	for(const auto& type : type_vec) {
		bool is_sparse = type.is_sparse();
		if(is_sparse) {
			feature_map.AllocateSparseFeature<float4>(type.get_name_key(), 1000);
		} else {
			feature_map.AllocateDenseFeature<float4>(type, {1000});
		}
	}
	
	//Check it
	for(const auto& type : type_vec) {
		bool is_sparse = type.is_sparse();
		if(is_sparse) {
			EXPECT_TRUE(feature_map.ExistFeature(type, MemoryContext::CpuMemory));
		} else {
			EXPECT_TRUE(feature_map.ExistFeature(type, MemoryContext::CpuMemory));
		}
	}
}

TEST(FeatureMapTest, MixedContextTest) {
	using namespace poser;
	FeatureMap feature_map;
	
	//Do allocation
	auto dense_depth = CommonFeatureChannelKey::RawDepthImage();
	EXPECT_TRUE(feature_map.AllocateDenseFeature<unsigned short>(dense_depth, TensorDim(480, 640)));
	EXPECT_FALSE(feature_map.AllocateDenseFeature<unsigned short>(dense_depth, TensorDim(480, 640)));
	
	//On gpu
	EXPECT_TRUE(feature_map.AllocateDenseFeature<unsigned short>(dense_depth, TensorDim(480, 640), MemoryContext::GpuMemory));
	EXPECT_FALSE(feature_map.AllocateDenseFeature<unsigned short>(dense_depth, TensorDim(480, 640), MemoryContext::GpuMemory));
	
	//Check the existance
	EXPECT_TRUE(feature_map.ExistDenseFeature(dense_depth.get_name_key(), MemoryContext::CpuMemory));
	EXPECT_TRUE(feature_map.ExistDenseFeature(dense_depth.get_name_key(), MemoryContext::GpuMemory));
	
	//Allocate the sparse feature
	EXPECT_TRUE(feature_map.AllocateSparseFeature<unsigned short>(dense_depth.get_name_key(), 4000, MemoryContext::CpuMemory));
	EXPECT_FALSE(feature_map.AllocateSparseFeature<unsigned short>(dense_depth.get_name_key(), 4000, MemoryContext::CpuMemory));
	
	//The gpu version
	EXPECT_TRUE(feature_map.AllocateSparseFeature<unsigned short>(dense_depth.get_name_key(), 5000, MemoryContext::GpuMemory));
	feature_map.ResizeSparseFeatureOrException(dense_depth.get_name_key(), 5000, MemoryContext::GpuMemory);
	auto feature_slice = feature_map.GetTypedSparseFeatureValueReadWrite<unsigned short>(dense_depth.get_name_key(), MemoryContext::GpuMemory);
	EXPECT_EQ(feature_slice.Size(), 5000);
}

TEST(FeatureMapTest, MetaTypeRetrieveTest) {
	using namespace poser;
	FeatureMap feature_map;
	
	//Do allocation
	auto dense_depth_key = CommonFeatureChannelKey::RawDepthImage();
	EXPECT_TRUE(feature_map.AllocateDenseFeature<unsigned short>(dense_depth_key, TensorDim(480, 640)));
	EXPECT_FALSE(feature_map.AllocateDenseFeature<unsigned short>(dense_depth_key, TensorDim(480, 640)));
	
	//On gpu
	EXPECT_TRUE(feature_map.AllocateDenseFeature<unsigned short>(dense_depth_key, TensorDim(480, 640), MemoryContext::GpuMemory));
	EXPECT_FALSE(feature_map.AllocateDenseFeature<unsigned short>(dense_depth_key, TensorDim(480, 640), MemoryContext::GpuMemory));
	
	//Use sparse feature
	std::string keypoint_key = "HandKeyPoints";
	EXPECT_TRUE(feature_map.AllocateSparseFeature<float4>(keypoint_key, 50, MemoryContext::CpuMemory));
	EXPECT_TRUE(feature_map.AllocateSparseFeature<float4>(keypoint_key, 50, MemoryContext::GpuMemory));
	EXPECT_FALSE(feature_map.AllocateSparseFeature<float4>(keypoint_key, 50, MemoryContext::CpuMemory));
	EXPECT_FALSE(feature_map.AllocateSparseFeature<float4>(keypoint_key, 50, MemoryContext::GpuMemory));
	
	//Construct the meta type
	FeatureChannelType keypoint(keypoint_key, sizeof(float4), sizeof(float4), true);
	
	//Check it
	EXPECT_TRUE(feature_map.ExistFeature(dense_depth_key, MemoryContext::CpuMemory));
	EXPECT_TRUE(feature_map.ExistFeature(dense_depth_key, MemoryContext::GpuMemory));
	EXPECT_TRUE(feature_map.ExistFeature(keypoint, MemoryContext::CpuMemory));
	EXPECT_TRUE(feature_map.ExistFeature(keypoint, MemoryContext::GpuMemory));
	
	//Get the sparse tensor and check it
	feature_map.ResizeSparseFeatureOrException(keypoint.get_name_key(), TensorDim{30}, MemoryContext::CpuMemory);
	auto tensor = feature_map.GetTypedFeatureValueReadWrite<float4>(keypoint, MemoryContext::CpuMemory);
	EXPECT_TRUE(tensor.Size() == 30);
	EXPECT_TRUE(tensor.Rows() == 30);
	EXPECT_TRUE(tensor.Cols() == 1);
	for(auto i = 0; i < 30; i++) {
		tensor[i] = make_float4(1, 2, 3, 4);
	}
	
	auto keypoint_view = feature_map.GetTypedFeatureValueReadOnly<float4>(keypoint, MemoryContext::CpuMemory);
	for(auto i = 0; i < 30; i++) {
		EXPECT_NEAR(keypoint_view[i].x, 1, 1e-6);
		EXPECT_NEAR(keypoint_view[i].y, 2, 1e-6);
		EXPECT_NEAR(keypoint_view[i].z, 3, 1e-6);
		EXPECT_NEAR(keypoint_view[i].w, 4, 1e-6);
	}
}

TEST(FeatureMapTest, InsertTest) {
	using namespace poser;
	const unsigned test_size = 4000;
	
	//The initial model should be empty
	FeatureMap geometric_model;
	
	//Construct some stupid input
	TensorBlob point_cloud;
	point_cloud.Reset<float4>(test_size, MemoryContext::CpuMemory);
	auto blob = point_cloud.GetTensorReadWrite();
	EXPECT_EQ(blob.FlattenSize(), test_size);
	for(auto i = 0; i < blob.FlattenSize(); i++) {
		blob.At<float4>(i) = make_float4(i + 1, i + 2, i + 3, i + 4);
	}
	
	//Insert into graph
	EXPECT_TRUE(geometric_model.InsertDenseFeature(CommonFeatureChannelKey::ReferenceVertex().get_name_key(), point_cloud.GetTensorReadOnly()));
	EXPECT_TRUE(geometric_model.InsertDenseFeature(CommonFeatureChannelKey::ReferenceNormal().get_name_key(), point_cloud.GetTensorReadOnly()));
	
	//Test the case in model
	EXPECT_TRUE(geometric_model.ExistFeature(CommonFeatureChannelKey::ReferenceVertex(), MemoryContext::CpuMemory));
	EXPECT_TRUE(geometric_model.ExistFeature(CommonFeatureChannelKey::ReferenceNormal(), MemoryContext::CpuMemory));
	auto vertex = geometric_model.GetTypedDenseFeatureReadOnly<float4>(CommonFeatureChannelKey::ReferenceVertex().get_name_key(), MemoryContext::CpuMemory);
	auto normal = geometric_model.GetTypedDenseFeatureReadOnly<float4>(CommonFeatureChannelKey::ReferenceNormal().get_name_key(), MemoryContext::CpuMemory);
	EXPECT_EQ(vertex.Size(), normal.Size());
	EXPECT_NE(vertex.RawPtr(), normal.RawPtr());
	for(auto i = 0; i < vertex.FlattenSize(); i++) {
		EXPECT_NEAR(vertex[i].x, i + 1, 1e-6f);
		EXPECT_NEAR(vertex[i].y, i + 2, 1e-6f);
		EXPECT_NEAR(vertex[i].z, i + 3, 1e-6f);
		EXPECT_NEAR(vertex[i].w, i + 4, 1e-6f);
		
		EXPECT_NEAR(normal[i].x, i + 1, 1e-6f);
		EXPECT_NEAR(normal[i].y, i + 2, 1e-6f);
		EXPECT_NEAR(normal[i].z, i + 3, 1e-6f);
		EXPECT_NEAR(normal[i].w, i + 4, 1e-6f);
	}
	
	//Test insert using typed feature
	geometric_model.InsertDenseFeature<float4>(CommonFeatureChannelKey::LiveVertex(), point_cloud.GetTypedTensorReadOnly<float4>());
	EXPECT_TRUE(geometric_model.ExistFeature(CommonFeatureChannelKey::LiveVertex(), MemoryContext::CpuMemory));
	auto live_vertex = geometric_model.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::LiveVertex(), MemoryContext::CpuMemory);
	for(auto i = 0; i < live_vertex.FlattenSize(); i++) {
		EXPECT_NEAR(live_vertex[i].x, i + 1, 1e-6f);
		EXPECT_NEAR(live_vertex[i].y, i + 2, 1e-6f);
		EXPECT_NEAR(live_vertex[i].z, i + 3, 1e-6f);
		EXPECT_NEAR(live_vertex[i].w, i + 4, 1e-6f);
	}
}

TEST(FeatureMapTest, CloneTest) {
	using namespace poser;
	const unsigned test_size = 4000;
	
	//The initial model should be empty
	FeatureMap geometric_model;
	
	//Construct some stupid input
	TensorBlob point_cloud;
	point_cloud.Reset<float4>(test_size, MemoryContext::CpuMemory);
	auto blob = point_cloud.GetTensorReadWrite();
	EXPECT_EQ(blob.FlattenSize(), test_size);
	for(auto i = 0; i < blob.FlattenSize(); i++) {
		blob.At<float4>(i) = make_float4(i + 1, i + 2, i + 3, i + 4);
	}
	
	//Insert into graph
	EXPECT_TRUE(geometric_model.InsertDenseFeature(CommonFeatureChannelKey::ReferenceVertex().get_name_key(), point_cloud.GetTensorReadOnly()));
	EXPECT_TRUE(geometric_model.InsertDenseFeature(CommonFeatureChannelKey::ReferenceNormal().get_name_key(), point_cloud.GetTensorReadOnly()));
	
	//Test the case in model
	EXPECT_TRUE(geometric_model.ExistFeature(CommonFeatureChannelKey::ReferenceVertex(), MemoryContext::CpuMemory));
	EXPECT_TRUE(geometric_model.ExistFeature(CommonFeatureChannelKey::ReferenceNormal(), MemoryContext::CpuMemory));
	auto vertex = geometric_model.GetTypedDenseFeatureReadOnly<float4>(CommonFeatureChannelKey::ReferenceVertex().get_name_key(), MemoryContext::CpuMemory);
	auto normal = geometric_model.GetTypedDenseFeatureReadOnly<float4>(CommonFeatureChannelKey::ReferenceNormal().get_name_key(), MemoryContext::CpuMemory);
	EXPECT_EQ(vertex.Size(), normal.Size());
	EXPECT_NE(vertex.RawPtr(), normal.RawPtr());
	for(auto i = 0; i < vertex.FlattenSize(); i++) {
		EXPECT_NEAR(vertex[i].x, i + 1, 1e-6f);
		EXPECT_NEAR(vertex[i].y, i + 2, 1e-6f);
		EXPECT_NEAR(vertex[i].z, i + 3, 1e-6f);
		EXPECT_NEAR(vertex[i].w, i + 4, 1e-6f);
		
		EXPECT_NEAR(normal[i].x, i + 1, 1e-6f);
		EXPECT_NEAR(normal[i].y, i + 2, 1e-6f);
		EXPECT_NEAR(normal[i].z, i + 3, 1e-6f);
		EXPECT_NEAR(normal[i].w, i + 4, 1e-6f);
	}
	
	FeatureMap copied_model;
	geometric_model.CloneTo(copied_model);
	
	//Test the case in copied model
	EXPECT_TRUE(copied_model.ExistFeature(CommonFeatureChannelKey::ReferenceVertex(), MemoryContext::CpuMemory));
	EXPECT_TRUE(copied_model.ExistFeature(CommonFeatureChannelKey::ReferenceNormal(), MemoryContext::CpuMemory));
	auto copied_vertex = copied_model.GetTypedDenseFeatureReadOnly<float4>(CommonFeatureChannelKey::ReferenceVertex().get_name_key(), MemoryContext::CpuMemory);
	auto copied_normal = copied_model.GetTypedDenseFeatureReadOnly<float4>(CommonFeatureChannelKey::ReferenceNormal().get_name_key(), MemoryContext::CpuMemory);
	EXPECT_EQ(copied_vertex.Size(), copied_normal.Size());
	EXPECT_NE(vertex.RawPtr(), copied_vertex.RawPtr());
	for(auto i = 0; i < vertex.FlattenSize(); i++) {
		EXPECT_NEAR(copied_vertex[i].x, i + 1, 1e-6f);
		EXPECT_NEAR(copied_vertex[i].y, i + 2, 1e-6f);
		EXPECT_NEAR(copied_vertex[i].z, i + 3, 1e-6f);
		EXPECT_NEAR(copied_vertex[i].w, i + 4, 1e-6f);
		
		EXPECT_NEAR(copied_normal[i].x, i + 1, 1e-6f);
		EXPECT_NEAR(copied_normal[i].y, i + 2, 1e-6f);
		EXPECT_NEAR(copied_normal[i].z, i + 3, 1e-6f);
		EXPECT_NEAR(copied_normal[i].w, i + 4, 1e-6f);
	}
}