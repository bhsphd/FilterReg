//
// Created by wei on 9/23/18.
//

#include <gtest/gtest.h>

#include <random>

#include "geometry_utils/permutohedral_common.h"
#include "common/feature_map.h"

class PermutohedralRandomCloudTest : public ::testing::Test {
protected:
	void SetUp() override {
		//Allocate the memory
		point_cloud_.Reset<float4>(cloud_size_, poser::MemoryContext::CpuMemory, sizeof(float3));
		
		//Init randomly
		auto cloud_tensor = point_cloud_.GetTypedTensorReadWrite<float4>();
		std::uniform_real_distribution<float> uniform;
		std::random_device device;
		for(auto i = 0; i < point_cloud_.TensorFlattenSize(); i++) {
			auto& tensor_i = cloud_tensor[i];
			tensor_i.x = uniform(device);
			tensor_i.y = uniform(device);
			tensor_i.z = uniform(device);
			tensor_i.w = 1.0f;
		}
	}
	
	//The underline data
	poser::TensorBlob point_cloud_;
	const unsigned cloud_size_ = 3000;
};

TEST(PermutohedralTest, BasicTest) {
	using namespace poser;
	LatticeCoordKey<3> lattice_key;
	EXPECT_EQ(sizeof(lattice_key), 3 * sizeof(short));
}

TEST_F(PermutohedralRandomCloudTest, SplatTest) {
	//Pre allocated memory
	constexpr int feature_dim = 3;
	poser::LatticeCoordKey<feature_dim> lattice_key[feature_dim + 1];
	float lattice_weight[feature_dim + 2];
	
	//The tensor
	auto tensor = point_cloud_.GetTensorReadOnly();
	for(auto i = 0; i < tensor.Size(); i++) {
		auto tensor_i = tensor.ValidElemVectorAt<float>(i);
		poser::permutohedral_lattice_noblur(tensor_i.ptr, lattice_key, lattice_weight);
		
		//Check the weight
		for(auto k = 0; k < feature_dim + 1; k++)
			EXPECT_TRUE(lattice_weight[k] >= 0);
	}
}