//
// Created by wei on 9/10/18.
//

#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <vector_functions.h>

#include "common/tensor_blob.h"
#include "common/safe_call_utils.h"

TEST(TensorBlobTest, CpuTest) {
	using namespace poser;
	TensorBlob tensor;
	//First test an vector
	TensorDim dim(1000);
	tensor.Reset<float>(dim);
	
	//Check it
	EXPECT_TRUE(tensor.TypeSizeMatched<float>());
	EXPECT_FALSE(tensor.TypeSizeMatched<double>());
	
	//Check the capacity
	EXPECT_EQ(tensor.ByteCapacity(), sizeof(float) * 1000);
	
	//Check size
	EXPECT_EQ(tensor.TensorFlattenSize(), 1000);
	EXPECT_TRUE(tensor.IsStrictVector());
	
	//Assign some value and move it
	void* ptr = tensor.RawPtr();
	float* array = (float*)ptr;
	for(auto i = 0; i < 1000; i ++)
		array[i] = float(i);
	
	TensorBlob moved_tensor(std::move(tensor));
	array = (float*) moved_tensor.RawPtr();
	for(auto i = 0; i < 1000; i++) {
		EXPECT_NEAR(array[i], i, 1e-6);
	}
}

TEST(TensorBlobTest, CpuMapTest) {
	using namespace poser;
	TensorBlob tensor;
	//First test an vector
	TensorDim dim(20, 50);
	tensor.Reset<float>(dim);
	
	//Check it
	EXPECT_TRUE(tensor.TypeSizeMatched<float>());
	EXPECT_FALSE(tensor.TypeSizeMatched<double>());
	
	//Check the capacity
	EXPECT_EQ(tensor.ByteCapacity(), sizeof(float) * 1000);
	
	//Check size
	EXPECT_EQ(tensor.TensorFlattenSize(), 1000);
	EXPECT_TRUE(tensor.IsStrictMatrix());
	
	//Assign some value and move it
	void* ptr = tensor.RawPtr();
	float* array = (float*)ptr;
	for(auto i = 0; i < 1000; i ++)
		array[i] = float(i);
	
	TensorBlob moved_tensor(std::move(tensor));
	array = (float*) moved_tensor.RawPtr();
	for(auto i = 0; i < 1000; i++) {
		EXPECT_NEAR(array[i], i, 1e-6);
	}
}

TEST(TensorBlobTest, GpuArrayTest) {
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
	
	//Move it
	TensorBlob moved_tensor(std::move(tensor));
	EXPECT_TRUE(moved_tensor.IsStrictVector());
}

TEST(TensorBlobTest, AllocateTest) {
	using namespace poser;
	TensorBlob tensor;
	tensor.Reserve<float4>(1000);
	EXPECT_EQ(tensor.TensorFlattenSize(), 0);
	EXPECT_FALSE(tensor.IsVector());
	
	//Resize it
	//tensor.ResetNoAllocate<float4>(20);
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
}

TEST(TensorBlobTest, CopyTest) {
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
	
	//Move it
	TensorBlob copied_tensor(tensor);
	EXPECT_TRUE(copied_tensor.IsStrictVector());
	
	//The underline ptr is the same
	EXPECT_EQ(copied_tensor.RawPtr(), tensor.RawPtr());
}

TEST(TensorBlobTest, CloneTest) {
	using namespace poser;
	const unsigned T_capacity = 1000;
	const unsigned T_size = 200;
	
	TensorBlob blob;
	blob.Reserve<unsigned>(T_capacity, MemoryContext::CpuMemory);
	blob.ResizeOrException(TensorDim{T_size});
	EXPECT_EQ(blob.TypedCapacity(), T_capacity);
	EXPECT_EQ(blob.ByteCapacity(), T_capacity * sizeof(unsigned));
	EXPECT_EQ(blob.TensorFlattenSize(), T_size);
	
	//Write some value
	auto tensor = blob.GetTypedTensorReadWrite<unsigned>();
	EXPECT_EQ(tensor.Size(), 200);
	for(auto i = 0; i < tensor.Size(); i++)
		tensor[i] = i;
	
	//Copy it
	TensorBlob copied_blob;
	blob.CloneTo(copied_blob);
	EXPECT_NE(copied_blob.RawPtr(), blob.RawPtr());
	EXPECT_EQ(copied_blob.ByteCapacity(), blob.ByteCapacity());
	EXPECT_EQ(copied_blob.TensorFlattenSize(), T_size);
	auto copied_tensor = copied_blob.GetTypedTensorReadOnly<unsigned>();
	for(auto i = 0; i < copied_tensor.Size(); i++)
		EXPECT_EQ(copied_tensor[i], i);
}