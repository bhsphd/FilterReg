//
// Created by wei on 9/10/18.
//

#include "common/tensor_blob.h"
#include "common/safe_call_utils.h"

#include <glog/logging.h>
#include <cuda_runtime_api.h>


poser::TensorBlob::TensorBlob()
: memory_context_(MemoryContext::CpuMemory),
  byte_capacity_(0), data_ptr_(nullptr),
  type_byte_(0), valid_type_byte_(0), tensor_dim_()
{}

poser::TensorBlob::TensorBlob(poser::TensorBlob &&other) noexcept
: memory_context_(other.memory_context_),
  byte_capacity_(other.byte_capacity_),
  data_ptr_(std::move(other.data_ptr_)),
  type_byte_(other.type_byte_),
  valid_type_byte_(other.valid_type_byte_),
  tensor_dim_(other.tensor_dim_)
{
	other.memory_context_ = MemoryContext::CpuMemory;
	other.byte_capacity_ = 0;
	other.data_ptr_ = nullptr;
	other.type_byte_ = 0;
	other.tensor_dim_ = TensorDim();
}

poser::TensorBlob& poser::TensorBlob::operator=(poser::TensorBlob &&other) noexcept {
	//First move it
	memory_context_ = other.memory_context_;
	byte_capacity_ = other.byte_capacity_;
	data_ptr_ = std::move(other.data_ptr_);
	type_byte_ = other.type_byte_;
	valid_type_byte_ = other.valid_type_byte_;
	tensor_dim_ = other.tensor_dim_;
	
	//Nullify the other
	other.memory_context_ = MemoryContext::CpuMemory;
	other.byte_capacity_ = 0;
	other.data_ptr_ = nullptr;
	other.type_byte_ = 0;
	other.tensor_dim_ = TensorDim();
	
	//Ok
	return *this;
}


/* The method to manage the memory
 */
void poser::TensorBlob::resetMemory(std::size_t byte_capacity, poser::MemoryContext context) {
	if(context == memory_context_ && byte_capacity_ >= byte_capacity)
		return;
	
	//Need to re-assign memory
	if(context == MemoryContext::CpuMemory)
		resetMemoryCPU(byte_capacity);
	else
		resetMemoryGPU(byte_capacity);
	
	//Clear the tensor after reset
	tensor_dim_ = TensorDim();
}

void poser::TensorBlob::resetMemoryCPU(std::size_t byte_capacity) {
	//Allocate the memory
	data_ptr_.reset(new char[byte_capacity], std::default_delete<char[]>());
	
	//After reset
	byte_capacity_ = byte_capacity;
	memory_context_ = MemoryContext::CpuMemory;
}

void poser::TensorBlob::resetMemoryGPU(std::size_t byte_capacity) {
	auto cuda_deleter = [](void* ptr) {
		cudaSafeCall(cudaFree(ptr));
	};
	
	//Allocate the memory
	char* ptr;
	cudaSafeCall(cudaMalloc((void**)&ptr, byte_capacity));
	data_ptr_.reset(ptr, cuda_deleter);
	
	//After reset
	byte_capacity_ = byte_capacity;
	memory_context_ = MemoryContext::GpuMemory;
}

void poser::TensorBlob::Reserve(
	unsigned typed_capacity,
	unsigned short type_byte,
	poser::MemoryContext context,
	unsigned short valid_type_byte
) {
	std::size_t byte_capacity = typed_capacity * type_byte;
	resetMemory(byte_capacity, context);
	
	tensor_dim_ = TensorDim();
	type_byte_ = type_byte;
	valid_type_byte_ = valid_type_byte > 0 ? valid_type_byte : type_byte;
}

void poser::TensorBlob::CloneTo(poser::TensorBlob &other) const {
	other.Reserve(TypedCapacity(), TypeByte(), GetMemoryContext(), ValidTypeByte());
	other.Reset(tensor_dim_, type_byte_, memory_context_, valid_type_byte_);
	
	//Copy the data
	const auto* from_ptr = RawPtr();
	auto* to_ptr = other.RawPtr();
	if(memory_context_ == MemoryContext::CpuMemory) {
		memcpy(to_ptr, from_ptr, ByteCapacity());
	} else {
		cudaSafeCall(cudaMemcpy(to_ptr, from_ptr, ByteCapacity(), cudaMemcpyDeviceToDevice));
	}
}

poser::BlobSlice poser::TensorBlob::GetTensorReadWrite() {
	return {(char*)data_ptr_.get(), tensor_dim_, type_byte_, valid_type_byte_, memory_context_};
}

poser::BlobView poser::TensorBlob::GetTensorReadOnly() const {
	return {(char*)data_ptr_.get(), tensor_dim_, type_byte_, valid_type_byte_, memory_context_};
}

//The method to save and load
void poser::TensorBlob::SaveToJson(nlohmann::json &node) const {
	//The actual data
	std::vector<char> data;
	data.resize(ByteCapacity());
	if(IsCpuMemory()) {
		memcpy(data.data(), RawPtr(), ByteCapacity());
	} else {
		cudaSafeCall(cudaMemcpy(data.data(), RawPtr(), ByteCapacity(), cudaMemcpyDeviceToHost));
	}
	node["data"] = data;
	
	node["context"] = GetMemoryContext();
	node["byte_capacity"] = ByteCapacity();
	node["type_byte"] = type_byte_;
	node["valid_type_byte"] = valid_type_byte_;
	node["tensor_dim"] = tensor_dim_;
}

void poser::TensorBlob::LoadFromJson(const nlohmann::json &node) {
	//Load meta data
	auto context = node["context"].get<poser::MemoryContext>();
	auto byte_capacity = node["byte_capacity"].get<std::size_t>();
	resetMemory(byte_capacity, context);
	type_byte_ = node["type_byte"].get<unsigned short>();
	valid_type_byte_ = node["valid_type_byte"].get<unsigned short>();
	tensor_dim_ = node["tensor_dim"].get<poser::TensorDim>();
	
	//Load data
	std::vector<char> data_vec = node["data"].get<std::vector<char>>();;
	if(IsCpuMemory()) {
		memcpy(RawPtr(), data_vec.data(), ByteCapacity());
	} else {
		cudaSafeCall(cudaMemcpy(RawPtr(), data_vec.data(), ByteCapacity(), cudaMemcpyHostToDevice));
	}
}

void poser::to_json(nlohmann::json &j, const poser::TensorBlob &rhs) {
	nlohmann::json json;
	rhs.SaveToJson(json);
	j = json;
}

void poser::from_json(const nlohmann::json &j, poser::TensorBlob &rhs) {
	rhs.LoadFromJson(j);
}