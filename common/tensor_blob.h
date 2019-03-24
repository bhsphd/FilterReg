//
// Created by wei on 9/10/18.
//

#pragma once

#include "common/tensor_utils.h"
#include "common/macro_copyable.h"
#include "common/tensor_access.h"
#include "common/blob_access.h"
#include "common/safe_call_utils.h"

#include <glog/logging.h>
#include <cuda_runtime_api.h>

#include <memory>

namespace poser {
	
	/* The tensor blob is the underline storage for all
	 * the features in poser. It maintain an ptr to
	 * either cpu or gpu memory, depends on the context.
	 * This version only works for Plain Old Object
	 */
	class TensorBlob {
	private:
		//The info about the memory
		MemoryContext memory_context_;
		std::size_t byte_capacity_;
		std::shared_ptr<void> data_ptr_;
		
		//The nominal size of the blob. It is guaranteed that
		//tensor_dim.total_size() * sizeof(T) <= byte_capacity_
		TensorDim tensor_dim_;
		unsigned short type_byte_; //sizeof(T)
		unsigned short valid_type_byte_;
	public:
		TensorBlob();
		~TensorBlob() = default;
		TensorBlob(const TensorBlob& other) = default;
		TensorBlob& operator=(const TensorBlob& other) = default;
		TensorBlob(TensorBlob&& other) noexcept;
		TensorBlob& operator=(TensorBlob&& other) noexcept;
		
		
		/* The method to resize the tensor, if the context is not
		 * correct, release current memory and allocate new memory
		 */
	private:
		void resetMemory(std::size_t byte_capacity, MemoryContext context);
		void resetMemoryCPU(std::size_t byte_capacity);
		void resetMemoryGPU(std::size_t byte_capacity);
	public:
		//The allocation method that update the capacity while keep the size zero
		void Reserve(
			unsigned typed_capacity,
			unsigned short type_byte,
			MemoryContext context = MemoryContext::CpuMemory,
			unsigned short valid_type_byte = 0);
		template <typename T>
		void Reserve(
			unsigned T_capacity,
			MemoryContext context = MemoryContext::CpuMemory,
			unsigned short valid_type_byte = 0) { Reserve(T_capacity, sizeof(T), context, valid_type_byte); }
		
		//This reset change the nominal size and type_byte, may cause allocation
		void Reset(
			TensorDim dim,
			unsigned short type_byte,
			MemoryContext context = MemoryContext::CpuMemory,
			unsigned short valid_type_byte = 0) { Reserve(dim.total_size(), type_byte, context, valid_type_byte); tensor_dim_ = dim; }
		template <typename T>
		void Reset(TensorDim dim, MemoryContext context = MemoryContext::CpuMemory, unsigned short valid_type_byte = 0) { Reset(dim, sizeof(T), context, valid_type_byte); }
		template<typename T>
		void Reset(unsigned rows, unsigned cols, MemoryContext context = MemoryContext::CpuMemory, unsigned short valid_type_byte = 0) { Reset<T>(TensorDim(rows, cols), context, valid_type_byte); }
		template<typename T>
		void Reset(unsigned flatten_size, MemoryContext context = MemoryContext::CpuMemory, unsigned short valid_type_byte = 0) { Reset<T>(TensorDim(flatten_size), context, valid_type_byte); }
		
		//The set of method for deep copy or clone
		void CloneTo(TensorBlob& other) const;
		
		//Do not touch the allocated memory, only change the nominal size/byte size
		inline void ResizeOrException(TensorDim dim) {
			LOG_ASSERT(type_byte_ > 0 && dim.total_size() * type_byte_ <= byte_capacity_);
			tensor_dim_ = dim;
		}
		
		//General query method
		template <typename T>
		bool TypeSizeMatched() const { return sizeof(T) == type_byte_; }
		unsigned short TypeByte() const { return type_byte_; }
		unsigned short ValidTypeByte() const { return valid_type_byte_; }
		bool IsCpuMemory() const { return memory_context_ == MemoryContext::CpuMemory; }
		bool IsGpuMemory() const { return memory_context_ == MemoryContext::GpuMemory; }
		MemoryContext GetMemoryContext() const { return memory_context_; }
		std::size_t ByteCapacity() const { return byte_capacity_; }
		std::size_t TypedCapacity() const {
			LOG_ASSERT(type_byte_ > 0 || byte_capacity_ == 0);
			if(byte_capacity_ == 0) return 0;
			return byte_capacity_ / type_byte_;
		}
		const TensorDim& TensorDimension() const { return tensor_dim_; }
		void* RawPtr() { return data_ptr_.get(); }
		const void* RawPtr() const { return data_ptr_.get(); }
		
		//The size related method
		unsigned TensorFlattenSize() const { return tensor_dim_.total_size(); }
		bool IsVector() const { return tensor_dim_.is_vector(); }
		bool IsMatrix() const { return tensor_dim_.is_matrix(); }
		bool IsStrictVector() const { return tensor_dim_.is_strict_vector(); }
		bool IsStrictMatrix() const { return tensor_dim_.is_strict_matrix(); }
		
		//The access to tensor slice
		template <typename T>
		TensorSlice<T> GetTypedTensorReadWrite();
		template <typename T>
		const TensorView<T> GetTypedTensorReadOnly() const;
		
		//The access for blob view/slice
		BlobSlice GetTensorReadWrite();
		BlobView GetTensorReadOnly() const;
		
		//The type serialize method
		void SaveToJson(json& node) const;
		void LoadFromJson(const json& node);
	};
}

template<typename T>
poser::TensorSlice<T> poser::TensorBlob::GetTypedTensorReadWrite() {
	LOG_ASSERT(TypeSizeMatched<T>());
	return poser::TensorSlice<T>((T*)data_ptr_.get(), tensor_dim_, memory_context_);
}

template<typename T>
const poser::TensorView<T> poser::TensorBlob::GetTypedTensorReadOnly() const {
	LOG_ASSERT(TypeSizeMatched<T>());
	return poser::TensorView<T>((T*)data_ptr_.get(), tensor_dim_, memory_context_);
}



/* For serialize
 */
namespace poser {
	void to_json(nlohmann::json& j, const TensorBlob& rhs);
	void from_json(const nlohmann::json& j, TensorBlob& rhs);
}