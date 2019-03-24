//
// Created by wei on 9/10/18.
//

#pragma once

#include "common/tensor_utils.h"
#include "common/macro_copyable.h"

#include <vector_types.h>

namespace poser {
	
	
	class TensorAccessBase {
	protected:
		TensorDim tensor_dim_;
		MemoryContext memory_context_;
	public:
		__host__ __device__ TensorAccessBase() : tensor_dim_(), memory_context_(MemoryContext::CpuMemory) {}
		__host__ __device__ TensorAccessBase(TensorDim dim, MemoryContext context) : tensor_dim_(dim), memory_context_(context) {}
		
		//Simple interface
		__host__ MemoryContext GetMemoryContext() const { return memory_context_; }
		__host__ __device__ unsigned Size() const { return tensor_dim_.total_size(); }
		__host__ __device__ unsigned FlattenSize() const { return tensor_dim_.total_size(); }
		__host__ __device__ const TensorDim& DimensionalSize() const { return tensor_dim_; }
		__host__ __device__ unsigned Rows() const { return tensor_dim_.rows(); }
		__host__ __device__ unsigned Cols() const { return tensor_dim_.cols(); }
		__host__ bool IsCpuTensor() const { return memory_context_ == MemoryContext::CpuMemory; }
		__host__ bool IsGpuTensor() const { return memory_context_ == MemoryContext::GpuMemory; }
		__host__ MemoryContext Context() const { return memory_context_; }
	};
	
	
	/* The tensor types as the main access interface. Unlike tensorblob,
	 * these tensors know its type, and will potentially be accessed on GPU kernels.
	 * The TensorView is READ-ONLY, NOT OWNED. TensorView is very similar
	 * to const TensorSlice, but it can exist as a member of non-const object.
	 */
	template <typename T>
	class TensorView : public TensorAccessBase {
	protected:
		const T* data_;
	public:
		__host__ __device__ TensorView() : TensorAccessBase(), data_(nullptr) {}
		__host__ __device__ TensorView(
			const T* array, unsigned start_idx, unsigned end_idx,
			MemoryContext context)
			: TensorAccessBase(end_idx - start_idx, context), data_(array + start_idx) {}
		__host__ __device__ TensorView(const T* array, TensorDim tensor_dim, MemoryContext context)
			: TensorAccessBase(tensor_dim, context), data_(array) {}
		__host__ TensorView(const std::vector<T>& vec)
		    : TensorAccessBase(vec.size(), MemoryContext::CpuMemory), data_(vec.data()) {}
		
		//Simple interface
		__host__ __device__ unsigned ByteSize() const { return tensor_dim_.total_size() * sizeof(T); }
		__host__ __device__ const T* RawPtr() const { return data_; }
		__host__ __device__ operator const T*() const { return data_; }
		
		//Flatten access interface
		__host__ __device__ const T& operator[](int flatten_idx) const { return data_[flatten_idx]; }
		
		//2D access interface
		__host__ __device__ const T& operator()(int r_idx, int c_idx) const { return data_[c_idx + r_idx * tensor_dim_.cols()]; }
	};
	
	
	/* The TensorSlice class maintains a know type and READ-WRITE, NOT OWNED access
	 * to the underline memory. It is potentially accessed on GPU kernels.
	 */
	template<typename T>
	class TensorSlice : public TensorView<T> {
	public:
		__host__ __device__ TensorSlice() : TensorView<T>() {}
		__host__ __device__ TensorSlice(
			T* array, unsigned start_idx, unsigned end_idx,
			MemoryContext context)
			: TensorView<T>(array, start_idx, end_idx, context) {}
		__host__ __device__ TensorSlice(T* array, TensorDim tensor_dim, MemoryContext context)
			: TensorView<T>(array, tensor_dim, context) {}
		__host__ TensorSlice(std::vector<T>& vec) : TensorView<T>(vec) {}
		
		//The non-const interface compared to TensorView
		__host__ __device__ T* RawPtr() { return (T*)TensorView<T>::data_; }
		__host__ __device__ T& operator[](int flatten_idx) { return ((T*)TensorView<T>::data_)[flatten_idx]; }
		__host__ __device__ T& operator()(int r_idx, int c_idx) { return ((T*)TensorView<T>::data_)[c_idx + r_idx * TensorView<T>::tensor_dim_.cols()]; }
	};
}
