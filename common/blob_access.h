//
// Created by wei on 9/11/18.
//

#pragma once

#include "common/tensor_access.h"
#include <glog/logging.h>

namespace poser {
	
	/* The BlobView/Slice class, similar to TensorView/Slice, maintains NOT OWNED
	 * access to the underline memory, which mainly comes from TensorBlob. The difference
	 * is that BlobView/Slice only know the BYTE SIZE of its element (instead of the detailed type).
	 * The design is very similar to OpenCV mat, but aims at accessed on GPU.
	 */
	class BlobView : public TensorAccessBase {
	protected:
		const char* data_ptr_;
		unsigned short type_byte_; // sizeof(T)
		unsigned short valid_type_byte_;
	public:
		__host__ __device__ BlobView()
			: TensorAccessBase(), data_ptr_(nullptr), type_byte_(0), valid_type_byte_(0) {}
		__host__ __device__ BlobView(
			const char* data,
			TensorDim tensor_dim, unsigned short type_byte_size,
			MemoryContext context)
			: TensorAccessBase(tensor_dim, context),
			  data_ptr_(data),
			  type_byte_(type_byte_size), valid_type_byte_(type_byte_size) {}
		__host__ __device__ BlobView(
			const char* data,
			TensorDim tensor_dim,
			unsigned short type_byte_size, unsigned short valid_type_byte,
			MemoryContext context)
			: TensorAccessBase(tensor_dim, context),
			  data_ptr_(data),
			  type_byte_(type_byte_size), valid_type_byte_(valid_type_byte) {}
		
		//Generic query method
		__host__ __device__ unsigned ByteSize() const { return tensor_dim_.total_size() * type_byte_; }
		__host__ __device__ unsigned short TypeByte() const { return type_byte_; }
		__host__ __device__ unsigned short ValidTypeByte() const { return valid_type_byte_; }
		__host__ __device__ const void* RawPtr() const { return data_ptr_; }
		
		//A small vector hold the value at an element
		template<typename T>
		struct ElemVector {
			T* ptr;
			unsigned short typed_size; //The size should be rather small here
			__host__ __device__ ElemVector(T* in_ptr, unsigned short in_size) : ptr(in_ptr), typed_size(in_size) {}
			
			//Simple access
			T& operator[](int idx) { return ptr[idx]; }
			const T& operator[](int idx) const { return ptr[idx]; }
		};
		
		//Get a sized ptr for element at flatten_idx
		template <typename T> __host__ __device__
		const ElemVector<T> ElemVectorAt(int flatten_idx) const {
			return ElemVector<T>((T*)(data_ptr_ + flatten_idx * type_byte_), type_byte_ / sizeof(T));
		}
		template <typename T> __host__ __device__
		ElemVector<T> ElemVectorAt(int r_idx, int c_idx) const {
			return ElemVector<T>((T*)(data_ptr_ + (c_idx + r_idx * tensor_dim_.cols()) * type_byte_), type_byte_ / sizeof(T));
		}
		
		//Use the valid size
		template <typename T> __host__ __device__
		const ElemVector<T> ValidElemVectorAt(int flatten_idx) const {
			return ElemVector<T>((T*)(data_ptr_ + flatten_idx * type_byte_), valid_type_byte_ / sizeof(T));
		}
		template <typename T> __host__ __device__
		ElemVector<T> ValidElemVectorAt(int r_idx, int c_idx) const {
			return ElemVector<T>((T*)(data_ptr_ + (c_idx + r_idx * tensor_dim_.cols()) * type_byte_), valid_type_byte_ / sizeof(T));
		}
		
		//The accessing method
		template<typename T> __host__ __device__
		const T& At(int flatten_idx) const {
			return ((const T*)(data_ptr_))[flatten_idx];
		}
		template<typename T> __host__ __device__
		const T& At(int r_idx, int c_idx) const {
			return ((const T*)(data_ptr_))[c_idx + r_idx * tensor_dim_.cols()];
		}
		
		//The accessing method with check
		template <typename T> __host__
		const T& AtWithCheck(int flatten_idx) const {
			LOG_ASSERT(sizeof(T) == type_byte_);
			return At<T>(flatten_idx);
		}
		template <typename T> __host__
		const T& AtWithCheck(int r_idx, int c_idx) const {
			LOG_ASSERT(sizeof(T) == type_byte_);
			return At<T>(r_idx, c_idx);
		}
		
		//To tensor type
		template <typename T> __host__
		TensorView<T> ToTensorView() const {
			LOG_ASSERT(sizeof(T) == type_byte_);
			return TensorView<T>((const T*)data_ptr_, tensor_dim_, memory_context_);
		}
		
		//The non-typed access method
		__host__ __device__ const void* At(int flatten_idx, unsigned short elem_byte_size) const {
			return (const void*)(data_ptr_ + (flatten_idx * elem_byte_size));
		}
		__host__ __device__ const void* At(int r_idx, int c_idx, unsigned short elem_byte_size) const {
			return (const void*)(data_ptr_ + ((r_idx * tensor_dim_.cols() + c_idx) * elem_byte_size));
		}
	};
	
	
	/* The slice class maintain read-write access
	 */
	class BlobSlice : public BlobView {
	public:
		explicit BlobSlice() = default;
		__host__ __device__ BlobSlice(
			char* data,
			TensorDim tensor_dim, unsigned short type_byte_size,
			MemoryContext context)
			: BlobView(data, tensor_dim, type_byte_size, context) {}
		__host__ __device__ BlobSlice(
			char* data,
			TensorDim tensor_dim,
			unsigned short type_byte_size, unsigned short valid_type_byte,
			MemoryContext context)
			: BlobView(data, tensor_dim, type_byte_size, valid_type_byte, context) {}
		
		//Generic query method
		__host__ __device__ void* RawPtr() { return (void*)data_ptr_; }
		template<typename T> __host__ __device__
		T& At(int flatten_idx) {
			return ((T*)(data_ptr_))[flatten_idx];
		}
		template<typename T> __host__ __device__
		T& At(int r_idx, int c_idx) {
			return ((T*)(data_ptr_))[c_idx + r_idx * tensor_dim_.cols()];
		}
		
		//The non-typed accessing method
		__host__ __device__ void* At(int flatten_idx, unsigned short elem_byte_size) {
			return (void*)(data_ptr_ + (flatten_idx * elem_byte_size));
		}
		__host__ __device__ void* At(int r_idx, int c_idx, unsigned short elem_byte_size) {
			return (void*)(data_ptr_ + ((r_idx * tensor_dim_.cols() + c_idx) * elem_byte_size));
		}
	};
}
