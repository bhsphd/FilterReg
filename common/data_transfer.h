//
// Created by wei on 9/12/18.
//

#pragma once

#include "common/blob_access.h"
#include "common/tensor_access.h"
#include "common/safe_call_utils.h"

#include <cuda_runtime_api.h>

namespace poser {
	
	//Copy from tensor view to tensor slice
	template <typename T>
	void TensorCopyNoSync(const TensorView<T>& from, TensorSlice<T> to, cudaStream_t stream = 0) {
		LOG_ASSERT(from.FlattenSize() == to.FlattenSize());
		if(from.IsGpuTensor() && to.IsCpuTensor())
			cudaSafeCall(cudaMemcpyAsync(to.RawPtr(), from.RawPtr(), from.ByteSize(), cudaMemcpyDeviceToHost, stream));
		else if(from.IsCpuTensor() && to.IsGpuTensor())
			cudaSafeCall(cudaMemcpyAsync(to.RawPtr(), from.RawPtr(), from.ByteSize(), cudaMemcpyHostToDevice, stream));
		else if(from.IsGpuTensor() && to.IsGpuTensor())
			cudaSafeCall(cudaMemcpyAsync(to.RawPtr(), from.RawPtr(), from.ByteSize(), cudaMemcpyDeviceToDevice, stream));
		else
			//Should not use cudaMemcpy as all the host ptr is not page locked from cudaMallocHost
			memcpy(to.RawPtr(), from.RawPtr(), from.ByteSize());
	}
	
	void BlobCopyNoSync(const BlobView& from, BlobSlice to, cudaStream_t stream = 0);
}
