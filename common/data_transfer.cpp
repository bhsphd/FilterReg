//
// Created by wei on 9/19/18.
//

#include "common/data_transfer.h"

void poser::BlobCopyNoSync(const poser::BlobView &from, poser::BlobSlice to, cudaStream_t stream) {
	LOG_ASSERT(from.ByteSize() == to.ByteSize());
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