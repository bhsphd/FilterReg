//
// Created by wei on 9/10/18.
//

#pragma once


#include <cstdio>
#include <exception>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

//Macro for cublas
#ifndef cublasSafeCall
#define cublasSafeCall(err)    poser::__cublasSafeCall(err, __FILE__, __LINE__)
#endif

//Macro for cuda driver api
#ifndef cuSafeCall
#define cuSafeCall(err) poser::__cuSafeCall(err, __FILE__, __LINE__)
#endif

//Macro for cuda runtime api
#ifndef cudaSafeCall
#define cudaSafeCall(err) poser::__cudaSafeCall(err, __FILE__, __LINE__)
#endif

namespace poser {
	
	//The actual handler for function for cublas errors
	static inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line)
	{
		if(CUBLAS_STATUS_SUCCESS != err) {
			fprintf(stderr, "CUBLAS error in file '%s', line %d\n \nerror %d \nterminating!\n",__FILE__, __LINE__, err);
			cudaDeviceReset();
			std::exit(1);
		}
	}
	
	static inline void __cuSafeCall(CUresult err, const char* file, const int line) {
		if(err != CUDA_SUCCESS) {
			//Query the name and string of the error
			const char* error_name;
			cuGetErrorName(err, &error_name);
			const char* error_string;
			cuGetErrorString(err, &error_string);
			fprintf(stderr, "CUDA driver error %s: %s in the line %d of file %s \n", error_name, error_string, line, file);
			cudaDeviceReset();
			std::exit(1);
		}
	}
	
	static inline void __cudaSafeCall(cudaError_t err, const char *file, const int line)
	{
		if (cudaSuccess != err) {
			const char* err_name = cudaGetErrorName(err);
			const char* err_str = cudaGetErrorString(err);
			fprintf(stderr, "CUDA error %s: %s at line %d of file %s \n", err_name, err_str, line, file);
			cudaDeviceReset();
			std::exit(1);
		}
	}
}