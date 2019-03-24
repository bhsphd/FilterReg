//
// Created by wei on 9/10/18.
//

#pragma once

#include <vector_types.h>

#include "common/common_type.h"

namespace poser {
	
	/* Which context does the memory lives in
	 */
	enum class MemoryContext {
		CpuMemory,
		GpuMemory
	};
	
	
	/* A very small struct used to hold the size of a tensor.
	 * The size is no more than 3d in this project.
	 */
	class TensorDim {
	private:
		unsigned dim[2]; //The underline member
	public:
		__host__ __device__ TensorDim() { dim[0] = 0; dim[1] = 1; }
		__host__ __device__ TensorDim(unsigned vector_size) { dim[0] = vector_size; dim[1] = 1; }
		__host__ __device__ TensorDim(unsigned row, unsigned col) { dim[0] = row; dim[1] = col; }
		
		//The query of type. scalar is a special case of vector, and so on.
		__host__ __device__ bool is_scalar() const { return (dim[0] == 1 && dim[1] == 1); }
		__host__ __device__ bool is_vector() const { return (dim[0] >= 1 && dim[1] == 1); }
		__host__ __device__ bool is_matrix() const { return (dim[0] >= 1 && dim[1] >= 1); }
		
		//The strict query interface. Scalar is not vector
		__host__ __device__ bool is_strict_scalar() const { return (dim[0] == 1 && dim[1] == 1); }
		__host__ __device__ bool is_strict_vector() const { return (dim[0] >  1 && dim[1] == 1); }
		__host__ __device__ bool is_strict_matrix() const { return (dim[0] >  1 && dim[1] >  1); }
		
		//The size query interface
		__host__ __device__ unsigned total_size() const { return dim[0] * dim[1]; }
		__host__ __device__ unsigned rows() const { return dim[0]; }
		__host__ __device__ unsigned cols() const { return dim[1]; }
		
		//Comparator
		__host__ __device__ bool operator==(const TensorDim& rhs) const {
			return dim[0] == rhs.dim[0] && dim[1] == rhs.dim[1];
		}
	};
}

//For serialize
namespace poser {
	void to_json(json& j, const TensorDim& rhs);
	void from_json(const json& j, TensorDim& rhs);
}