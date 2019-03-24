//
// Created by wei on 9/14/18.
//

#pragma once

#include <vector_functions.h>

namespace poser {
	
	
	/* The intrinsic parameter to compute the transform.
	 * The default value is taken from Kinetc
	 */
	struct Intrinsic {
		__host__ __device__ Intrinsic()
			: principal_x(320), principal_y(240),
			  focal_x(570), focal_y(570) {}
		
		__host__ __device__ Intrinsic(
			const float focal_x_, const float focal_y_,
			const float principal_x_, const float principal_y_
		) : principal_x(principal_x_), principal_y(principal_y_),
		    focal_x(focal_x_), focal_y(focal_y_) {}
		
		//Cast to float4
		__host__ operator float4() {
			return make_float4(principal_x, principal_y, focal_x, focal_y);
		}
		
		// The paramters for camera intrinsic
		float principal_x, principal_y;
		float focal_x, focal_y;
	};
	
	struct IntrinsicInverse {
		__host__ __device__ IntrinsicInverse()
		: principal_x(320), principal_y(240),
		  inv_focal_x((1.0f / 320.0f)), inv_focal_y((1.0f / 240.0f)) {}
		  
		  
		__host__ __device__ IntrinsicInverse(const Intrinsic& intrinsic)
		: principal_x(intrinsic.principal_x), principal_y(intrinsic.principal_y),
		  inv_focal_x(1.0f / intrinsic.focal_x), inv_focal_y(1.0f / intrinsic.focal_y) {}
		
		// The paramters for camera intrinsic
		float principal_x, principal_y;
		float inv_focal_x, inv_focal_y;
	};
}
