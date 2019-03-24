#include "imgproc/normal_map_computer.h"
#include "geometry_utils/eigen_pca33.h"
#include <device_launch_parameters.h>

namespace poser { namespace device {
	
	
	__global__ void createNormalRadiusMapKernel(
		const TensorView<float4> vertex_map,
		const int halfsize,
		float4* normal_map //The output, same size as input
	){
		const auto x = threadIdx.x + blockDim.x*blockIdx.x;
		const auto y = threadIdx.y + blockDim.y*blockIdx.y;
		
		//Out of bound, directly return
		if (x >= vertex_map.Cols() || y >= vertex_map.Rows()) return;
		
		//Very closed to the boundary, set zero and return
		if (x < halfsize || y < halfsize || x >= vertex_map.Cols() - halfsize || y >= vertex_map.Rows() - halfsize) {
			normal_map[x + y * vertex_map.Cols()] = make_float4(0, 0, 0, 0);
		}
		
		//This value must be written to surface at end
		float4 normal_value = make_float4(0, 0, 0, 0);
		
		//The vertex at the center
		const float4 vertex_center = vertex_map(y, x);
		if (!is_zero_vertex(vertex_center)) {
			float4 centeroid = make_float4(0, 0, 0, 0);
			int counter = 0;
			//First window search to determine the center
			for (int cy = y - halfsize; cy <= y + halfsize; cy += 1) {
				for (int cx = x - halfsize; cx <= x + halfsize; cx += 1) {
					const float4 p = vertex_map(cy, cx);
					if (!is_zero_vertex(p)) {
						centeroid.x += p.x;
						centeroid.y += p.y;
						centeroid.z += p.z;
						counter++;
					}
				}
			}//End of first window search
			
			//At least half of the window is valid
			if(counter > ((2 * halfsize + 1) * (2 * halfsize + 1) / 2)) {
				centeroid *= (1.0f / counter);
				float covariance[6] = { 0 };
				
				//Second window search to compute the normal
				for (int cy = y - halfsize; cy < y + halfsize; cy += 1) {
					for (int cx = x - halfsize; cx < x + halfsize; cx += 1) {
						const float4 p = vertex_map(cy, cx);
						if (!is_zero_vertex(p)) {
							const float4 diff = p - centeroid;
							//Compute the covariance
							covariance[0] += diff.x * diff.x; //(0, 0)
							covariance[1] += diff.x * diff.y; //(0, 1)
							covariance[2] += diff.x * diff.z; //(0, 2)
							covariance[3] += diff.y * diff.y; //(1, 1)
							covariance[4] += diff.y * diff.z; //(1, 2)
							covariance[5] += diff.z * diff.z; //(2, 2)
						}
					}
				}//End of second window search
				
				//The eigen value for normal
				eigen_pca33 eigen(covariance);
				float3 normal;
				eigen.compute(normal);
				if (dotxyz(normal, vertex_center) >= 0.0f) normal *= -1;
				
				//Write to local variable
				normal_value.x = normal.x;
				normal_value.y = normal.y;
				normal_value.z = normal.z;
				normal_value.w = 0.0;
			}//End of check the number of valid pixels
		}//If the vertex is non-zero
		
		//Write to the surface
		normal_map[x + y * vertex_map.Cols()] = normal_value;
	}
	
	
} // device
} // poser


void poser::DepthNormalMapComputer::ComputeNormalMapGPU(
	const poser::TensorView<unsigned short> &depth_map,
	const poser::TensorView<float4> &vertex_map,
	poser::TensorSlice<float4> normal_map,
	cudaStream_t stream
) {
	dim3 blk(16, 16);
	dim3 grid(div_up(depth_map.Cols(), blk.x), div_up(depth_map.Rows(), blk.y));
	device::createNormalRadiusMapKernel<<<grid, blk, 0, stream>>>(vertex_map, window_halfsize_, normal_map.RawPtr());
}