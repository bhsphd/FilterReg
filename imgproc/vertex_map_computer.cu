#include "imgproc/vertex_map_computer.h"

namespace poser { namespace device {
	
	__global__ void computeVertexMapKernel(
		const TensorView<unsigned short> depth_map,
		const IntrinsicInverse intrinsic_inv,
		float4* vertex_map
	) {
		const auto x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto y = threadIdx.y + blockDim.y * blockIdx.y;
		if(x >= depth_map.Cols() || y >= depth_map.Rows()) return;
		
		//Obtain the value and perform back-projecting
		const unsigned short raw_depth = depth_map(y, x);
		float4 vertex;
		
		//scale the depth to [m]
		//The depth image is always in [mm]
		vertex.z = float(raw_depth) / (1000.f);
		vertex.x = (x - intrinsic_inv.principal_x) * intrinsic_inv.inv_focal_x * vertex.z;
		vertex.y = (y - intrinsic_inv.principal_y) * intrinsic_inv.inv_focal_y * vertex.z;
		vertex_map[y * depth_map.Cols() + x] = vertex;
	}
	
	
} // device
} // poser

void poser::DepthVertexMapComputer::ComputeVertexMapGPU(
	const poser::TensorView<unsigned short> &depth_img,
	poser::TensorSlice<float4> vertex_map,
	cudaStream_t stream
) {
	dim3 blk(16, 16);
	dim3 grid(div_up(depth_img.Cols(), blk.x), div_up(depth_img.Rows(), blk.y));
	device::computeVertexMapKernel<<<grid, blk, 0, stream>>>(
		depth_img,
		intrinsic_inv_,
		vertex_map.RawPtr()
	);
}