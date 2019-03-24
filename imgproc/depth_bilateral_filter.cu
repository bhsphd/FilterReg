#include "imgproc/depth_bilateral_filter.h"


namespace poser { namespace device {
	
	__global__ void bilateralFilterDepthKernel(
		const TensorView<unsigned short> raw_depth,
		const float sigma_s_inv_square,
		const float sigma_r_inv_square,
		unsigned short* filter_depth
	) {
		//Parallel over the clipped image
		const auto x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto y = threadIdx.y + blockDim.y * blockIdx.y;
		if (y >= raw_depth.Rows() || x >= raw_depth.Cols()) return;
		
		//Compute the center on raw depth
		const auto half_width = 5;
		const unsigned short center_depth = raw_depth(y, x);
		
		//Iterate over the window
		float sum_all = 0.0f; float sum_weight = 0.0f;
		for(auto y_idx = y - half_width; y_idx <= y + half_width; y_idx++) {
			for(auto x_idx = x - half_width; x_idx <= x + half_width; x_idx++) {
				//const unsigned short depth = tex2D<unsigned short>(raw_depth, x_idx, y_idx);
				unsigned short depth = 0;
				if(x_idx < raw_depth.Cols() && y_idx < raw_depth.Rows())
					depth = raw_depth(y_idx, x_idx);
				
				//Do filtering
				const float depth_diff2 = (depth - center_depth) * (depth - center_depth);
				const float pixel_diff2 = (x_idx - x) * (x_idx - x) + (y_idx - y) * (y_idx - y);
				const float this_weight = (depth > 0) * expf(-sigma_s_inv_square * pixel_diff2) * expf(-sigma_r_inv_square * depth_diff2);
				sum_weight += this_weight;
				sum_all += this_weight * depth;
			}
		}
		
		//Put back to the filtered depth
		unsigned short filtered_depth_value = __float2uint_rn(sum_all / sum_weight);
		filter_depth[y * raw_depth.Cols() + x] = filtered_depth_value;
	}
	
	
} // device
} // poser

void poser::DepthBilateralFilter::PerformBilateralFilterGPU(
	const poser::TensorView<unsigned short> &raw_depth,
	poser::TensorSlice<unsigned short> filter_depth,
	cudaStream_t stream
) {
	dim3 blk(16, 16);
	dim3 grid(div_up(raw_depth.Cols(), blk.x), div_up(raw_depth.Rows(), blk.y));
	const float sigma_s_inv_square = 1.0f / (4.5f * 4.5f);
	const float sigma_r_inv_square = 1.0f / (depth_sigma_ * depth_sigma_);
	device::bilateralFilterDepthKernel<<<grid, blk, 0, stream>>>(
		raw_depth,
		sigma_s_inv_square, sigma_r_inv_square,
		filter_depth.RawPtr()
	);
}