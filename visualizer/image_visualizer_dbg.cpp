//
// Created by wei on 9/12/18.
//

#include "common/tensor_blob.h"
#include "common/data_transfer.h"
#include "visualizer/debug_visualizer.h"

#include <vector_functions.h>

//The opencv header depends on the version
#if CV_MAJOR_VERSION >= 3
#include <opencv2/highgui.hpp>
#else
#include <opencv2/highgui/highgui.hpp>
#endif


/* The implementation of actual method
 */
void poser::DebugVisualizer::DrawRGBImage(const cv::Mat &cv_bgr_img) {
	cv::imshow("ColorImage", cv_bgr_img);
	cv::waitKey(0);
}

void poser::DebugVisualizer::DrawRGBImage(const poser::TensorView<uchar4> &rgb_img) {
	if(rgb_img.IsCpuTensor())
		drawRGBImageCPU(rgb_img);
	else
		drawRGBImageGPU(rgb_img);
}

void poser::DebugVisualizer::drawRGBImageCPU(const poser::TensorView<uchar4> &rgb_img) {
	LOG_ASSERT(rgb_img.IsCpuTensor());
	cv::Mat cv_bgr_img(rgb_img.Rows(), rgb_img.Cols(), CV_8UC3);
	for(auto r_idx = 0; r_idx < rgb_img.Rows(); r_idx++) {
		for(auto c_idx = 0; c_idx < rgb_img.Cols(); c_idx++) {
			auto rgba = rgb_img(r_idx, c_idx);
			cv_bgr_img.at<uchar3>(r_idx, c_idx) = make_uchar3(rgba.z, rgba.y, rgba.x);
		}
	}
	
	DrawRGBImage(cv_bgr_img);
}

void poser::DebugVisualizer::drawRGBImageGPU(const poser::TensorView<uchar4> &rgb_img) {
	LOG_ASSERT(rgb_img.IsGpuTensor());
	auto rows = rgb_img.Rows();
	auto cols = rgb_img.Cols();
	TensorBlob cpu_blob;
	cpu_blob.Reset<uchar4>(rows, cols, MemoryContext::CpuMemory);
	auto rgb_cpu = cpu_blob.GetTypedTensorReadWrite<uchar4>();
	TensorCopyNoSync(rgb_img, rgb_cpu);
	drawRGBImageCPU(rgb_cpu);
}

/* The method for depth image
 */
void poser::DebugVisualizer::DrawDepthImage(const cv::Mat& depth_img) {
	double max_depth, min_depth;
	cv::minMaxIdx(depth_img, &min_depth, &max_depth);
	//Visualize depth-image in opencv
	cv::Mat depth_scale;
	cv::convertScaleAbs(depth_img, depth_scale, 255 / max_depth);
	cv::imshow("DepthImage", depth_scale);
	cv::waitKey(0);
}

void poser::DebugVisualizer::DrawDepthImage(const poser::TensorView<unsigned short> &depth_img) {
	if(depth_img.IsCpuTensor())
		drawDepthImageCPU(depth_img);
	else
		drawDepthImageGPU(depth_img);
}

void poser::DebugVisualizer::DrawForegroundMask(const poser::TensorView<unsigned char> &mask){
	LOG_ASSERT(mask.IsCpuTensor());
	auto rows = mask.Rows();
	auto cols = mask.Cols();
	cv::Mat depth_cpu(rows, cols, CV_8UC1);
	for(auto r_idx = 0; r_idx < rows; r_idx++)
		for(auto c_idx = 0; c_idx < cols; c_idx++)
			depth_cpu.at<unsigned char>(r_idx, c_idx) = mask(r_idx, c_idx);
	
	//Draw it
	DrawDepthImage(depth_cpu);
}

void poser::DebugVisualizer::drawDepthImageCPU(const poser::TensorView<unsigned short> &depth_img) {
	LOG_ASSERT(depth_img.IsCpuTensor());
	auto rows = depth_img.Rows();
	auto cols = depth_img.Cols();
	cv::Mat depth_cpu(rows, cols, CV_16UC1);
	for(auto r_idx = 0; r_idx < rows; r_idx++)
		for(auto c_idx = 0; c_idx < cols; c_idx++)
			depth_cpu.at<unsigned short>(r_idx, c_idx) = depth_img(r_idx, c_idx);
		
	//Draw it
	DrawDepthImage(depth_cpu);
}

void poser::DebugVisualizer::drawDepthImageGPU(const poser::TensorView<unsigned short> &depth_img) {
	LOG_ASSERT(depth_img.IsGpuTensor());
	auto rows = depth_img.Rows();
	auto cols = depth_img.Cols();
	TensorBlob cpu_blob;
	cpu_blob.Reset<unsigned short>(rows, cols, MemoryContext::CpuMemory);
	auto depth_cpu = cpu_blob.GetTypedTensorReadWrite<unsigned short>();
	TensorCopyNoSync(depth_img, depth_cpu);
	drawDepthImageCPU(depth_cpu);
}