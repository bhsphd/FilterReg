//
// Created by wei on 9/14/18.
//

#include "imgproc/frame_loader_file.h"

#include <boost/filesystem.hpp>
#include <vector_functions.h>


poser::FrameLoaderFile::FrameLoaderFile(
	std::string path, bool cpu_only,
	unsigned int img_rows, unsigned int img_cols,
	FeatureChannelType dense_depth_key, FeatureChannelType dense_rgba_key) :
	FrameLoaderBase(
		img_rows, img_cols,
		std::move(dense_depth_key), std::move(dense_rgba_key),
		cpu_only),
	data_path_(std::move(path)) {
	//Allocate the cv::Mat
	cv_depth_img_ = cv::Mat(img_rows, img_cols, CV_16UC1);
	cv_bgr_img_ = cv::Mat(img_rows, img_cols, CV_8UC3);
}

void poser::FrameLoaderFile::FetchDepthImageCPU(int frame_idx, poser::TensorSlice<unsigned short> depth_map) {
	//Do it
	auto file_path = FileNameVolumeDeform(frame_idx, true);
	cv_depth_img_ = cv::imread(file_path, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
	
	//Copy the data: this is somehow stupid
	cv::Mat wrapped(depth_map.Rows(), depth_map.Cols(), CV_16UC1, depth_map.RawPtr());
	cv_depth_img_.copyTo(wrapped);
}

void poser::FrameLoaderFile::FetchRGBImageCPU(int frame_idx, poser::TensorSlice<uchar4> rgba_map) {
	//The opencv mat is uchar3, and in bgr format
	auto file_path = FileNameVolumeDeform(frame_idx, false);
	cv_bgr_img_ = cv::imread(file_path, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
	
	//Convert it to rgba_map
	LOG_ASSERT(cv_bgr_img_.rows == rgba_map.Rows());
	LOG_ASSERT(cv_bgr_img_.cols == rgba_map.Cols());
	for(auto r_idx = 0; r_idx < rgba_map.Rows(); r_idx++) {
		for(auto c_idx = 0; c_idx < rgba_map.Cols(); c_idx++) {
			const uchar3 cv_bgr = cv_bgr_img_.at<uchar3>(r_idx, c_idx);
			rgba_map(r_idx, c_idx) = make_uchar4(cv_bgr.z, cv_bgr.y, cv_bgr.x, 255);
		}
	}
}


//The actual loader
std::string poser::FrameLoaderFile::FileNameVolumeDeform(int frame_idx, bool is_depth_img) const {
	using boost::filesystem::path;
	
	//Construct the file_name
	char frame_idx_str[20];
	sprintf(frame_idx_str, "%06d", frame_idx);
	std::string file_name = "frame-";
	file_name += std::string(frame_idx_str);
	if (is_depth_img) {
		file_name += ".depth";
	}
	else {
		file_name += ".color";
	}
	file_name += ".png";
	
	//Construct the path
	auto file_path = path(data_path_) / path(file_name);
	return file_path.string();
}