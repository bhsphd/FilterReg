//
// Created by wei on 9/14/18.
//

#pragma once

//#include "imgproc/image_loader_interface.h"
#include "imgproc/frame_loader_base.h"
#include <opencv2/opencv.hpp>

namespace poser {
	
	class FrameLoaderFile : public FrameLoaderBase {
	public:
		explicit FrameLoaderFile(
			std::string path, bool cpu_only = true,
			unsigned img_rows = 480, unsigned img_cols = 640,
			FeatureChannelType dense_depth_key = FeatureChannelType(),
			FeatureChannelType dense_rgba_key = FeatureChannelType());
		~FrameLoaderFile() override = default;
		
		//The loader interface
	protected:
		void FetchDepthImageCPU(int frame_idx, TensorSlice<unsigned short> depth_map) override;
		void FetchRGBImageCPU(int frame_idx, TensorSlice<uchar4> rgba_map) override;
	
	private:
		cv::Mat cv_depth_img_;
		cv::Mat cv_bgr_img_; //The bgr image format used by opencv
		std::string data_path_;
		
		//Get the string name
		std::string FileNameVolumeDeform(int frame_idx, bool is_depth_img) const;
	};
	
}
