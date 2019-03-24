//
// Created by wei on 11/29/18.
//

#include "imgproc/imgproc.h"
#include "cloudproc/cloudproc.h"
#include "geometry_utils/vector_operations.hpp"
#include "geometry_utils/permutohedral_common.h"


#include "corr_search/nn_search/nn_search.h"
#include "corr_search/gmm/gmm.h"
#include "kinematic/rigid/rigid.h"
#include "kinematic/affine/affine.h"
#include "visualizer/debug_visualizer.h"

#include <chrono>
#include <fstream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>


void load_test_data(
	poser::FeatureMap& geometric_model,
	poser::FeatureMap& observation,
	poser::AffineKinematicModel& kinematic
) {
	using namespace poser;
	{
		std::string model_file_name = "bunny.pcd";
		//std::string model_file_name = "bunny_model_noise_std_0.020000.pcd";
		//std::string model_file_name = "bunny_1.000000.pcd";
		std::string full_cloud_path = "/home/wei/Documents/programs/poser/data/robust_test/" + model_file_name;
		
		//Load the cloud
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
		LOG_ASSERT(pcl::io::loadPCDFile(full_cloud_path, *cloud) != -1);
		
		//Save to feature map
		geometric_model.AllocateDenseFeature(CommonFeatureChannelKey::ReferenceVertex(), TensorDim(cloud->points.size()));
		auto map_cloud = geometric_model.GetTypedFeatureValueReadWrite<float4>(CommonFeatureChannelKey::ReferenceVertex(), MemoryContext::CpuMemory);
		for(auto i = 0; i < cloud->points.size(); i++) {
			auto& map_cloud_i = map_cloud[i];
			const auto pcl_cloud_i = cloud->points[i];
			map_cloud_i.x = pcl_cloud_i.x;
			map_cloud_i.y = pcl_cloud_i.y;
			map_cloud_i.z = pcl_cloud_i.z;
			map_cloud_i.w = 1.0f;
		}
	}
	
	//The observation is corrupted by noise
	std::string obs_file_name = "bunny.pcd";
	//std::string obs_file_name = "bunny_1.000000.pcd";
	//std::string obs_file_name = "bunny_noise_std_0.020000.pcd";
	//std::string obs_file_name = "bunny_gaussian_1.000000.pcd";
	{
		std::string full_cloud_path = "/home/wei/Documents/programs/poser/data/robust_test/" + obs_file_name;
		
		//Load the cloud
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
		LOG_ASSERT(pcl::io::loadPCDFile(full_cloud_path, *cloud) != -1);
		//LOG(INFO) << "The size of observation point cloud is " << cloud->points.size();
		
		//Save to feature map
		observation.AllocateDenseFeature(CommonFeatureChannelKey::ObservationVertexCamera(), TensorDim(cloud->points.size()));
		auto map_cloud = observation.GetTypedFeatureValueReadWrite<float4>(CommonFeatureChannelKey::ObservationVertexCamera(), MemoryContext::CpuMemory);
		for(auto i = 0; i < cloud->points.size(); i++) {
			auto& map_cloud_i = map_cloud[i];
			const auto pcl_cloud_i = cloud->points[i];
			map_cloud_i.x = pcl_cloud_i.x;
			map_cloud_i.y = pcl_cloud_i.y;
			map_cloud_i.z = pcl_cloud_i.z;
			map_cloud_i.w = 1.0f;
		}
	}
	
	//Some random value
	const auto angle = 0.1f;
	Eigen::Vector3f axis = Eigen::Vector3f::UnitZ();
	Eigen::Isometry3f eigen_rand_init_SE3(Eigen::AngleAxisf(angle, axis));
	Eigen::Affine3f affine_init = eigen_rand_init_SE3 * Eigen::Scaling(Eigen::Vector3f(1.8, 1.2, 1.1));
	
	//The kinematic model
	Eigen::Matrix4f rand_init_trans = affine_init.matrix();
	kinematic.SetMotionParameter(rand_init_trans);
	kinematic.CheckGeometricModelAndAllocateAttribute(geometric_model);
	LOG(INFO) << rand_init_trans;
}

void process_icp() {
	using namespace poser;
	FeatureMap model, observation;
	AffineKinematicModel kinematic;
	load_test_data(model, observation, kinematic);
	
	//Do it
	//TrimmedNN corr_search(CommonFeatureChannelKey::ObservationVertexCamera(), CommonFeatureChannelKey::LiveVertex());
	//corr_search.SetTruncatedDistance(0.05f);
	//corr_search.SetTrimmedRatio(0.75f);
	/*GMMPermutohedralFixedSigma<3> corr_search(
		CommonFeatureChannelKey::ObservationVertexCamera(),
		CommonFeatureChannelKey::LiveVertex());*/
	GMMPermutohedralUpdatedSigma corr_search(
		CommonFeatureChannelKey::ObservationVertexCamera(),
		CommonFeatureChannelKey::LiveVertex());
	
	//The target
	DenseGeometricTarget target;
	corr_search.CheckAndAllocateTarget(observation, model, target);
	
	//The affine part
	AffinePoint2PointAnalyticalCPU affine_estimator;
	affine_estimator.CheckAndAllocate(model, kinematic, target);
	
	//Do it
	corr_search.UpdateObservation(observation, 0.02f);
	for(auto i = 0; i < 20; i++) {
		UpdateLiveVertexCPU(kinematic, model);
		
		//Update variance
		if(i >= 1) {
			auto sigma = corr_search.ComputeSigmaValue(model, target);
			//LOG(INFO) << "The sigma value is " << sigma;
			if(!std::isnan(sigma) && sigma > 0.002f)
				corr_search.UpdateObservation(observation, sigma);
		}
		
		//Compute target
		corr_search.ComputeTarget(observation, model, target);
		
		//Compute transform
		Eigen::Matrix4f transform;
		affine_estimator.ComputeAffineTransform(model, kinematic, target, transform);
		kinematic.SetMotionParameter(transform);
		
		/*if(i % 5 == 0) {
			auto live_vertex = model.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::LiveVertex(), MemoryContext::CpuMemory);
			auto depth_vertex = observation.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::ObservationVertexCamera(), MemoryContext::CpuMemory);
			DebugVisualizer::DrawMatchedCloudPair(live_vertex, depth_vertex);
		}*/
	}
	
	//Final update here
	UpdateLiveVertexCPU(kinematic, model);
	auto live_vertex = model.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::LiveVertex(), MemoryContext::CpuMemory);
	auto depth_vertex = observation.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::ObservationVertexCamera(), MemoryContext::CpuMemory);
	DebugVisualizer::DrawMatchedCloudPair(live_vertex, depth_vertex);
}

int main() {
	using namespace poser;
	process_icp();
}