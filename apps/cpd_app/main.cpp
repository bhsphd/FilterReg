//
// Created by wei on 1/17/19.
//

#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <glog/logging.h>

#include "common/feature_map.h"
#include "corr_search/gmm/gmm.h"
#include "corr_search/nn_search/nn_search.h"
#include "kinematic/cpd/cpd_point2point.h"
#include "kinematic/cpd/cpd_geometry_update.h"
#include "visualizer/debug_visualizer.h"


void load_and_sample_data(
	const std::string& full_cloud_path,
	float voxel_resolution,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	bool centering = false,
	float offset = 0.0f
) {
	//The cloud at full resolution
	pcl::PointCloud<pcl::PointXYZ>::Ptr full_cloud(new pcl::PointCloud<pcl::PointXYZ>());
	LOG_ASSERT(pcl::io::loadPCDFile(full_cloud_path, *full_cloud) != -1);
	
	//Do subsampling
	pcl::VoxelGrid<pcl::PointXYZ> sor;
	sor.setInputCloud(full_cloud);
	sor.setLeafSize(voxel_resolution, voxel_resolution, voxel_resolution);
	cloud->clear();
	sor.filter(*cloud);
	
	//Do centering on the cloud
	pcl::PointXYZ center;
	center.x = center.y = center.z = 0.0f;
	
	//Whether or not we actually use centering
	if(centering) {
		for(auto i = 0; i < cloud->size(); i++) {
			auto cloud_i = cloud->points[i];
			center.x += cloud_i.x;
			center.y += cloud_i.y;
			center.z += cloud_i.z;
		}
		center.x /= float(cloud->size());
		center.y /= float(cloud->size());
		center.z /= float(cloud->size());
	}
	
	//Apply it
	for(auto i = 0; i < cloud->size(); i++) {
		auto& cloud_i = cloud->points[i];
		cloud_i.x -= center.x + offset;
		cloud_i.y -= center.y + offset;
		cloud_i.z -= center.z + offset;
	}
	
	//Output the size of the cloud
	LOG(INFO) << "The number of subsampled cloud is " << cloud->size();
	
	//Take a look
	//using namespace poser;
	//DebugVisualizer::DrawPointCloud(cloud);
}

void copy_vertex_data(
	const pcl::PointCloud<pcl::PointXYZ>& pcl_cloud,
	poser::FeatureMap& cloud,
	const poser::FeatureChannelType& channel
) {
	using namespace poser;
	cloud.AllocateDenseFeature(channel, pcl_cloud.size(), MemoryContext::CpuMemory);
	auto slice = cloud.GetTypedFeatureValueReadWrite<float4>(channel, MemoryContext::CpuMemory);
	LOG_ASSERT(pcl_cloud.size() == slice.Size());
	
	for(auto i = 0; i < slice.Size(); i++) {
		auto& slice_i = slice[i];
		auto cloud_i = pcl_cloud.points[i];
		slice_i.x = cloud_i.x;
		slice_i.y = cloud_i.y;
		slice_i.z = cloud_i.z;
		slice_i.w = 1.0f;
	}
}

void dbg_visualize(const poser::FeatureMap &model, const poser::FeatureMap &target) {
	using namespace poser;
	auto live_vertex = model.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::LiveVertex(), MemoryContext::CpuMemory);
	auto depth_vertex = target.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::ObservationVertexCamera(), MemoryContext::CpuMemory);
	
	//DebugVisualizer::DrawPointCloud(live_vertex, 1000.0f);
	DebugVisualizer::DrawMatchedCloudPair(live_vertex, depth_vertex);
}

void test_cpd(
	const pcl::PointCloud<pcl::PointXYZ>& pcl_model,
	const pcl::PointCloud<pcl::PointXYZ>& pcl_observation
) {
	//Construct the feature map
	using namespace poser;
	FeatureMap model, observation;
	copy_vertex_data(pcl_model, model, CommonFeatureChannelKey::ReferenceVertex());
	copy_vertex_data(pcl_observation, observation, CommonFeatureChannelKey::ObservationVertexCamera());
	
	//The kinematic model
	CoherentPointDriftKinematic kinematic(0.015f);
	kinematic.CheckGeometricModelAndAllocateAttribute(model);
	UpdateLiveVertex(kinematic, model);
	
	//Correspondence finder
	GMMPermutohedralUpdatedSigma corr_search(
		CommonFeatureChannelKey::ObservationVertexCamera(),
		CommonFeatureChannelKey::LiveVertex());
	/*TruncatedNN corr_search(
		CommonFeatureChannelKey::ObservationVertexCamera(),
		CommonFeatureChannelKey::LiveVertex());
	corr_search.SetTruncatedDistance(0.03f);*/
	DenseGeometricTarget target;
	corr_search.CheckAndAllocateTarget(observation, model, target);
	
	//The matrix cache
	const auto n_point = model.GetDenseFeatureDim().total_size();
	Eigen::MatrixXf A, b, w_cache;
	A.resize(n_point, n_point);
	b.resize(n_point, 3);
	w_cache.resize(n_point, 3);
	
	//The icp loop
	const float sigma_init = 0.01f;
	const float lamda_sigmasquare = 0.1f;
	corr_search.UpdateObservation(observation, sigma_init);
	for(auto i = 0; i < 10; i++) {
		UpdateLiveVertex(kinematic, model);
		
		//Update variance
		if(i >= 1) {
			auto sigma = corr_search.ComputeSigmaValue(model, target);
			LOG(INFO) << "The sigma value is " << sigma;
			if(!std::isnan(sigma) && sigma > 0.001f)
				corr_search.UpdateObservation(observation, sigma);
		}
		
		//Compute target
		corr_search.ComputeTarget(observation, model, target);
		
		//Assemble the matrix
		ConstructLinearEquationDense(kinematic, target, lamda_sigmasquare, A, b);
		
		//Solve and update
		w_cache = A.colPivHouseholderQr().solve(b);
		kinematic.SetMotionParameterW(w_cache);
	}
	
	//Do visualization
	UpdateLiveVertex(kinematic, model);
	dbg_visualize(model, observation);
}

int main() {
	auto model_pcd_path = "/home/wei/Coding/poser/data/tmp/shoe_model.pcd";
	auto obs_pcd_path = "/home/wei/Coding/poser/data/tmp/shoe_obs.pcd";
	float resolution_model = 0.01f;
	float resolution_obs = 0.001f;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_model(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_obs(new pcl::PointCloud<pcl::PointXYZ>());
	load_and_sample_data(model_pcd_path, resolution_model, cloud_model);
	load_and_sample_data(obs_pcd_path, resolution_obs, cloud_obs);
	
	//Do test
	test_cpd(*cloud_model, *cloud_obs);
}