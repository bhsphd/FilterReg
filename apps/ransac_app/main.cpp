//
// Created by wei on 12/3/18.
//

#include "imgproc/imgproc.h"
#include "corr_search/gmm/gmm.h"
#include "corr_search/nn_search/nn_search.h"
#include "corr_search/fast_global_registration.h"
#include "corr_search/target_reweight.h"
#include "kinematic/rigid/rigid.h"
#include "kinematic/affine/affine.h"
#include "ransac/ransac.h"
#include "visualizer/debug_visualizer.h"

#include <pcl/io/pcd_io.h>
#include <chrono>

void load_observation_pcd(poser::FeatureMap& feature_map) {
	//The path of cloud
	//std::string pcd_dir = "/home/wei/data/pdc/logs_shoes/2018-05-15-00-28-06/processed/pcl-cloud/";
	//std::string pcd_path = pcd_dir + "frame-000000.pcd";
	std::string pcd_dir = "/home/wei/data/pdc/logs_shoes/2018-11-16-17-43-27/processed/";
	std::string pcd_path = pcd_dir + "descriptor_template.pcd";
	
	//First load the point cloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
	if(pcl::io::loadPCDFile<pcl::PointXYZ> (pcd_path, *cloud) == -1) {
		PCL_ERROR ("Couldn't read vertex \n");
	}
	
	//Next load the normal cloud
	pcl::PointCloud<pcl::Normal>::Ptr normal(new pcl::PointCloud<pcl::Normal>());
	if(pcl::io::loadPCDFile<pcl::Normal>(pcd_path, *normal) == -1) {
		PCL_ERROR ("Couldn't read normal \n");
	}
	
	//Check the size
	LOG_ASSERT(normal->points.size() == cloud->points.size());
	
	//Convert to feature map
	using namespace poser;
	feature_map.AllocateDenseFeature(CommonFeatureChannelKey::ObservationVertexCamera(), cloud->points.size(), MemoryContext::CpuMemory);
	feature_map.AllocateDenseFeature(ImageFeatureChannelKey::DONDescriptor(3), MemoryContext::CpuMemory);
	auto vertex_feature = feature_map.GetTypedFeatureValueReadWrite<float4>(CommonFeatureChannelKey::ObservationVertexCamera(), MemoryContext::CpuMemory);
	auto don_feature = feature_map.GetTypedFeatureValueReadWrite<float3>(ImageFeatureChannelKey::DONDescriptor(3), MemoryContext::CpuMemory);
	
	//Get the feature
	for(auto i = 0; i < cloud->points.size(); i++) {
		auto& vertex_i = vertex_feature[i];
		const auto& pcl_point_i = cloud->points[i];
		vertex_i = make_float4(pcl_point_i.x, pcl_point_i.y, pcl_point_i.z, 1.0f);
		
		//The normal
		auto& don_feature_i = don_feature[i];
		const auto& pcl_normal_i = normal->points[i];
		don_feature_i = make_float3(pcl_normal_i.normal_x, pcl_normal_i.normal_y, pcl_normal_i.normal_z);
	}
}

void load_model_pcd(poser::FeatureMap& feature_map) {
	//The path of cloud
	std::string pcd_dir = "/home/wei/data/pdc/logs_shoes/2018-05-17-00-56-44_000000/";
	std::string pcd_path = pcd_dir + "shoe_3.pcd";
	
	//First load the point cloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
	if(pcl::io::loadPCDFile<pcl::PointXYZ> (pcd_path, *cloud) == -1) {
		PCL_ERROR ("Couldn't read vertex \n");
	}
	
	//Next load the normal cloud
	pcl::PointCloud<pcl::Normal>::Ptr normal(new pcl::PointCloud<pcl::Normal>());
	if(pcl::io::loadPCDFile<pcl::Normal>(pcd_path, *normal) == -1) {
		PCL_ERROR ("Couldn't read normal \n");
	}
	
	//Check the size
	LOG_ASSERT(normal->points.size() == cloud->points.size());
	
	//Convert to feature map
	using namespace poser;
	feature_map.AllocateDenseFeature(CommonFeatureChannelKey::ReferenceVertex(), cloud->points.size(), MemoryContext::CpuMemory);
	feature_map.AllocateDenseFeature(ImageFeatureChannelKey::DONDescriptor(3), MemoryContext::CpuMemory);
	auto vertex_feature = feature_map.GetTypedFeatureValueReadWrite<float4>(CommonFeatureChannelKey::ReferenceVertex(), MemoryContext::CpuMemory);
	auto don_feature = feature_map.GetTypedFeatureValueReadWrite<float3>(ImageFeatureChannelKey::DONDescriptor(3), MemoryContext::CpuMemory);
	
	//Get the feature
	for(auto i = 0; i < cloud->points.size(); i++) {
		auto& vertex_i = vertex_feature[i];
		const auto& pcl_point_i = cloud->points[i];
		vertex_i = make_float4(pcl_point_i.x, pcl_point_i.y, pcl_point_i.z, 1.0f);
		
		//The normal
		auto& don_feature_i = don_feature[i];
		const auto& pcl_normal_i = normal->points[i];
		don_feature_i = make_float3(pcl_normal_i.normal_x, pcl_normal_i.normal_y, pcl_normal_i.normal_z);
	}
}

void process_ransac(
	const poser::FeatureMap& observation,
	poser::FeatureMap& model,
	poser::RigidKinematicModel& kinematic
) {
	using namespace poser;
	using namespace std::chrono;
	
	//The parameters
	RansacParameters parameters;
	parameters.n_point_per_sample = 3;
	parameters.n_samples = 1000;
	parameters.model_feature = ImageFeatureChannelKey::DONDescriptor(3);
	parameters.obs_feature = ImageFeatureChannelKey::DONDescriptor(3);
	parameters.model_reference_vertex = CommonFeatureChannelKey::ReferenceVertex();
	parameters.obs_vertex = CommonFeatureChannelKey::ObservationVertexCamera();
	
	//The fitness evaluator
	FitnessGeometryOnlyPermutohedral fitness_evaluator;
	fitness_evaluator.SetCorrespondenceSigma(0.0012f);
	
	//FitnessGeometricOnlyKDTree fitness_evaluator;
	//fitness_evaluator.SetCorrespondenceThreshold(0.003f);
	fitness_evaluator.UpdateObservation(observation);
	
	//OK
	RansacRigidFeature ransac(parameters, kinematic, fitness_evaluator);
	
	//Do it
	ransac.CheckAndAllocate(observation, model);
	auto t1 = high_resolution_clock::now();
	ransac.Compute(observation, model);
	auto t2 = high_resolution_clock::now();
	LOG(INFO) << "The passed time is " << duration_cast<milliseconds>(t2 - t1).count();
	
	//Get the best k pose
	auto best_pose = ransac.GetBestHypothesisPose();
	kinematic.SetMotionParameter(best_pose);
	UpdateLiveVertexCPU(kinematic, model);
}

void process_refine(
	const poser::FeatureMap& observation,
	poser::FeatureMap& model,
	poser::RigidKinematicModel& kinematic
) {
	using namespace poser;
	GMMPermutohedralUpdatedSigma corr_search(
		CommonFeatureChannelKey::ObservationVertexCamera(),
		CommonFeatureChannelKey::LiveVertex());
	
	//The target
	DenseGeometricTarget target;
	corr_search.CheckAndAllocateTarget(observation, model, target);
	
	//Use point2point error now
	RigidPoint2PointTermAssemblerCPU point2point;
	point2point.CheckAndAllocate(model, kinematic, target);
	
	//Do processing
	Eigen::Matrix6f JtJ;
	Eigen::Vector6f Jt_error;
	Eigen::Vector6f d_twist;
	
	//The icp loop
	corr_search.UpdateObservation(observation, 0.01f);
	for(auto i = 0; i < 20; i++) {
		UpdateLiveVertexCPU(kinematic, model);
		
		//Update variance
		if(i >= 1) {
			auto sigma = corr_search.ComputeSigmaValue(model, target);
			if(!std::isnan(sigma) && sigma > 0.002f)
				corr_search.UpdateObservation(observation, sigma);
		}
		
		//Compute target
		corr_search.ComputeTarget(observation, model, target);
		
		//Assemble the matrix
		point2point.ProcessAssemble(
			model, kinematic, target, JtJ, Jt_error);
		
		//Solve it and update
		d_twist = JtJ.ldlt().solve(Jt_error);
		kinematic.UpdateWithTwist(d_twist);
	}
	
	//The final update
	UpdateLiveVertexCPU(kinematic, model);
}

void debug_visualize(const poser::FeatureMap& observation,
                     const poser::FeatureMap& model,
                     const poser::RigidKinematicModel& kinematic
) {
	//Draw it
	using namespace poser;
	auto live_vertex = model.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::LiveVertex(), MemoryContext::CpuMemory);
	auto depth_vertex = observation.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::ObservationVertexCamera(), MemoryContext::CpuMemory);
	DebugVisualizer::DrawMatchedCloudPair(live_vertex, depth_vertex);
	
	auto model_feature = model.GetFeatureValueReadOnly(ImageFeatureChannelKey::DONDescriptor(3), MemoryContext::CpuMemory);
	auto obs_feature = observation.GetFeatureValueReadOnly(ImageFeatureChannelKey::DONDescriptor(3), MemoryContext::CpuMemory);
	DebugVisualizer::DrawMatchedColorCloudPair(depth_vertex, obs_feature, live_vertex, model_feature);
}

void process_rigid() {
	//Load the geometric model
	using namespace poser;
	FeatureMap geometric_model;
	load_model_pcd(geometric_model);
	
	FeatureMap observation_map;
	load_observation_pcd(observation_map);
	
	//The kinematic model
	RigidKinematicModel rigid_kinematic(MemoryContext::CpuMemory);
	rigid_kinematic.CheckGeometricModelAndAllocateAttribute(geometric_model);
	
	//Do it
	process_ransac(observation_map, geometric_model, rigid_kinematic);
	process_refine(observation_map, geometric_model, rigid_kinematic);
	debug_visualize(observation_map, geometric_model, rigid_kinematic);
}

int main() {
	process_rigid();
}