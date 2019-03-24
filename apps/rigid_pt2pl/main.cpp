//
// Created by wei on 2/28/19.
//

#include "corr_search/gmm/gmm.h"
#include "corr_search/nn_search/nn_search.h"
#include "kinematic/rigid/rigid.h"
#include "visualizer/debug_visualizer.h"

#include <glog/logging.h>
#include <pcl/io/pcd_io.h>
#include <chrono>

void load_feature_map(poser::FeatureMap& feature_map, const std::string& pcd_cloud_path) {
	//Load the cloud
	pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZINormal>());
	LOG_ASSERT(pcl::io::loadPCDFile(pcd_cloud_path, *cloud) != -1);
	
	//Save to feature map
	using namespace poser;
	feature_map.AllocateDenseFeature(
		CommonFeatureChannelKey::ObservationVertexCamera(),
		TensorDim(cloud->points.size()));
	feature_map.AllocateDenseFeature(
		CommonFeatureChannelKey::ObservationNormalCamera(),
		TensorDim(cloud->points.size()));
	
	//Map to writeable
	auto map_vertex = feature_map.GetTypedFeatureValueReadWrite<float4>(
		CommonFeatureChannelKey::ObservationVertexCamera(),
		MemoryContext::CpuMemory);
	auto map_normal = feature_map.GetTypedFeatureValueReadWrite<float4>(
		CommonFeatureChannelKey::ObservationNormalCamera(),
		MemoryContext::CpuMemory);
	
	//The iteration
	for(auto i = 0; i < cloud->points.size(); i++) {
		auto& map_cloud_i = map_vertex[i];
		auto& map_normal_i = map_normal[i];
		const auto pcl_cloud_i = cloud->points[i];
		
		map_cloud_i.x = pcl_cloud_i.x;
		map_cloud_i.y = pcl_cloud_i.y;
		map_cloud_i.z = pcl_cloud_i.z;
		map_cloud_i.w = 1.0f;
		
		map_normal_i.x = pcl_cloud_i.normal_x;
		map_normal_i.y = pcl_cloud_i.normal_y;
		map_normal_i.z = pcl_cloud_i.normal_z;
		map_normal_i.w = 0;
	}
}


void process_pt2pl(
	const std::string& model_path,
	const std::string& obs_path
) {
	using namespace poser;
	FeatureMap observation, model;
	load_feature_map(model, model_path);
	load_feature_map(observation, obs_path);
	LOG(INFO) << "The number of points in model is " << model.GetDenseFeatureDim().total_size();
	LOG(INFO) << "The number of points in observation is " << observation.GetDenseFeatureDim().total_size();
	
	//The kinematic model
	RigidKinematicModel kinematic(MemoryContext::CpuMemory, CommonFeatureChannelKey::ObservationVertexCamera());
	kinematic.CheckGeometricModelAndAllocateAttribute(model);
	
	//The initial pose
	kinematic.SetMotionParameter(mat34::identity());
	
	//The correspondence finder
	GMMPermutohedralFixedSigmaPt2Pl<3> corr_search(
		CommonFeatureChannelKey::ObservationVertexCamera(),
		CommonFeatureChannelKey::ObservationNormalCamera(),
		CommonFeatureChannelKey::LiveVertex());
	/*GMMPermutohedralUpdatedSigmaPt2Pl corr_search(
		CommonFeatureChannelKey::ObservationVertexCamera(),
		CommonFeatureChannelKey::ObservationNormalCamera(),
		CommonFeatureChannelKey::LiveVertex());*/
	DenseGeometricTarget target;
	corr_search.CheckAndAllocateTarget(observation, model, target);
	
	//The assembler
	Eigen::Matrix6f JtJ_point2plane, JtJ_point2point, JtJ;
	Eigen::Vector6f Jte_point2plane, Jte_point2point, Jte, d_twist;
	RigidPoint2PlaneTermAssemblerCPU point2plane;
	RigidPoint2PointTermAssemblerCPU point2point;
	point2plane.CheckAndAllocate(model, kinematic, target);
	point2point.CheckAndAllocate(model, kinematic, target);
	
	//The processing loop
	using namespace std::chrono;
	auto t1 = high_resolution_clock::now();
	
	//Build index
	UpdateLiveVertexCPU(kinematic, model);
	float gaussian_sigma = 0.08f; // 8cm
	corr_search.UpdateObservation(observation, gaussian_sigma);
	for(auto i = 0; i < 15; i++) {
		//Might need to update the variance
		/*if(i >= 1) {
			float sigma = corr_search.ComputeSigmaValue(model, target);
			if(!(std::isnan(sigma) || sigma < 0.02f)) {
				corr_search.UpdateObservation(observation, sigma);
			}
		}*/
		
		//Compute the target
		corr_search.ComputeTarget(observation, model, target);
		
		//Zero out
		JtJ.setZero(); Jte.setZero();
		JtJ_point2plane.setZero();
		Jte_point2plane.setZero();
		JtJ_point2point.setZero();
		Jte_point2point.setZero();
		
		//Assemble the matrix
		point2plane.ProcessAssemble(model, kinematic, target, JtJ_point2plane, Jte_point2plane);
		//point2point.ProcessAssemble(model, kinematic, target, JtJ_point2point, Jte_point2point);
		
		//Solve it
		JtJ = JtJ_point2plane + JtJ_point2point;
		Jte = Jte_point2plane + Jte_point2point;
		d_twist = JtJ.ldlt().solve(Jte);
		
		//Update
		kinematic.UpdateWithTwist(d_twist);
		UpdateLiveVertexCPU(kinematic, model);
	}
	
	auto t2 = high_resolution_clock::now();
	LOG(INFO) << "The time is " << duration_cast<milliseconds>(t2 - t1).count();
	
	//Draw it
	auto live_vertex = model.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::LiveVertex(), MemoryContext::CpuMemory);
	auto depth_vertex = observation.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::ObservationVertexCamera(), MemoryContext::CpuMemory);
	DebugVisualizer::DrawMatchedCloudPair(live_vertex, depth_vertex);
}


int main(int argc, char* argv[]) {
	LOG_ASSERT(argc == 3) << "Usage: ./rigid_pt2pl /path/to/cloud_0.pcd /path/to/cloud_1.pcd";
	std::string cloud_0_path = argv[1];
	std::string cloud_1_path = argv[2];
	process_pt2pl(cloud_0_path, cloud_1_path);
}