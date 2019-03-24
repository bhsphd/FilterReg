//
// Created by wei on 2/28/19.
//

#include "corr_search/nn_search/nn_search.h"
#include "corr_search/gmm/gmm.h"
#include "kinematic/rigid/rigid.h"
#include "visualizer/debug_visualizer.h"

#include <chrono>
#include <fstream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>


void load_test_data(
	const std::string& cloud_path,
	poser::FeatureMap& geometric_model,
	poser::FeatureMap& observation,
	poser::RigidKinematicModel& kinematic
) {
	using namespace poser;
	{
		const auto& full_cloud_path = cloud_path;
		
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
	
	//The observation
	{
		const auto& full_cloud_path = cloud_path;
		
		//Load the cloud
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
		LOG_ASSERT(pcl::io::loadPCDFile(full_cloud_path, *cloud) != -1);
		LOG(INFO) << "The size of the point cloud is " << cloud->points.size();
		
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
	
	//Random initial alignment
	const auto angle_error = 0.87f; //About 50 degree
	Eigen::Vector3f axis;
	axis.setRandom(); axis.normalize();
	LOG(INFO) << "The initial rotation error is " << angle_error * 180.0 / 3.24 << " degree w.r.t a random axis.";
	
	//The translational error
	const auto translation_error = 0.1;
	Eigen::Isometry3f eigen_rand_init_SE3(Eigen::AngleAxisf(angle_error, axis));
	axis.setRandom(); axis.normalize();
	eigen_rand_init_SE3.translation() = translation_error * axis; //About the length of bunny
	mat34 rand_init_SE3(eigen_rand_init_SE3);
	LOG(INFO) << "The initial translation error is " << translation_error << " meter w.r.t a random direction.";
	
	//The kinematic model
	kinematic.SetMotionParameter(rand_init_SE3);
	kinematic.CheckGeometricModelAndAllocateAttribute(geometric_model);
}

float compute_meansquare_error(
	const poser::FeatureMap &model,
	const poser::mat34 &estimated_pose,
	const poser::mat34 &gt_pose
) {
	//Get the reference vertex
	using namespace poser;
	auto reference_vertex_channel = CommonFeatureChannelKey::ReferenceVertex();
	const auto ref_vertex = model.GetTypedFeatureValueReadOnly<float4>(reference_vertex_channel, MemoryContext::CpuMemory);
	
	//Do it
	float accumlate_mse = 0.0f;
	float3 gt_vertex, estimated_vertex;
	for(auto i = 0; i < ref_vertex.Size(); i++) {
		const auto vertex_i = ref_vertex[i];
		estimated_vertex = estimated_pose.rotation() * vertex_i + estimated_pose.translation;
		gt_vertex = gt_pose.rotation() * vertex_i + gt_pose.translation;
		accumlate_mse += norm(estimated_vertex - gt_vertex);
	}
	
	//The return value
	return (accumlate_mse / ref_vertex.Size());
}

void process_pt2pt(const std::string& cloud_path) {
	using namespace poser;
	FeatureMap model, observation;
	RigidKinematicModel kinematic(MemoryContext::CpuMemory);
	load_test_data(cloud_path, model, observation, kinematic);
	
	//If you want to use fixed sigma, uncomment the GMMPermutohedralFixedSigma<3> one
	//A recommended sigma value would be 1/5~1/20 of the point cloud diameter.
	//For Stanford bunny, you might use sigma = 0.02 meter (2 centimeter).
	/*GMMPermutohedralFixedSigma<3> corr_search(
		CommonFeatureChannelKey::ObservationVertexCamera(),
		CommonFeatureChannelKey::LiveVertex());*/
	GMMPermutohedralUpdatedSigma corr_search(
		CommonFeatureChannelKey::ObservationVertexCamera(),
		CommonFeatureChannelKey::LiveVertex());
	
	//The target
	DenseGeometricTarget target;
	corr_search.CheckAndAllocateTarget(observation, model, target);
	
	//The assembler
	RigidPoint2PointTermAssemblerCPU point2point_assembler;
	point2point_assembler.CheckAndAllocate(model, kinematic, target);
	
	RigidPoint2PointKabsch kabsch;
	kabsch.CheckAndAllocate(model, kinematic, target);
	
	//The icp loop
	using namespace std::chrono;
	auto t1 = high_resolution_clock::now();
	
	//Build the lattice index
	const float init_guassian_sigma = 0.03f;
	const float min_gaussian_sigma = 0.002f;
	corr_search.UpdateObservation(observation, init_guassian_sigma);
	for(auto i = 0; i < 16; i++) {
		UpdateLiveVertexCPU(kinematic, model);
		
		//Update variance
		if(i >= 1) {
			auto sigma = corr_search.ComputeSigmaValue(model, target);
			if(!std::isnan(sigma) && sigma > min_gaussian_sigma)
				corr_search.UpdateObservation(observation, sigma);
		}
		
		//Compute target
		corr_search.ComputeTarget(observation, model, target);
		
		//Do kinematic
		mat34 transform;
		kabsch.ComputeTransformToTarget(model, kinematic, target, transform);
		kinematic.SetMotionParameter(transform);
	}
	
	//The time
	auto t2 = high_resolution_clock::now();
	LOG(INFO) << "The initial sigma value is " << init_guassian_sigma << " meter.";
	LOG(INFO) << "The running time is " << duration_cast<milliseconds>(t2 - t1).count() << " milliseconds.";
	
	//The error
	LOG(INFO) << "The final averaged alignment error per point is " <<
	          compute_meansquare_error(model, kinematic.GetRigidTransform(), mat34::identity()) << " m.";
	
	//Final update and draw the vertex here
	UpdateLiveVertexCPU(kinematic, model);
	auto live_vertex = model.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::LiveVertex(), MemoryContext::CpuMemory);
	auto depth_vertex = observation.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::ObservationVertexCamera(), MemoryContext::CpuMemory);
	DebugVisualizer::DrawMatchedCloudPair(live_vertex, depth_vertex);
}

int main(int argc, char* argv[]) {
	using namespace poser;
	LOG_ASSERT(argc == 2) << "Usage: ./rigid_pt2pt path/to/bunny_pcd.pcd";
	std::string cloud_path = argv[1];
	process_pt2pt(cloud_path);
}