//
// Created by wei on 12/9/18.
//

#include "estimation.h"
#include "ransac/ransac.h"
#include "corr_search/gmm/gmm.h"
#include "corr_search/nn_search/nn_search.h"
#include "visualizer/debug_visualizer.h"
#include "kinematic/affine/affine.h"
#include "geometry_utils/device2eigen.h"

#include "preprocessing.h"

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

#include <chrono>
#include <boost/filesystem.hpp>

void debug_visualize(const poser::FeatureMap& observation,
                     const poser::FeatureMap& model
) {
	//Draw it
	using namespace poser;
	auto live_vertex = model.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::LiveVertex(), MemoryContext::CpuMemory);
	auto depth_vertex = observation.GetTypedFeatureValueReadOnly<float4>(CommonFeatureChannelKey::ReferenceVertex(), MemoryContext::CpuMemory);
	DebugVisualizer::DrawMatchedCloudPair(live_vertex, depth_vertex);
	
	auto model_feature = model.GetFeatureValueReadOnly(ImageFeatureChannelKey::DONDescriptor(3), MemoryContext::CpuMemory);
	auto obs_feature = observation.GetFeatureValueReadOnly(ImageFeatureChannelKey::DONDescriptor(3), MemoryContext::CpuMemory);
	DebugVisualizer::DrawMatchedColorCloudPair(depth_vertex, obs_feature, live_vertex, model_feature);
}

poser::mat34 process_estimation_dense(
	const poser::FeatureMap& observation_map,
	poser::FeatureMap& geometric_model,
	poser::RigidKinematicModel& rigid_kinematic
) {
	using namespace poser;
	
	//The correspondence
	GMMPermutohedralFixedSigma<3> feature_corr(
		CommonFeatureChannelKey::ReferenceVertex(),
		ImageFeatureChannelKey::DONDescriptor(3),
		ImageFeatureChannelKey::DONDescriptor(3));
	feature_corr.SetOutlierConstant(0.6f);
	
	//The target for feature
	DenseGeometricTarget feature_target;
	feature_corr.CheckAndAllocateTarget(observation_map, geometric_model, feature_target);
	
	//Compute the target
	constexpr float sigma_feature = 0.07f;
	UpdateLiveVertexCPU(rigid_kinematic, geometric_model);
	feature_corr.UpdateObservation(observation_map, sigma_feature);
	feature_corr.ComputeTarget(observation_map, geometric_model, feature_target);
	
	//The pose estimator
	RigidPoint2PointKabsch kabsch;
	kabsch.CheckAndAllocate(geometric_model, rigid_kinematic, feature_target);
	mat34 transform;
	kabsch.ComputeTransformToTarget(geometric_model, rigid_kinematic, feature_target, transform);
	return transform;
}

poser::mat34 process_refine_rigid(
	const poser::FeatureMap& observation,
	poser::FeatureMap& model,
	poser::RigidKinematicModel& kinematic
) {
	using namespace poser;
	GMMPermutohedralUpdatedSigma corr_search(
		CommonFeatureChannelKey::ReferenceVertex(),
		CommonFeatureChannelKey::LiveVertex());
	auto init_pose = kinematic.GetRigidTransform();
	
	//The target
	DenseGeometricTarget target;
	corr_search.CheckAndAllocateTarget(observation, model, target);
	
	//Use point2point error now
	RigidPoint2PointKabsch kabsch;
	kabsch.CheckAndAllocate(model, kinematic, target);
	mat34 transform;
	
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
		
		//Compute the update
		kabsch.ComputeTransformToTarget(model, kinematic, target, transform);
		kinematic.SetMotionParameter(transform);
	}
	
	//The final update
	UpdateLiveVertexCPU(kinematic, model);
	
	//Restore the init pose
	auto refined_pose = kinematic.GetRigidTransform();
	kinematic.SetMotionParameter(init_pose);
	return refined_pose;
}


void proess_refine_affine(
	const std::vector<poser::mat34>& rigid_init,
	const poser::FeatureMap& target, poser::FeatureMap& source,
	const poser::FitnessEvaluator& evaluator,
	std::vector<poser::mat34>& affine_refined,
	std::vector<poser::FitnessEvaluator::Result>& affine_fitness
) {
	using namespace poser;
	AffineKinematicModel kinematic(CommonFeatureChannelKey::ObservationVertexWorld());
	kinematic.CheckGeometricModelAndAllocateAttribute(source);
	
	//The nn search
	TruncatedNN corr_search(CommonFeatureChannelKey::ReferenceVertex(), CommonFeatureChannelKey::LiveVertex());
	//GMMPermutohedralFixedSigma<3> corr_search(CommonFeatureChannelKey::ReferenceVertex(), CommonFeatureChannelKey::LiveVertex());
	DenseGeometricTarget corr_target;
	corr_search.CheckAndAllocateTarget(target, source, corr_target);
	
	//The affine part
	AffinePoint2PointAnalyticalCPU affine_estimator;
	affine_estimator.CheckAndAllocate(source, kinematic, corr_target);
	
	//Do it
	corr_search.UpdateObservation(target);
	
	//For different init
	affine_refined.clear(); affine_fitness.clear();
	for(auto i = 0; i < rigid_init.size(); i++) {
		kinematic.SetMotionParameter(rigid_init[i]);
		
		//The icp loop
		for(auto j = 0; j < 20; j++) {
			UpdateLiveVertexCPU(kinematic, source);
			
			//Compute target
			corr_search.ComputeTarget(target, source, corr_target);
			
			//Compute transform
			Eigen::Matrix4f transform;
			affine_estimator.ComputeAffineTransform(source, kinematic, corr_target, transform);
			kinematic.SetMotionParameter(transform);
		}
		
		//Write the result
		UpdateLiveVertexCPU(kinematic, source);
		affine_refined.emplace_back(mat34(kinematic.GetAffineTransformationMappedMatrix()));
		auto fitness_score = evaluator.Evaluate(source);
		
		//Threshold on kinemantic's singluar values
		float3 scale = kinematic.GetAffineTransformationSingularValues();
		const float threshold = 0.2f;
		if(std::abs(scale.x) < threshold || std::abs(scale.y) < threshold || std::abs(scale.z) < threshold) {
			fitness_score.fitness_score = 0.0f;
		}
		
		//The score
		affine_fitness.emplace_back(fitness_score);
	}
}

std::pair<poser::mat34, poser::mat34> perform_estimation_impl(poser::FeatureMap &source, const poser::FeatureMap &target, bool visualize) {
	//The kinematic model
	using namespace poser;
	RigidKinematicModel rigid_kinematic(MemoryContext::CpuMemory, CommonFeatureChannelKey::ObservationVertexWorld());
	rigid_kinematic.CheckGeometricModelAndAllocateAttribute(source);
	
	//The parameter for ransac
	RansacParameters parameters;
	parameters.n_point_per_sample = 3;
	parameters.n_samples = 1000;
	parameters.model_feature = ImageFeatureChannelKey::DONDescriptor(3);
	parameters.obs_feature = ImageFeatureChannelKey::DONDescriptor(3);
	parameters.model_reference_vertex = CommonFeatureChannelKey::ObservationVertexWorld();
	parameters.obs_vertex = CommonFeatureChannelKey::ReferenceVertex();
	
	//The fitness evaluator
	FitnessGeometryOnlyPermutohedral fitness_evaluator(
		CommonFeatureChannelKey::ReferenceVertex(),
		CommonFeatureChannelKey::LiveVertex());
	fitness_evaluator.SetCorrespondenceSigma(0.0012f);
	
	//FitnessGeometricOnlyKDTree fitness_evaluator;
	//fitness_evaluator.SetCorrespondenceThreshold(0.003f);
	
	//Do it
	LOG_ASSERT(target.ExistFeature(CommonFeatureChannelKey::ReferenceVertex(), MemoryContext::CpuMemory));
	fitness_evaluator.UpdateObservation(target);
	
	//OK
	RansacRigidFeature ransac(parameters, rigid_kinematic, fitness_evaluator);
	ransac.CheckAndAllocate(target, source);
	
	//Do it
	ransac.Compute(target, source);
	const auto k = 5;
	std::vector<mat34> best_k_pose; best_k_pose.resize(k);
	ransac.GetTopKHypothesisPose(k, best_k_pose);
	
	//Append the result from dense estimation
	best_k_pose.emplace_back(process_estimation_dense(target, source, rigid_kinematic));
	
	//Do rigid refinement on the best k
	const auto original_size = best_k_pose.size();
	for(auto i = 0; i < original_size; i++) {
		rigid_kinematic.SetMotionParameter(best_k_pose[i]);
		auto refined_pose = process_refine_rigid(target, source, rigid_kinematic);
		best_k_pose.emplace_back(refined_pose);
	}
	LOG_ASSERT(best_k_pose.size() == original_size * 2);
	
	//Do evaluation on rigid pose
	vector<FitnessEvaluator::Result> fitness_result;
	fitness_result.resize(best_k_pose.size());
	ransac.EvaluateHypothesisRigid(best_k_pose, fitness_result);
	
	//Functor to return the best pose
	auto best_pose_func = [](
		const vector<mat34>& pose_vec,
		const vector<FitnessEvaluator::Result>& fitness_vec
	) -> mat34 {
		LOG_ASSERT(pose_vec.size() == fitness_vec.size());
		int best_idx = -1;
		float best_fitness = 0.0f;
		for(auto i = 0; i < fitness_vec.size(); i++) {
			if(fitness_vec[i].fitness_score > best_fitness) {
				best_fitness = fitness_vec[i].fitness_score;
				best_idx = i;
			}
		}
		
		//Ok
		return pose_vec[best_idx];
	};
	
	//First the best rigid pose
	auto best_rigid_pose = best_pose_func(best_k_pose, fitness_result);
	
	//Do affine refinement
	std::vector<mat34> affine_refined_pose; affine_refined_pose.clear();
	std::vector<FitnessEvaluator::Result> affine_refined_fitness; affine_refined_fitness.clear();
	proess_refine_affine(best_k_pose, target, source, fitness_evaluator, affine_refined_pose, affine_refined_fitness);
	
	//Combine them and get the best one, on both
	for(auto i = 0; i < affine_refined_fitness.size(); i++) {
		best_k_pose.emplace_back(affine_refined_pose[i]);
		fitness_result.emplace_back(affine_refined_fitness[i]);
	}
	auto best_affine_pose = best_pose_func(best_k_pose, fitness_result);
	
	//The final setup on RIGID pose
	rigid_kinematic.SetMotionParameter(best_rigid_pose);
	UpdateLiveVertexCPU(rigid_kinematic, source);
	
	//Do visualization on RIGID pose
	if(visualize) {
		debug_visualize(target, source);
	}
	
	std::pair<mat34, mat34> rigid_affine_pair;
	rigid_affine_pair.first = best_rigid_pose;
	rigid_affine_pair.second = best_affine_pose;
	return rigid_affine_pair;
}

void save_point_cloud(
	const poser::FeatureMap& feature_map,
	const poser::FeatureChannelType& channel,
	const std::string& save_path
) {
	using namespace poser;
	const auto world_cloud = feature_map.GetTypedFeatureValueReadOnly<float4>(channel, MemoryContext::CpuMemory);
	
	//Do pcl cloud
	pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
	pcl_cloud.resize(world_cloud.Size());
	for(auto i = 0; i < world_cloud.Size(); i++) {
		const auto cloud_i = world_cloud[i];
		pcl::PointXYZ pcl_point;
		pcl_point.x = cloud_i.x;
		pcl_point.y = cloud_i.y;
		pcl_point.z = cloud_i.z;
		pcl_cloud[i] = pcl_point;
	}
	
	//Save it
	auto extension_type = boost::filesystem::extension(save_path);
	if(extension_type == ".pcd") {
		pcl::io::savePCDFileASCII(save_path, pcl_cloud);
	} else {
		pcl::PLYWriter writer;
		writer.write(save_path, pcl_cloud);
	}
}

std::pair<poser::mat34, poser::mat34> poser::perform_estimation(const poser::PoserRequestYaml &request) {
	FeatureMap target;
	load_geometric_template(target, request.template_path);
	
	//The image loading part
	FeatureMap source_img;
	load_depth_image(source_img, request.depth_img_path);
	load_segment_mask(source_img, request.foreground_mask_path);
	auto descriptor_channel = load_descriptor_image(source_img, request.descriptor_npy_path);
	
	//The image and point cloud processing part
	FeatureMap source_cloud;
	process_image(source_img, request.camera2world);
	process_cloud(source_img, descriptor_channel, source_cloud);
	auto source2target = perform_estimation_impl(source_cloud, target, request.visualize);
	
	//Save the processed cloud if requested
	if(!request.save_world_observation_cloud_path.empty()) {
		save_point_cloud(source_cloud, CommonFeatureChannelKey::ObservationVertexWorld(), request.save_world_observation_cloud_path);
	}
	
	//Save the template if requested
	if(!request.save_template_path.empty()) {
		save_point_cloud(target, CommonFeatureChannelKey::ReferenceVertex(), request.save_template_path);
	}
	
	//Need an inverse
	auto eigen_inverse = [](const mat34& source2target) -> mat34 {
		Eigen::Affine3f eigen_s2t; eigen_s2t.setIdentity();
		eigen_s2t.linear() = to_eigen(source2target.linear);
		eigen_s2t.translation() = to_eigen(source2target.translation);
		Eigen::Matrix4f eigen_trans = eigen_s2t.inverse().matrix();
		return mat34(eigen_trans);
	};
	
	decltype(source2target) inversed_s2t;
	inversed_s2t.first = eigen_inverse(source2target.first);
	inversed_s2t.second = eigen_inverse(source2target.second);
	return inversed_s2t;
}