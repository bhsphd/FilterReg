//
// Created by wei on 9/18/18.
//

#include "common/feature_channel_type.h"
#include "common/safe_call_utils.h"
#include "kinematic/articulated/articulated_kinematic_model.h"
#include "articulated_kinematic_model.h"

#include <cuda_runtime_api.h>

poser::ArticulatedKinematicModel::ArticulatedKinematicModel(
	poser::MemoryContext context,
	const RigidBodyTree<double>& kinematic_tree,
	poser::FeatureChannelType reference_vertex,
	poser::FeatureChannelType live_vertex,
	poser::FeatureChannelType reference_normal,
	poser::FeatureChannelType live_normal,
	poser::FeatureChannelType body_index
) : KinematicModelBase(
		context,
		std::move(reference_vertex),
		std::move(live_vertex),
		std::move(reference_normal),
		std::move(live_normal)),
	rigidbody_index_channel_(std::move(body_index)),
	kinematic_tree_(kinematic_tree.Clone()),
	kinematic_cache_(kinematic_tree.CreateKinematicsCacheWithType<double>())
{
	//Reset the buffer
	resetBufferForTree();
}


poser::ArticulatedKinematicModel::ArticulatedKinematicModel(
	const poser::ArticulatedKinematicModel &rhs
) : KinematicModelBase(rhs),
    rigidbody_index_channel_(rhs.rigidbody_index_channel_),
    kinematic_tree_(rhs.kinematic_tree_->Clone()),
    kinematic_cache_(rhs.kinematic_tree_->CreateKinematicsCacheWithType<double>())
{
	//Do allocation
	resetBufferForTree();
	
	//Do copying
	copyIndexAndTransform(rhs);
}

poser::ArticulatedKinematicModel& poser::ArticulatedKinematicModel::operator=(const poser::ArticulatedKinematicModel &rhs) {
	//First release the buffer
	releaseBuffer();
	
	//Copy the base types
	KinematicModelBase::operator=(rhs);
	rigidbody_index_channel_ = rhs.rigidbody_index_channel_;
	kinematic_tree_ = rhs.kinematic_tree_->Clone();
	kinematic_cache_ = rhs.kinematic_tree_->CreateKinematicsCacheWithType<double>();
	
	//Do allocation
	resetBufferForTree();
	
	//Do copying
	copyIndexAndTransform(rhs);
}

void poser::ArticulatedKinematicModel::resetBufferForTree() {
	//Check the default type
	if(!rigidbody_index_channel_.is_valid())
		rigidbody_index_channel_ = CommonFeatureChannelKey::ArticulatedRigidBodyIndex();
	
	//Reserve the space
	q_.resize(kinematic_tree_->get_num_positions());
	const auto max_num_bodies = kinematic_tree_->get_num_bodies();
	body2vertex_interval_map_.clear();
	body2vertex_interval_map_.reserve(max_num_bodies);
	body_transformation_.body2world_isometry.clear();
	body_transformation_.body2world_isometry.reserve(max_num_bodies);
	
	//Allocate the pagelocked memory
	cudaSafeCall(cudaMallocHost((void**)&(body_transformation_.body_transform_pagelocked), max_num_bodies * sizeof(mat34)));
	if(GetContext() == MemoryContext::GpuMemory) {
		cudaSafeCall(cudaMalloc((void**)&(body_transformation_.body_transform_dev), max_num_bodies * sizeof(mat34)));
	} else {
		body_transformation_.body_transform_dev = nullptr;
	}
}

void poser::ArticulatedKinematicModel::releaseBuffer() {
	cudaSafeCall(cudaFreeHost(body_transformation_.body_transform_pagelocked));
	if(GetContext() == MemoryContext::GpuMemory)
		cudaSafeCall(cudaFree(body_transformation_.body_transform_dev));
}

void poser::ArticulatedKinematicModel::copyIndexAndTransform(const poser::ArticulatedKinematicModel &rhs) {
	//Copy the index and transformation
	body2vertex_interval_map_ = rhs.body2vertex_interval_map_;
	q_ = rhs.q_;
	body_transformation_.body2world_isometry = rhs.body_transformation_.body2world_isometry;
	
	//Copy the device value
	const auto num_body = rhs.body_transformation_.body2world_isometry.size();
	memcpy(body_transformation_.body_transform_pagelocked, rhs.body_transformation_.body_transform_pagelocked, sizeof(mat34) * num_body);
	if(body_transformation_.body_transform_dev != nullptr) {
		cudaSafeCall(cudaMemcpy(
			body_transformation_.body_transform_dev,
			rhs.body_transformation_.body_transform_dev,
			sizeof(mat34) * num_body,
			cudaMemcpyDeviceToDevice));
	}
}

poser::ArticulatedKinematicModel::~ArticulatedKinematicModel() {
	releaseBuffer();
}


/* The method to build the body index
 */
void poser::ArticulatedKinematicModel::CheckGeometricModelAndAllocateAttribute(poser::FeatureMap &geometric_model) {
	//The default method
	KinematicModelBase::CheckGeometricModelAndAllocateAttribute(geometric_model);
	
	//Fetch the rigid body index and parse it
 	const auto body_index_view = geometric_model.GetTypedFeatureValueReadOnly<int>(
 		rigidbody_index_channel_,
 		MemoryContext::CpuMemory);
 	buildBody2VertexMap(geometric_model, body_index_view);
}


void poser::ArticulatedKinematicModel::buildBody2VertexMap(
	const poser::FeatureMap &geometric_model,
	const poser::TensorView<int> &body_index_view
) {
	LOG_ASSERT(geometric_model.GetDenseFeatureDim() == body_index_view.DimensionalSize());
	LOG_ASSERT(body_index_view.Size() > 1);
	
	//Pre-allocated variable
	int body_idx = body_index_view[0];
	unsigned start_idx = 0;
	for(unsigned i = 1; i < body_index_view.Size(); i++) {
		const auto current_body_idx = body_index_view[i];
		if(current_body_idx == body_idx)
			continue;
		
		//This i a new index, first check it
		for(const auto& body_elem : body2vertex_interval_map_){
			LOG_ASSERT(body_elem.body_index != body_idx) << "Duplicate body is not allowed";
		}
		
		//Insert into map
		RigidBodyElementMapType body2elem{body_idx, start_idx, i};
		body2vertex_interval_map_.emplace_back(body2elem);
		
		//Update the element
		body_idx = current_body_idx;
		start_idx = i;
	}
	
	//The last element
	for(const auto& body_elem : body2vertex_interval_map_){
		LOG_ASSERT(body_elem.body_index != body_idx) << "Duplicate body is not allowed";
	}
	RigidBodyElementMapType body2elem{body_idx, start_idx, body_index_view.Size()};
	body2vertex_interval_map_.emplace_back(body2elem);
	
	//Check the size
	LOG_ASSERT(body2vertex_interval_map_.size() <= kinematic_tree_->get_num_bodies());
	body_transformation_.body2world_isometry.resize(body2vertex_interval_map_.size());
	//LOG(INFO) << "The number of valid body is " << body2vertex_interval_map_.size();
}


/* The method related to kinematic of the robot and
 * the pose of each rigid body.
 */
void poser::ArticulatedKinematicModel::SetMotionParameter(const Eigen::Ref<const Eigen::VectorXd> &q) {
	LOG_ASSERT(q.rows() == kinematic_tree_->get_num_positions());
	q_ = q;
}

void poser::ArticulatedKinematicModel::UpdateWithDeltaQ(
	const Eigen::Ref<const Eigen::VectorXf> &dq,
	cudaStream_t stream
) {
	LOG_ASSERT(dq.rows() == kinematic_tree_->get_num_positions());
	for(auto i = 0; i < q_.size(); i++)
		q_(i) += static_cast<double>(dq(i));
	DoKinematicAndUpdateBodyPoseNoSync(stream);
}

void poser::ArticulatedKinematicModel::DoKinematicAndUpdateBodyPoseNoSync(cudaStream_t stream) {
	//Simple sanity check
	LOG_ASSERT(body2vertex_interval_map_.size() == body_transformation_.body2world_isometry.size());
	
	//Do kinematic
	kinematic_cache_.initialize(q_);
	kinematic_tree_->doKinematics(kinematic_cache_);
	
	//Update the body2world at host
	for(auto i = 0; i < body2vertex_interval_map_.size(); i++) {
		const auto body_idx = body2vertex_interval_map_[i].body_index;
		auto& body2world = body_transformation_.body2world_isometry[i];
		auto& body2world_mat = body_transformation_.body_transform_pagelocked[i];
		body2world = kinematic_tree_->relativeTransform(kinematic_cache_, 0, body_idx).cast<float>();
		body2world_mat = mat34(body2world);
	}
	
	//Upload to device if required
	if(body_transformation_.body_transform_dev != nullptr && GetContext() == MemoryContext::GpuMemory) {
		cudaSafeCall(cudaMemcpyAsync(
			body_transformation_.body_transform_dev,
			body_transformation_.body_transform_pagelocked,
			sizeof(mat34) * body_transformation_.body2world_isometry.size(),
			cudaMemcpyHostToDevice,
			stream
		));
	}
}


poser::TensorView<poser::mat34> poser::ArticulatedKinematicModel::GetBody2WorldTransformCPU() const {
	return TensorView<mat34>(
		body_transformation_.body_transform_pagelocked,
		TensorDim(body2vertex_interval_map_.size()),
		MemoryContext::CpuMemory
	);
}

poser::TensorView<poser::mat34> poser::ArticulatedKinematicModel::GetBody2WorldTransformGPU() const {
	return TensorView<mat34>(
		body_transformation_.body_transform_dev,
		TensorDim(body2vertex_interval_map_.size()),
		MemoryContext::GpuMemory
	);
}