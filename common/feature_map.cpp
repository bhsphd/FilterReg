//
// Created by wei on 9/10/18.
//

#include "common/feature_map.h"
#include "common/data_transfer.h"
#include "feature_map.h"

poser::FeatureMap::FeatureMap() {
	const auto maxnum_channel = 50;
	
	//Reserve the space for dense feature
	dense_feature_map_.feature2offset_map_.reserve(maxnum_channel);
	dense_feature_map_.feature_value_.reserve(maxnum_channel);
	dense_feature_map_.dense_feature_dim_ = TensorDim(0);
	dense_feature_map_.dense_feature_capacity_ = 0;
	
	//For sparse ones
	sparse_feature_map_.feature2offset_map_.reserve(maxnum_channel);
	sparse_feature_map_.feature_value_.reserve(maxnum_channel);
	sparse_feature_map_.feature_index_.reserve(maxnum_channel);
}

void poser::FeatureMap::CloneTo(poser::FeatureMap &other) const {
	//The dense part
	other.dense_feature_map_.feature2offset_map_ = dense_feature_map_.feature2offset_map_;
	other.dense_feature_map_.dense_feature_dim_ = dense_feature_map_.dense_feature_dim_;
	other.dense_feature_map_.dense_feature_capacity_ = dense_feature_map_.dense_feature_capacity_;
	
	//Clone all the value
	auto& dense_value = other.dense_feature_map_.feature_value_;
	dense_value.clear();
	for(auto i = 0; i < dense_feature_map_.feature_value_.size(); i++) {
		dense_value.emplace_back(TensorBlob());
		auto& blob = dense_value.back();
		dense_feature_map_.feature_value_[i].CloneTo(blob);
	}
	
	//The sparse part
	other.sparse_feature_map_.feature2offset_map_ = sparse_feature_map_.feature2offset_map_;
	auto& sparse_value = other.sparse_feature_map_.feature_value_; sparse_value.clear();
	auto& sparse_index = other.sparse_feature_map_.feature_index_; sparse_index.clear();
	for(auto i = 0; i < sparse_feature_map_.feature_value_.size(); i++) {
		sparse_value.emplace_back(TensorBlob());
		auto& value_blob = sparse_value.back();
		sparse_feature_map_.feature_value_[i].CloneTo(value_blob);
		
		sparse_index.emplace_back(TensorBlob());
		auto& index_blob = sparse_index.back();
		sparse_feature_map_.feature_index_[i].CloneTo(index_blob);
	}
}

/* The method for allocation
 */
bool poser::FeatureMap::allocateDenseFeature(
	const string& feature_name_key,
	unsigned short byte_size,
	const TensorDim& tensor_dim,
	MemoryContext context,
	std::size_t typed_capacity,
	unsigned short channel_valid_byte
) {
	//Handle default argument
	if(typed_capacity == 0)
		typed_capacity = tensor_dim.total_size();
	
	//There exist this feature, return false
	const auto equal_range = dense_feature_map_.feature2offset_map_.equal_range(feature_name_key);
	for(auto iter = equal_range.first; iter != equal_range.second; iter++) {
		//Need to check both type and context
		int index = iter->second;
		if(dense_feature_map_.feature_value_[index].GetMemoryContext() == context)
			return false;
	}
	
	//Check or update the size
	if(dense_feature_map_.dense_feature_dim_.total_size() == 0) {
		dense_feature_map_.dense_feature_dim_ = tensor_dim;
		dense_feature_map_.dense_feature_capacity_ = typed_capacity;
	}
	LOG_ASSERT(tensor_dim == dense_feature_map_.dense_feature_dim_) << "The feature dim should matched in dense feature";
	LOG_ASSERT(typed_capacity <= dense_feature_map_.dense_feature_capacity_);
	typed_capacity = dense_feature_map_.dense_feature_capacity_;
	
	//This is new feature, insert into map
	const auto index = dense_feature_map_.feature_value_.size();
	dense_feature_map_.feature2offset_map_.emplace(feature_name_key, index);
	
	//Allocate the memory
	dense_feature_map_.feature_value_.emplace_back(TensorBlob());
	auto& blob = dense_feature_map_.feature_value_.back();
	size_t T_capacity = typed_capacity;
	blob.Reserve(T_capacity, byte_size, context);
	//blob.ResetNoAllocate(byte_size, tensor_dim, channel_valid_byte);
	blob.Reset(tensor_dim, byte_size, context, channel_valid_byte);
	
	//Seems ok
	return true;
}

bool poser::FeatureMap::AllocateDenseFeature(const poser::FeatureChannelType &channel, poser::MemoryContext context) {
	LOG_ASSERT(dense_feature_map_.dense_feature_capacity_ > 0) << "The size is not initialized yet";
	return allocateDenseFeature(
		channel.get_name_key(),
		channel.type_byte(),
		dense_feature_map_.dense_feature_dim_,
		context,
		dense_feature_map_.dense_feature_capacity_,
		channel.valid_type_byte());
}

bool poser::FeatureMap::AllocateDenseFeature(
	const poser::FeatureChannelType &channel,
	const poser::TensorDim &tensor_dim,
	poser::MemoryContext context,
	std::size_t T_capacity
) {
	return allocateDenseFeature(
		channel.get_name_key(),
		channel.type_byte(),
		tensor_dim,
		context,
		T_capacity,
		channel.valid_type_byte());
}

bool poser::FeatureMap::InsertDenseFeature(
	const std::string &channel_key,
	const poser::BlobView &tensor,
	poser::MemoryContext context
) {
	const auto allocate_success = allocateDenseFeature(
		channel_key,
		tensor.TypeByte(), tensor.DimensionalSize(),
		context,
		0,
		tensor.ValidTypeByte());
	if(!allocate_success)
		return false;
	
	//Do memory copy
	auto copy_to = GetDenseFeatureReadWrite(channel_key, tensor.Context());
	BlobCopyNoSync(tensor, copy_to);
	return true;
}

bool poser::FeatureMap::InsertDenseFeature(
	const std::string &channel_key,
	const poser::BlobView &tensor) {
	return InsertDenseFeature(channel_key, tensor, tensor.GetMemoryContext());
}

bool poser::FeatureMap::ExistDenseFeature(const std::string& name_key, MemoryContext context) const {
	const auto equal_range = dense_feature_map_.feature2offset_map_.equal_range(name_key);
	for(auto iter = equal_range.first; iter != equal_range.second; iter++)
	{
		auto index = iter->second;
		LOG_ASSERT(index < dense_feature_map_.feature_value_.size());
		if(dense_feature_map_.feature_value_[index].GetMemoryContext() == context)
			return true;
	}
	
	//Not exist
	return false;
}

//The method to process sparse feature
bool poser::FeatureMap::allocateSparseFeature(
	const std::string &feature_channel_key,
	unsigned short channel_byte_size,
	std::size_t T_capacity,
	poser::MemoryContext context,
	unsigned short valid_type_byte
) {
	//There exist this feature, return false
	const auto equal_range = sparse_feature_map_.feature2offset_map_.equal_range(feature_channel_key);
	for(auto iter = equal_range.first; iter != equal_range.second; iter++) {
		//Need to check both type and context
		int index = iter->second;
		if(sparse_feature_map_.feature_value_[index].GetMemoryContext() == context)
			return false;
	}
	
	//This is new feature, insert into map
	const auto index = sparse_feature_map_.feature_value_.size();
	sparse_feature_map_.feature2offset_map_.emplace(feature_channel_key, index);
	
	//Allocate the memory
	sparse_feature_map_.feature_value_.emplace_back(TensorBlob());
	sparse_feature_map_.feature_index_.emplace_back(TensorBlob());
	auto& value_blob = sparse_feature_map_.feature_value_.back();
	auto& index_blob = sparse_feature_map_.feature_index_.back();
	//Process the feature and index, resize to zero by default
	value_blob.Reserve(T_capacity, channel_byte_size, context);
	value_blob.Reset(TensorDim(0), channel_byte_size, context, valid_type_byte);
	
	index_blob.Reserve<unsigned>(T_capacity, context);
	return true;
}

bool poser::FeatureMap::ExistSparseFeature(const std::string& name_key, MemoryContext context) const {
	const auto equal_range = sparse_feature_map_.feature2offset_map_.equal_range(name_key);
	for(auto iter = equal_range.first; iter != equal_range.second; iter++)
	{
		auto index = iter->second;
		LOG_ASSERT(index < sparse_feature_map_.feature_value_.size());
		if(sparse_feature_map_.feature_value_[index].GetMemoryContext() == context)
			return true;
	}
	
	//not exist
	return false;
}

//For dense feature
int poser::FeatureMap::getDenseFeatureIndex(const std::string& name_key, MemoryContext context) const {
	const auto equal_range = dense_feature_map_.feature2offset_map_.equal_range(name_key);
	for(auto iter = equal_range.first; iter != equal_range.second; iter++)
	{
		auto index = iter->second;
		LOG_ASSERT(index < dense_feature_map_.feature_value_.size());
		if(dense_feature_map_.feature_value_[index].GetMemoryContext() == context)
			return index;
	}
	LOG(FATAL) << "The feature " << name_key << " is not found";
}

poser::TensorBlob& poser::FeatureMap::GetDenseFeatureRawBlob(const std::string& channel, MemoryContext context) {
	auto index = getDenseFeatureIndex(channel, context);
	return dense_feature_map_.feature_value_[index];
}

const poser::TensorBlob &poser::FeatureMap::GetDenseFeatureRawBlob(const std::string& channel, MemoryContext context) const {
	auto index = getDenseFeatureIndex(channel, context);
	return dense_feature_map_.feature_value_[index];
}

poser::BlobView poser::FeatureMap::GetDenseFeatureReadOnly(const std::string& channel, MemoryContext context) const {
	auto index = getDenseFeatureIndex(channel, context);
	return dense_feature_map_.feature_value_[index].GetTensorReadOnly();
}

poser::BlobSlice poser::FeatureMap::GetDenseFeatureReadWrite(
	const std::string &channel,
	poser::MemoryContext context
) {
	auto& raw_blob = GetDenseFeatureRawBlob(channel, context);
	return raw_blob.GetTensorReadWrite();
}

void poser::FeatureMap::ResizeDenseFeatureOrException(poser::TensorDim dim) {
	for(auto& blob : dense_feature_map_.feature_value_)
		blob.ResizeOrException(dim);
	
	//Modify the dense tensor dim
	dense_feature_map_.dense_feature_dim_ = dim;
}

//For sparse feature
int poser::FeatureMap::getSparseFeatureIndex(const std::string& name_key, MemoryContext context) const {
	const auto equal_range = sparse_feature_map_.feature2offset_map_.equal_range(name_key);
	for(auto iter = equal_range.first; iter != equal_range.second; iter++)
	{
		auto index = iter->second;
		LOG_ASSERT(index < sparse_feature_map_.feature_value_.size());
		if(sparse_feature_map_.feature_value_[index].GetMemoryContext() == context)
			return index;
	}
	LOG(FATAL) << "The feature " << name_key << " is not found";
}

poser::TensorBlob &poser::FeatureMap::GetSparseFeatureValueRawBlob(const std::string& channel, MemoryContext context) {
	const auto index = getSparseFeatureIndex(channel, context);
	return sparse_feature_map_.feature_value_[index];
}

const poser::TensorBlob &poser::FeatureMap::GetSparseFeatureValueRawBlob(const std::string& channel, MemoryContext context) const {
	const auto index = getSparseFeatureIndex(channel, context);
	return sparse_feature_map_.feature_value_[index];
}

poser::BlobView poser::FeatureMap::GetSparseFeatureValueReadOnly(const std::string& channel, MemoryContext context) const {
	const auto index = getSparseFeatureIndex(channel, context);
	return sparse_feature_map_.feature_value_[index].GetTensorReadOnly();
}

poser::BlobSlice poser::FeatureMap::GetSparseFeatureValueReadWrite(const std::string &channel, poser::MemoryContext context) {
	const auto index = getSparseFeatureIndex(channel, context);
	return sparse_feature_map_.feature_value_[index].GetTensorReadWrite();
}

poser::TensorSlice<unsigned int> poser::FeatureMap::GetSparseFeatureIndexReadWrite(
	const std::string& name_key,
	MemoryContext context
) {
	const auto index = getSparseFeatureIndex(name_key, context);
	return sparse_feature_map_.feature_index_[index].GetTypedTensorReadWrite<unsigned>();
}

const poser::TensorView<unsigned>
poser::FeatureMap::GetSparseFeatureIndexReadOnly(
	const std::string& name_key,
	MemoryContext context
) const {
	const auto index = getSparseFeatureIndex(name_key, context);
	return sparse_feature_map_.feature_index_[index].GetTypedTensorReadOnly<unsigned>();
}

void poser::FeatureMap::ResizeSparseFeatureOrException(
	const std::string &channel,
	poser::TensorDim dim,
	poser::MemoryContext context
) {
	const auto index = getSparseFeatureIndex(channel, context);
	sparse_feature_map_.feature_value_[index].ResizeOrException(dim);
	sparse_feature_map_.feature_index_[index].ResizeOrException(dim);
}

//The getter using general feature
bool poser::FeatureMap::ExistFeature(const poser::FeatureChannelType &type, MemoryContext context) const {
	if(type.is_sparse())
		return ExistSparseFeature(type.get_name_key(), context);
	else
		return ExistDenseFeature(type.get_name_key(), context);
}

poser::TensorBlob &poser::FeatureMap::GetFeatureValueRawBlob(const poser::FeatureChannelType &type, MemoryContext context) {
	if(type.is_dense())
		return GetDenseFeatureRawBlob(type.get_name_key(), context);
	else
		return GetSparseFeatureValueRawBlob(type.get_name_key(), context);
}


const poser::TensorBlob &poser::FeatureMap::GetFeatureValueRawBlobReadOnly(const poser::FeatureChannelType &type, MemoryContext context) const {
	if(type.is_dense())
		return GetDenseFeatureRawBlob(type.get_name_key(), context);
	else
		return GetSparseFeatureValueRawBlob(type.get_name_key(), context);
}

poser::BlobView poser::FeatureMap::GetFeatureValueReadOnly(const poser::FeatureChannelType& type, MemoryContext context) const {
	if(type.is_dense())
		return GetDenseFeatureReadOnly(type.get_name_key(), context);
	else
		return GetSparseFeatureValueReadOnly(type.get_name_key(), context);
}

poser::BlobSlice poser::FeatureMap::GetFeatureValueReadWrite(
	const poser::FeatureChannelType &type,
	poser::MemoryContext context
) {
	if(type.is_dense())
		return GetDenseFeatureReadWrite(type.get_name_key(), context);
	else
		return GetSparseFeatureValueReadWrite(type.get_name_key(), context);
}

//The method for save and load
void poser::FeatureMap::SaveToJson(nlohmann::json &node) const {
	node["dense_type_index"] = dense_feature_map_.feature2offset_map_;
	node["dense_value"] = dense_feature_map_.feature_value_;
	node["dense_dim"] = dense_feature_map_.dense_feature_dim_;
	node["dense_capacity"] = dense_feature_map_.dense_feature_capacity_;
	node["sparse_type_index"] = sparse_feature_map_.feature2offset_map_;
	node["sparse_value"] = sparse_feature_map_.feature_value_;
	node["sparse_index"] = sparse_feature_map_.feature_index_;
}

void poser::FeatureMap::LoadFromJson(const nlohmann::json &node) {
	//The dense part
	dense_feature_map_.feature2offset_map_.clear();
	dense_feature_map_.feature2offset_map_ = node["dense_type_index"].get<FeatureMultiMap<int>>();
	dense_feature_map_.feature_value_.clear();
	dense_feature_map_.feature_value_ = node["dense_value"].get<std::vector<TensorBlob>>();
	dense_feature_map_.dense_feature_dim_ = node["dense_dim"].get<TensorDim>();
	dense_feature_map_.dense_feature_capacity_ = node["dense_capacity"].get<std::size_t >();
	
	//The sparse part
	sparse_feature_map_.feature2offset_map_.clear();
	sparse_feature_map_.feature2offset_map_ = node["sparse_type_index"].get<FeatureMultiMap<int>>();
	sparse_feature_map_.feature_value_.clear();
	sparse_feature_map_.feature_value_ = node["sparse_value"].get<std::vector<TensorBlob>>();
	sparse_feature_map_.feature_index_.clear();
	sparse_feature_map_.feature_index_ = node["sparse_index"].get<std::vector<TensorBlob>>();
}

void poser::to_json(nlohmann::json &j, const poser::FeatureMap& rhs) {
	json map_j;
	rhs.SaveToJson(map_j);
	j = map_j;
}

void poser::from_json(const nlohmann::json &j, poser::FeatureMap &rhs) {
	rhs.LoadFromJson(j);
}