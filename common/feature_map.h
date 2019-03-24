//
// Created by wei on 9/10/18.
//

#pragma once

#include "common/common_type.h"
#include "common/feature_channel_type.h"
#include "common/tensor_blob.h"
#include "common/data_transfer.h"

namespace poser {

	/* The feature map, by its name, maintains a map from FeatureChannelType
	 * to the actual feature values. This is the container class maintains all buffers.
	 * The feature can be dense feature or sparse feature, and all features should
	 * be consistent with each other. The FeatureMap actually corresponds to
	 * pcl::PointCloud, but use dynamic index of different channels (instead of compile time)
	 */
	class FeatureMap {
	public:
		FeatureMap();
		~FeatureMap() = default;
		POSER_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(FeatureMap);
		
		//The method would Clone everything in this map to another one
		void CloneTo(FeatureMap& other) const;
		
		
		/* The allocation interface for dense feature. The blob is allocated with the given capacity
		 * and resize to the given size. If capacity is less than the size, then the capacity is
		 * inferred from T and tensor_dim. The feature will be inserted into dense feature.
		 * If the feature is already here, a false will be returned.
		 */
	private:
		bool allocateDenseFeature(
			const string& feature_channel_key,
			unsigned short channel_byte_size,
			const TensorDim& tensor_dim,
			MemoryContext context,
			std::size_t typed_capacity = 0,
			unsigned short channel_valid_byte = 0);
	public:
		template<typename T>
		bool AllocateDenseFeature(
			const FeatureChannelType& channel,
			const TensorDim& tensor_dim,
			MemoryContext context = MemoryContext::CpuMemory,
			std::size_t T_capacity = 0);
		bool AllocateDenseFeature(
			const FeatureChannelType& channel,
			const TensorDim& tensor_dim,
			MemoryContext context = MemoryContext::CpuMemory,
			std::size_t typed_capacity = 0);
		
		//Allocate using existing size
		bool AllocateDenseFeature(const FeatureChannelType& channel, MemoryContext context = MemoryContext::CpuMemory);
		
		//The methods to insert a feature from existing blob.
		//These methods are usually used to build the geometric model.
		template<typename T>
		bool InsertDenseFeature(const FeatureChannelType& channel, const TensorView<T>& tensor);
		template<typename T>
		bool InsertDenseFeature(const FeatureChannelType& channel, const TensorView<T>& tensor, MemoryContext context);
		bool InsertDenseFeature(const std::string& channel_key, const BlobView& tensor);
		bool InsertDenseFeature(const std::string& channel_key, const BlobView& tensor, MemoryContext context);
		
		//Check if the feature is declared here
		bool ExistDenseFeature(const std::string& name_key, MemoryContext context = MemoryContext::CpuMemory)  const;
		
		
		/* The method to insert sparse feature
		 */
	private:
		bool allocateSparseFeature(
			const string& feature_channel_key,
			unsigned short channel_byte_size,
			std::size_t T_capacity,
			MemoryContext context,
			unsigned short valid_type_byte);
	public:
		template<typename T>
		bool AllocateSparseFeature(
			const string& feature_name_key,
			std::size_t T_capacity,
			MemoryContext context = MemoryContext::CpuMemory,
			unsigned short valid_type_byte = 0);
		
		//Check the existance of feature
		bool ExistSparseFeature(const std::string& name_key, MemoryContext context = MemoryContext::CpuMemory) const;
		
		
		/* The query interface for dense feature by string key. The dense feature only have
		 * value tensor and all components of dense feature should be corresponded.
		 */
	private:
		int getDenseFeatureIndex(const std::string& name_key, MemoryContext context) const;
		TensorBlob& GetDenseFeatureRawBlob(const std::string& channel, MemoryContext context);
	public:
		//The direct version
		const TensorBlob& GetDenseFeatureRawBlob(const std::string& channel, MemoryContext context) const;
		BlobView GetDenseFeatureReadOnly(const std::string& channel, MemoryContext context = MemoryContext::CpuMemory) const;
		BlobSlice GetDenseFeatureReadWrite(const std::string& channel, MemoryContext context = MemoryContext::CpuMemory);
		
		//The template typed version
		template<typename T> TensorSlice<T> GetTypedDenseFeatureReadWrite(
			const std::string& channel, MemoryContext context = MemoryContext::CpuMemory);
		template<typename T> const TensorView<T> GetTypedDenseFeatureReadOnly(
			const std::string& channel, MemoryContext context = MemoryContext::CpuMemory) const;
		
		//Resize the dense feature, all dense feature should have aligned index
		void ResizeDenseFeatureOrException(TensorDim dim);
		TensorDim GetDenseFeatureDim() const { return dense_feature_map_.dense_feature_dim_; }
		std::size_t GetDenseFeatureCapacity() const { return dense_feature_map_.dense_feature_capacity_; }
		
		
		/* The getter for sparse feature with string key. The sparse feature has value and
		 * index tensor. The index are the relative to dense feature.
		 */
	private:
		int getSparseFeatureIndex(const std::string& name_key, MemoryContext context) const;
		TensorBlob& GetSparseFeatureValueRawBlob(const std::string& channel, MemoryContext context = MemoryContext::CpuMemory);
	public:
		//The non-template version
		const TensorBlob& GetSparseFeatureValueRawBlob(const std::string& channel, MemoryContext context = MemoryContext::CpuMemory) const;
		BlobView GetSparseFeatureValueReadOnly(const std::string& channel, MemoryContext context = MemoryContext::CpuMemory) const;
		BlobSlice GetSparseFeatureValueReadWrite(const std::string& channel, MemoryContext context = MemoryContext::CpuMemory);
		
		//The template typed version
		template<typename T>
		const TensorView<T> GetTypedSparseFeatureValueReadOnly(
			const std::string& channel, MemoryContext context = MemoryContext::CpuMemory) const;
		template<typename T>
		TensorSlice<T> GetTypedSparseFeatureValueReadWrite(
			const std::string& channel, MemoryContext context = MemoryContext::CpuMemory);
		
		//The index should be always uint
		const TensorView<unsigned> GetSparseFeatureIndexReadOnly(
			const std::string& channel, MemoryContext context = MemoryContext::CpuMemory) const;
		TensorSlice<unsigned> GetSparseFeatureIndexReadWrite(
			const std::string& channel, MemoryContext context = MemoryContext::CpuMemory);
		
		//Each sparse feature may has its own size
		void ResizeSparseFeatureOrException(const std::string& channel, TensorDim dim, MemoryContext context = MemoryContext::CpuMemory);
		
		
		/* The getter using FeatureChannelType, which can determine the context and density
		 * by the type itself. The feature value are available for both dense and sparse case,
		 * while the index is only available for sparse case.
		 */
	private:
		TensorBlob& GetFeatureValueRawBlob(const FeatureChannelType& type, MemoryContext context);
	public:
		bool ExistFeature(const FeatureChannelType& type, MemoryContext context) const;
		//The non-template interface
		const TensorBlob& GetFeatureValueRawBlobReadOnly(const FeatureChannelType& type, MemoryContext context) const;
		BlobView GetFeatureValueReadOnly(const FeatureChannelType& type, MemoryContext context) const;
		BlobSlice GetFeatureValueReadWrite(const FeatureChannelType& type, MemoryContext context);
		
		//The template interface
		template<typename T>
		TensorView<T> GetTypedFeatureValueReadOnly(const FeatureChannelType& type, MemoryContext context) const;
		template<typename T>
		TensorSlice<T> GetTypedFeatureValueReadWrite(const FeatureChannelType& type, MemoryContext context);
		
		//The method for save and load from json
		void SaveToJson(json& node) const;
		void LoadFromJson(const json& node);
	private:
		//The dense map
		struct {
			FeatureMultiMap<int> feature2offset_map_;
			std::vector<TensorBlob> feature_value_;
			TensorDim dense_feature_dim_;
			std::size_t dense_feature_capacity_;
		} dense_feature_map_;
		
		//The sparse map, the feature index is used to
		//locate the corresponded dense part.
		struct {
			FeatureMultiMap<int> feature2offset_map_;
			std::vector<TensorBlob> feature_value_;
			std::vector<TensorBlob> feature_index_; //The index in dense feature
		} sparse_feature_map_;
	};
}

//The template processor for dense feature
template<typename T>
bool poser::FeatureMap::AllocateDenseFeature(
	const FeatureChannelType& channel,
	const TensorDim& tensor_dim,
	MemoryContext context,
	std::size_t T_capacity
) {
	return allocateDenseFeature(channel.get_name_key(), sizeof(T), tensor_dim, context, T_capacity, channel.valid_type_byte());
}

template<typename T>
bool poser::FeatureMap::InsertDenseFeature(
	const FeatureChannelType& channel,
	const poser::TensorView<T> &tensor
) {
	return InsertDenseFeature<T>(channel, tensor, tensor.GetMemoryContext());
}


template<typename T>
bool poser::FeatureMap::InsertDenseFeature(
	const FeatureChannelType& channel,
	const poser::TensorView<T> &tensor,
	MemoryContext context
) {
	const auto allocate_success = allocateDenseFeature(
		channel.get_name_key(),
		sizeof(T),
		tensor.DimensionalSize(),
		context,
		0,
		channel.valid_type_byte());
	if(!allocate_success)
		return false;
	
	//Do memory copy
	auto copy_to = GetTypedDenseFeatureReadWrite<T>(channel.get_name_key(), tensor.GetMemoryContext());
	TensorCopyNoSync<T>(tensor, copy_to);
	return true;
}

template<typename T>
poser::TensorSlice<T> poser::FeatureMap::GetTypedDenseFeatureReadWrite(const std::string& channel, MemoryContext context) {
	auto& blob = GetDenseFeatureRawBlob(channel, context);
	return blob.GetTypedTensorReadWrite<T>();
}

template<typename T>
const poser::TensorView<T> poser::FeatureMap::GetTypedDenseFeatureReadOnly(const std::string& channel, MemoryContext context) const {
	const auto& blob = GetDenseFeatureReadOnly(channel, context);
	return blob.ToTensorView<T>();
}

template<typename T>
bool poser::FeatureMap::AllocateSparseFeature(
	const string& feature_name_key,
	std::size_t T_capacity,
	MemoryContext context,
	unsigned short valid_type_byte
) {
	return allocateSparseFeature(feature_name_key, sizeof(T), T_capacity, context, valid_type_byte);
}

template<typename T>
poser::TensorSlice<T> poser::FeatureMap::GetTypedSparseFeatureValueReadWrite(const std::string& channel, MemoryContext context) {
	auto& blob = GetSparseFeatureValueRawBlob(channel, context);
	return blob.GetTypedTensorReadWrite<T>();
}

template<typename T>
const poser::TensorView<T> poser::FeatureMap::GetTypedSparseFeatureValueReadOnly(const std::string& channel, MemoryContext context) const {
	const auto& blob = GetSparseFeatureValueReadOnly(channel, context);
	return blob.ToTensorView<T>();
}

//The getter using FeatureChannelType
template<typename T>
poser::TensorView<T> poser::FeatureMap::GetTypedFeatureValueReadOnly(const poser::FeatureChannelType &type, MemoryContext context) const {
	LOG_ASSERT(type.type_byte() == sizeof(T));
	const auto blob = GetFeatureValueReadOnly(type, context);
	return blob.ToTensorView<T>();
}

template<typename T>
poser::TensorSlice<T> poser::FeatureMap::GetTypedFeatureValueReadWrite(const poser::FeatureChannelType &type, MemoryContext context) {
	LOG_ASSERT(type.type_byte() == sizeof(T));
	auto& blob = GetFeatureValueRawBlob(type, context);
	return blob.GetTypedTensorReadWrite<T>();
}

/* For serialize
 */
namespace poser {
	void to_json(nlohmann::json& j, const FeatureMap& rhs);
	void from_json(const nlohmann::json& j, FeatureMap& rhs);
}