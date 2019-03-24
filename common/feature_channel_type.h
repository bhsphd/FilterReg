//
// Created by wei on 9/10/18.
//

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <glog/logging.h>

#include "common/macro_copyable.h"
#include "common/common_type.h"
#include "common/tensor_utils.h"

namespace poser {
	
	/* The feature type is used as the key type for
	 * feature container, which is the base class for
	 * CameraObservation and Geometric Model
	 */
	class FeatureChannelType {
	private:
		//The key as the actual index
		std::string name_key_;
		bool valid_;
		
		//Whether this is a sparse feature
		bool is_sparse_feature_;
		
		//The byte size for the whole type
		//For instance, vertex is represented as float4,
		//type_byte_ is sizeof(float4), valid_type_byte is sizeof(float3)
		//The valid type is only used when accessed as BlobView/Slice
		unsigned short type_byte_;
		unsigned short valid_type_byte_;
	public:
		explicit FeatureChannelType() : name_key_(), valid_(false), is_sparse_feature_(false), type_byte_(0), valid_type_byte_(0) {};
		explicit FeatureChannelType(
			std::string name_key,
			unsigned short type_byte)
			: name_key_(std::move(name_key)),
			  type_byte_(type_byte), valid_type_byte_(type_byte),
			  is_sparse_feature_(false),
			  valid_(true) {};
		explicit FeatureChannelType(
			std::string name_key,
			unsigned short type_byte,
			unsigned short valid_type_byte,
			bool is_sparse)
			: name_key_(std::move(name_key)),
			  type_byte_(type_byte), valid_type_byte_(valid_type_byte),
			  is_sparse_feature_(is_sparse),
			  valid_(true) {};
		
		bool is_valid() const { return valid_ && (!name_key_.empty()); }
		bool is_dense() const { return !is_sparse_feature_; }
		bool is_sparse() const { return is_sparse_feature_; }
		const std::string& get_name_key() const { return name_key_; }
		unsigned short type_byte() const { return type_byte_; }
		unsigned short valid_type_byte() const { return valid_type_byte_; }
		template<typename T>
		bool type_size_matched() const { return type_byte_ == sizeof(T); }
		
		//The compare is base on string
		bool operator==(const FeatureChannelType& other) const {
			return name_key_ == other.name_key_;
		}
		bool operator!=(const FeatureChannelType& other) const {
			return name_key_ != other.name_key_;
		}
	};
	
	
	/* This class provide a set of all declared features
	 * and some commonly used feature types.
	 */
	struct CommonFeatureChannelKey {
		//The default depth part
		static const FeatureChannelType& RawDepthImage();
		static const FeatureChannelType& FilteredDepthImage();
		static const FeatureChannelType& ObservationVertexCamera();
		static const FeatureChannelType& ObservationNormalCamera();
		static const FeatureChannelType& ObservationVertexWorld();
		static const FeatureChannelType& ObservationNormalWorld();
		
		//The default rgb part
		static const FeatureChannelType& RawRGBImage();
		
		//The index for gathering from another feature map
		static const FeatureChannelType& ForegroundMask();
		static const FeatureChannelType& GatherIndex();
		
		//The default model part
		static const FeatureChannelType& ReferenceVertex();
		static const FeatureChannelType& LiveVertex();
		static const FeatureChannelType& ReferenceNormal();
		static const FeatureChannelType& LiveNormal();
		static const FeatureChannelType& VisibilityScore();
		
		//For articulated kinematic model
		static const FeatureChannelType& ArticulatedRigidBodyIndex();
	};
	
	
	/* The feature key for Image feature type, like SIFT, DON, or
	 * something else. These might be dynamically sized.
	 */
	struct ImageFeatureChannelKey {
		static const FeatureChannelType DONDescriptor(int feature_dim);
	};
	
	
	//The hash function and unordered_map for the feature type
	template<typename T>
	using FeatureMultiMap = std::unordered_multimap<std::string, T>;
}


//For serialize
namespace poser {
	void to_json(json& j, const FeatureChannelType& rhs);
	void from_json(const json& j, FeatureChannelType& rhs);
}

//For multimap
#ifndef __CUDACC__
namespace nlohmann {
	template<typename T>
	struct adl_serializer<::poser::FeatureMultiMap<T>> {
		static void to_json(json& j, const ::poser::FeatureMultiMap<T>& rhs) {
			json node;
			//Construct the key and value
			std::vector<std::string> key_vec;
			std::vector<T> value_vec;
			for(auto iter = rhs.begin(); iter != rhs.end(); iter++) {
				key_vec.emplace_back(iter->first);
				value_vec.emplace_back(iter->second);
			}
			
			//Insert it
			node["key"] = key_vec;
			node["value"] = value_vec;
			j = node;
		}
		
		static void from_json(const json& j, ::poser::FeatureMultiMap<T>& rhs) {
			//Load the kv map
			auto key_vec = j["key"].get<std::vector<std::string>>();
			auto value_vec = j["value"].get<std::vector<T>>();
			LOG_ASSERT(key_vec.size() == value_vec.size());
			
			//Insert into map
			rhs.clear();
			for(auto i = 0; i < key_vec.size(); i++) {
				rhs.emplace(key_vec[i], value_vec[i]);
			}
		}
	};
}
#endif