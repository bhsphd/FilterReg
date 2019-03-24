//
// Created by wei on 9/12/18.
//

#pragma once

#include "common/common_type.h"
#include "common/feature_map.h"
#include "common/macro_copyable.h"

namespace poser {
	
	class GeometricTargetBase {
	protected:
		//A small enum class for target type
		enum class TargetType {
			Dense, Sparse
		};
	
		//All the target has vertex
		TensorBlob target_vertex_;
	public:
		//Should be a virtual class
		GeometricTargetBase() = default;
		virtual ~GeometricTargetBase() = default;
		
		//The type of the target
		virtual TargetType GetTargetType() const = 0;
		bool IsSparseTarget() const { return GetTargetType() == TargetType::Sparse; }
		bool IsDenseTarget() const { return GetTargetType() == TargetType::Dense; }
		
		//The getter for vertex
		unsigned GetTargetFlattenSize() const { return target_vertex_.TensorFlattenSize(); }
		TensorSlice<float4> GetTargetVertexReadWrite() { return target_vertex_.GetTypedTensorReadWrite<float4>(); };
		TensorView<float4> GetTargetVertexReadOnly() const { return target_vertex_.GetTypedTensorReadOnly<float4>(); }
		
		//The context for the target
		MemoryContext GetMemoryContext() const { return target_vertex_.GetMemoryContext(); }
		bool IsCpuTarget() const { return target_vertex_.IsCpuMemory(); }
		bool IsGpuTarget() const { return target_vertex_.IsGpuMemory(); }
	};
	
	class DenseGeometricTarget : public GeometricTargetBase {
	private:
		//Maybe has normal
		TensorBlob target_normal_;
	public:
		DenseGeometricTarget() = default;
		~DenseGeometricTarget() override = default;
		TargetType GetTargetType() const override { return TargetType::Dense; }
		
		//The allocate method
		void AllocateTargetForModel(
			const FeatureMap& model_feature,
			MemoryContext context,
			bool with_normal = false);
		
		//The normal related
		TensorSlice<float4> GetTargetNormalReadWrite() { return target_normal_.GetTypedTensorReadWrite<float4>(); }
		TensorView<float4> GetTargetNormalReadOnly() const { return target_normal_.GetTypedTensorReadOnly<float4>(); }
		bool has_normal() const { return target_normal_.TensorFlattenSize() > 0; }
	};
	
	class SparseGeometricTarget : public GeometricTargetBase {
	private:
		//The channel used for sparse feature
		FeatureChannelType sparse_channel_;
		TensorBlob observation_index_;
		TensorBlob model_index_;
	public:
		SparseGeometricTarget() = default;
		~SparseGeometricTarget() override = default;
		TargetType GetTargetType() const override { return TargetType::Sparse; }
		
		//The allocate method, MUST on CPU
		void AllocateForSparseFeature(
			const FeatureMap& model_feature,
			const FeatureChannelType& channel,
			bool with_observation_idx = false);
		void AllocateTargetForModel(int capacity);
		void ResizeSparseTarget(int size);
		
		//The getter methods
		FeatureChannelType GetSparseFeatureChannel() const { return sparse_channel_; }
		TensorView<unsigned> GetTargetModelIndexReadOnly() const { return model_index_.GetTypedTensorReadOnly<unsigned>(); }
		TensorSlice<unsigned> GetTargetModelIndexReadWrite() { return model_index_.GetTypedTensorReadWrite<unsigned>(); }
		
		//The query method for observation index
		bool has_observation_index() const { return observation_index_.TensorFlattenSize() > 0; }
		TensorView<unsigned> GetObservationIndexReadOnly() const { return observation_index_.GetTypedTensorReadOnly<unsigned>(); }
		TensorSlice<unsigned> GetObservationIndexReadWrite() { return observation_index_.GetTypedTensorReadWrite<unsigned>(); }
	};
}
