#pragma once

#include "common/feature_map.h"
#include "geometry_utils/permutohedral_common.h"
#include "corr_search/target_computer_base.h"

namespace poser {
	
	template<int FeatureDim, typename LatticeValueT>
	class GMMPermutohedralBase : public SingleFeatureTargetComputerBase {
	protected:
		//The sigma value for this problem can be potential different
		float sigma_value_[FeatureDim];
		//The constant value to indicate the outlier in weight
		float outlier_constant_ = 0.2f;
	public:
		GMMPermutohedralBase(
			FeatureChannelType observation_world_vertex,
			FeatureChannelType model_feature,
			FeatureChannelType observation_feature = FeatureChannelType());
		~ GMMPermutohedralBase() override = default;
		
		//The checking method
		void CheckAndAllocateTarget(
			const FeatureMap& observation,
			const FeatureMap& model,
			GeometricTargetBase& target) override;
		void SetOutlierConstant(float outlier_constant) { outlier_constant_ = outlier_constant; }
	
	protected:
		//The hash and index for permutohedral lattice
		struct PermutohedralHasher {
			std::size_t operator()(const LatticeCoordKey<FeatureDim>& lattice) const {
				return lattice.hash();
			}
		};
		
		//The hash map of lattice
		unordered_map<LatticeCoordKey<FeatureDim>, LatticeValueT, PermutohedralHasher> lattice_map_;
		
		//The method to build lattice index
		using ValueInitializer = std::function<void(float weight, const float4& vertex, int obs_idx, LatticeValueT& info)>;
		using ValueUpdater = std::function<void(float weight, const float4& vertex, int obs_idx, LatticeValueT& info)>;
		void buildLatticeIndexNoBlur(
			const poser::FeatureMap &observation,
			const ValueInitializer& initializer,
			const ValueUpdater& updater);
		
		
		//The method to compute the target
		template<typename AggregatedT>
		using AggregateValueInitializer = std::function<void(AggregatedT&)>;
		template<typename AggregatedT>
		using AggregateValueUpdater = std::function<void(AggregatedT&, float lattice_weight, const LatticeValueT& lattice)>;
		template<typename AggregatedT>
		using TargetComputerFromAggregatedValue = std::function<void(float4& target_i, int model_idx, const AggregatedT&)>;
		template<typename AggregatedT>
		void computeTargetNoBlur(
			const FeatureMap& observation,
			const FeatureMap& model,
			GeometricTargetBase& target,
			const AggregateValueInitializer<AggregatedT>& initializer,
			const AggregateValueUpdater<AggregatedT>& updater,
			const TargetComputerFromAggregatedValue<AggregatedT>& result_computer);
	public:
		//The getter for lattice map, used in ransac/fitness
		const decltype(lattice_map_)& GetLatticeMap() const { return lattice_map_; }
	};
}

#include "corr_search/gmm/gmm_permutohedral_base.hpp"