//
// Created by wei on 9/23/18.
//

#pragma once

#include "geometry_utils/vector_operations.hpp"
#include <vector_functions.h>


namespace poser {
	
	
	/* The scale factor used for permutohedral lattice
	 * The value is dependent on whether blurring is performed
	 */
	template <int FeatureDim> __host__ __device__
	float permutohedral_scale_noblur(int index);
	template <int FeatureDim> __host__ __device__
	float permutohedral_scale_withblur(int index);
	
	
	//Small struct to hold the lattice coordinate as key
	template<int FeatureDim>
	struct LatticeCoordKey {
		//Only maintain the first FeatureDim elements.
		//As the total sum to zero.
		short key[FeatureDim];
		
		//The hashing of this key
		__host__ __device__ __forceinline__ unsigned hash() const {
			unsigned hash_value = 0;
			for(auto i = 0; i < FeatureDim; i++) {
				hash_value += key[i];
				hash_value *= 1500007; //This is a prime number
			}
			return hash_value;
		}
		
		//The comparator of a key
		__host__ __device__ __forceinline__ char less_than(const LatticeCoordKey<FeatureDim>& rhs) const {
			char is_less_than = 0;
			for(auto i = 0; i < FeatureDim; i++) {
				if(key[i] < rhs.key[i]) {
					is_less_than = 1;
					break;
				} else if(key[i] > rhs.key[i]) {
					is_less_than = -1;
					break;
				}
				//Else, continue
			}
			return is_less_than;
		}
		
		//Operator
		__host__ __device__ __forceinline__ bool operator==(const LatticeCoordKey<FeatureDim>& rhs) const {
			for(auto i = 0; i < FeatureDim; i++) {
				if(key[i] != rhs.key[i]) return false;
			}
			return true;
		}
	};
	
	
	/**
	 * \brief Compute the lattice key and the weight of the lattice point
	 *        surround this feature.
	 * \tparam FeatureDim
	 * \param feature The feature vector, in the size of FeatureDim
	 * \param lattice_coord_keys The lattice coord keys nearby this feature. The
	 *                           array is in the size of FeatureDim + 1.
	 * \param barycentric The weight value, in the size of FeatureDim + 2, while
	 *                    the first FeatureDim + 1 elements match the weight
	 *                    of the lattice_coord_keys
	 */
	template<int FeatureDim> __host__ __device__ __forceinline__
	void permutohedral_lattice_noblur(
		const float* feature,
		LatticeCoordKey<FeatureDim>* lattice_coord_keys,
		float* barycentric
	);
	
	template<int FeatureDim> __host__ __device__ __forceinline__
	void permutohedral_lattice_withblur(
		const float* feature,
		LatticeCoordKey<FeatureDim>* lattice_coord_keys,
		float* barycentric
	);
}


#include "geometry_utils/permutohedral_common.hpp"