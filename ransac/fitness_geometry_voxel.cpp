//
// Created by wei on 12/1/18.
//

#include "ransac/fitness_geometry_voxel.h"
#include <vector_functions.h>

void poser::FitnessGeometricOnlyVoxel::UpdateObservation(const poser::FeatureMap &observation) {
	//Fetch the vertex
	const auto vertex_in = observation.GetTypedFeatureValueReadOnly<float4>(observation_vertex_channel_, MemoryContext::CpuMemory);
	
	auto insert_to_set = [&](const int3& voxel) -> void {
		const auto iter = voxel_set_.find(voxel);
		if(iter == voxel_set_.end()) {
			voxel_set_.emplace(voxel);
		}
	};
	
	//Iterate through the input cloud
	voxel_set_.clear();
	for(unsigned i = 0; i < vertex_in.Size(); i++) {
		//Check it against bounding box
		const auto& vertex_i = vertex_in[i];
		
		//Compute location, there will be 8 possible ones
		//Note that to-int is round down
		int3 voxel;
		voxel.x = int(floorf(vertex_i.x * inv_leaf_size_));
		voxel.y = int(floorf(vertex_i.y * inv_leaf_size_));
		voxel.z = int(floorf(vertex_i.z * inv_leaf_size_));
		
		//The insert group
		insert_to_set(make_int3(voxel.x + 0, voxel.y + 0, voxel.z + 0));
		insert_to_set(make_int3(voxel.x + 1, voxel.y + 0, voxel.z + 0));
		insert_to_set(make_int3(voxel.x + 0, voxel.y + 1, voxel.z + 0));
		insert_to_set(make_int3(voxel.x + 0, voxel.y + 0, voxel.z + 1));
		insert_to_set(make_int3(voxel.x + 1, voxel.y + 1, voxel.z + 0));
		insert_to_set(make_int3(voxel.x + 0, voxel.y + 1, voxel.z + 1));
		insert_to_set(make_int3(voxel.x + 1, voxel.y + 0, voxel.z + 1));
		insert_to_set(make_int3(voxel.x + 1, voxel.y + 1, voxel.z + 1));
	}
}

poser::FitnessEvaluator::Result poser::FitnessGeometricOnlyVoxel::Evaluate(const poser::FeatureMap &model) const {
	//Get the model vertex
	const auto model_v = model.GetTypedFeatureValueReadOnly<float4>(model_vertex_channel_, MemoryContext::CpuMemory);
	
	//Do it
	unsigned inlier_count = 0;
	for(auto i = 0; i < model_v.Size(); i++) {
		//Compute the NEAREST voxel
		const auto& vertex_i = model_v[i];
		int3 voxel;
		voxel.x = int(nearbyintf(vertex_i.x * inv_leaf_size_));
		voxel.y = int(nearbyintf(vertex_i.y * inv_leaf_size_));
		voxel.z = int(nearbyintf(vertex_i.z * inv_leaf_size_));
		
		//Only check this one
		if(voxel_set_.find(voxel) != voxel_set_.end())
			inlier_count++;
	}
	
	//OK
	Result result;
	result.fitness_score = float(inlier_count) / float(model_v.Size());
	result.inlier_l2_cost = std::numeric_limits<float>::max();
	return result;
}