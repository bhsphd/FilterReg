//
// Created by wei on 1/16/19.
//

#include "kinematic/cpd/cpd_geometry_update.h"

void poser::UpdateLiveVertex(
	const poser::CoherentPointDriftKinematic &kinematic,
	poser::FeatureMap &geometric_model
) {
	//Get data
	auto ref_vertex = geometric_model.GetTypedFeatureValueReadOnly<float4>(kinematic.ReferenceVertexChannel(), MemoryContext::CpuMemory);
	auto live_vertex = geometric_model.GetTypedFeatureValueReadWrite<float4>(kinematic.LiveVertexChannel(), MemoryContext::CpuMemory);
	const Eigen::MatrixXf& displacement = kinematic.GetMotionVectorField();

	//Check the size
	LOG_ASSERT(ref_vertex.Size() == displacement.rows());
	LOG_ASSERT(displacement.cols() == 3);

	//Apply displacement
	for (auto i = 0; i < ref_vertex.Size(); i++) {
		const auto ref_point_i = ref_vertex[i];
		auto& live_point_i = live_vertex[i];
		const auto *ref_point_i_ptr = (const float *)(&ref_point_i);
		auto *live_point_i_ptr = (float *)(&live_point_i);

		//Assign it
		for (auto j = 0; j < 3; j++) {
			live_point_i_ptr[j] = ref_point_i_ptr[j] + displacement(i, j);
		}

		//The last element
		live_point_i.w = ref_point_i.w;
	}
}

void poser::UpdateLiveVertex(
	const poser::CoherentPointDriftKinematic &kinematic,
	const poser::BlobView &reference_points,
	poser::BlobSlice live_points
) {
	LOG(FATAL) << "Not implemented yet";
}