//
// Created by wei on 1/16/19.
//

#pragma once

#include "kinematic/cpd/cpd_kinematic_model.h"

namespace poser {
	
	//If the live vertex is just the reference vertex plus deformation field
	//Don't need to query the kd-tree for interpolation
	void UpdateLiveVertex(
		const CoherentPointDriftKinematic& kinematic,
		FeatureMap& geometric_model);
	
	//Need to query the kdtree and build the G matrix for new inputs
	void UpdateLiveVertex(
		const CoherentPointDriftKinematic& kinematic,
		const BlobView& reference_points,
		BlobSlice live_points);
}
