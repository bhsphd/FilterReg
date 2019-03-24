//
// Created by wei on 9/12/18.
//

#include "common/tensor_utils.h"

void poser::to_json(nlohmann::json &j, const poser::TensorDim &rhs) {
	json tensor_j;
	tensor_j[0] = rhs.rows();
	tensor_j[1] = rhs.cols();
	j = tensor_j;
}

void poser::from_json(const nlohmann::json &j, poser::TensorDim &rhs) {
	auto rows = j[0].get<unsigned>();
	auto cols = j[1].get<unsigned>();
	rhs = poser::TensorDim(rows, cols);
}