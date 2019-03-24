#include "common/feature_channel_type.h"


#include <atomic>

//The named channel key for depth part
const poser::FeatureChannelType& poser::CommonFeatureChannelKey::RawDepthImage() {
	static const FeatureChannelType type("RawDepth", sizeof(unsigned short));
	return type;
}

const poser::FeatureChannelType& poser::CommonFeatureChannelKey::FilteredDepthImage() {
	static const FeatureChannelType type("FilterDepth", sizeof(unsigned short));
	return type;
}

const poser::FeatureChannelType& poser::CommonFeatureChannelKey::ObservationVertexCamera() {
	static const FeatureChannelType type("DepthVertex", sizeof(float4), sizeof(float3), false);
	return type;
}

const poser::FeatureChannelType& poser::CommonFeatureChannelKey::ObservationNormalCamera() {
	static const FeatureChannelType type("DepthNormal", sizeof(float4), sizeof(float3), false);
	return type;
}

const poser::FeatureChannelType& poser::CommonFeatureChannelKey::ObservationVertexWorld() {
	static const FeatureChannelType type("DepthVertexWorld", sizeof(float4), sizeof(float3), false);
	return type;
}

const poser::FeatureChannelType& poser::CommonFeatureChannelKey::ObservationNormalWorld() {
	static const FeatureChannelType type("DepthNormalWorld", sizeof(float4), sizeof(float3), false);
	return type;
}

const poser::FeatureChannelType& poser::CommonFeatureChannelKey::RawRGBImage(){
	static const FeatureChannelType type("RawRGB", sizeof(uchar4));
	return type;
}

const poser::FeatureChannelType& poser::CommonFeatureChannelKey::ForegroundMask(){
	static const FeatureChannelType type("ForegroundMask", sizeof(unsigned char));
	return type;
}

const poser::FeatureChannelType& poser::CommonFeatureChannelKey::GatherIndex() {
	static const FeatureChannelType type("GatherIndex", sizeof(unsigned));
	return type;
}

//The named channel key for model part
const poser::FeatureChannelType& poser::CommonFeatureChannelKey::ReferenceVertex() {
	static const FeatureChannelType type("ReferenceVertex", sizeof(float4), sizeof(float3), false);
	return type;
}

const poser::FeatureChannelType& poser::CommonFeatureChannelKey::LiveVertex() {
	static const FeatureChannelType type("LiveVertex", sizeof(float4), sizeof(float3), false);
	return type;
}

const poser::FeatureChannelType& poser::CommonFeatureChannelKey::ReferenceNormal() {
	static const FeatureChannelType type("ReferenceNormal", sizeof(float4), sizeof(float3), false);
	return type;
}

const poser::FeatureChannelType& poser::CommonFeatureChannelKey::LiveNormal() {
	static const FeatureChannelType type("LiveNormal", sizeof(float4), sizeof(float3), false);
	return type;
}

const poser::FeatureChannelType& poser::CommonFeatureChannelKey::VisibilityScore() {
	static const FeatureChannelType type("Visibility", sizeof(float), sizeof(float), false);
	return type;
}

const poser::FeatureChannelType& poser::CommonFeatureChannelKey::ArticulatedRigidBodyIndex() {
	static const FeatureChannelType type("ArticulatedBodyIndex", sizeof(int), sizeof(int), false);
	return type;
}

//The image feature
const poser::FeatureChannelType poser::ImageFeatureChannelKey::DONDescriptor(int feature_dim) {
	FeatureChannelType type("DONDescriptor", sizeof(float) * feature_dim);
	return type;
}

//For serialize
void poser::to_json(nlohmann::json &j, const poser::FeatureChannelType &rhs) {
	json feature_j;
	feature_j[0] = rhs.get_name_key();
	feature_j[1] = rhs.is_sparse();
	feature_j[2] = rhs.is_valid();
	feature_j[3] = rhs.type_byte();
	feature_j[4] = rhs.valid_type_byte();
	j = feature_j;
}

void poser::from_json(const nlohmann::json &j, poser::FeatureChannelType &rhs) {
	auto name = j[0].get<std::string>();
	auto is_sparse = j[1].get<bool>();
	auto valid = j[2].get<bool>();
	auto byte_size = j[3].get<unsigned short>();
	auto valid_type_byte = j[4].get<unsigned short>();
	if(valid)
		rhs = poser::FeatureChannelType(std::move(name), byte_size, valid_type_byte, is_sparse);
	else
		rhs = poser::FeatureChannelType();
}