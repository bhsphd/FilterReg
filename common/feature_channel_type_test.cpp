//
// Created by wei on 9/10/18.
//

#include "common/feature_channel_type.h"

#include <gtest/gtest.h>

TEST(FeatureChannelTypeTest, BasicTest) {
	using namespace poser;
	FeatureChannelType type("Dense Depth Image", sizeof(unsigned short));
	EXPECT_TRUE(type.is_dense());
	EXPECT_FALSE(type.is_sparse());
	
	//Another type
	FeatureChannelType type_1("Another Dense Depth", sizeof(unsigned short));
	EXPECT_NE(type, type_1);
}