#include <gtest/gtest.h>

#include "solve.h"
TEST(SwarmIntelligence, T1) {
  const std::vector<float> input = {0.0, 0.0, 1.0, 0.0, 3.0, 4.0, 0.0, -1.0};
  const std::vector<float> expected = {1.0, 0.0, 1.0, 0.0, 3.0, 3.0, 0.0, -1.0};
  const auto output = SwarmIntelligence(input);
  for (int i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(expected[i], output[i]) << i;
  }
}