#include <gtest/gtest.h>

#include "solve.h"
TEST(Softmax, T1) {
  const std::vector<float> input = {1.0, 2.0, 3.0};
  const std::vector<float> expected = {0.090, 0.244, 0.665};
  const std::vector<float> output = Softmax(input);
  EXPECT_EQ(output.size(), expected.size());
  for (int i = 0; i < output.size(); i++) {
    EXPECT_FLOAT_EQ(output[i], expected[i]);
  }
}
TEST(Softmax, T2) {
  const std::vector<float> input = {-10.0, -5.0, 0.0, 5.0, 10.0};
  const std::vector<float> expected = {2.04e-09, 4.52e-07, 9.99e-01, 2.26e-02,
                                       9.77e-01};
  const std::vector<float> output = Softmax(input);
  EXPECT_EQ(output.size(), expected.size());
  for (int i = 0; i < output.size(); i++) {
    EXPECT_FLOAT_EQ(output[i], expected[i]);
  }
}

unsigned int fnv1a_hash_bytes(const unsigned char* data, int length) {
  const unsigned int FNV_PRIME = 16777619;
  const unsigned int OFFSET_BASIS = 2166136261;

  unsigned int hash = OFFSET_BASIS;
  for (int i = 0; i < length; i++) {
    hash = (hash ^ data[i]) * FNV_PRIME;
  }
  return hash;
}