#include "solve.h"
#include "gtest/gtest.h"

TEST(RainbowTable, T1) {
  std::vector<unsigned int> expect = {1636807824, 1273011621, 2193987222};
  std::vector<unsigned int> actual = RainbowTable({123, 456, 789}, 2);
  EXPECT_EQ(expect, actual);
}

TEST(RainbowTable, T2) {
  std::vector<unsigned int> expect = {96754810, 3571711400, 2006156166};
  std::vector<unsigned int> actual = RainbowTable({0, 1, 2147483647}, 3);
  EXPECT_EQ(expect, actual);
}
