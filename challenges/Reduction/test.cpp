#include "solve.h"
#include <gtest/gtest.h>
TEST(Reduction, T1) {
  const std::vector<float> input = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  EXPECT_FLOAT_EQ(Reduction(input), 36.0);
}

TEST(Reduction, T2) {
  const std::vector<float> input = {-2.5, 1.5, -1.0, 2.0};
  EXPECT_FLOAT_EQ(Reduction(input), 0.0);
}
