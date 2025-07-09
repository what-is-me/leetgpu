#include <gtest/gtest.h>

#include "solve.h"
TEST(KMeansClustering, T1) {
  const std::vector<float> data_x = {1.0, 1.5, 1.2,  1.3,  1.1,  5.0,  5.2, 5.1,
                                     5.3, 5.4, 10.1, 10.2, 10.0, 10.3, 10.5};
  const std::vector<float> data_y = {1.0, 1.5, 1.2,  1.3,  1.1,  5.0,  5.2, 5.1,
                                     5.3, 5.4, 10.1, 10.2, 10.0, 10.3, 10.5};
  const std::vector<float> initial_centroid_x = {3.4, 7.1, 8.5};
  const std::vector<float> initial_centroid_y = {3.4, 7.1, 8.5};
  constexpr int max_iterations = 20;
  auto result = KMeansClustering(data_x, data_y, initial_centroid_x,
                                 initial_centroid_y, max_iterations);

  const std::vector<int> labels = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2};
  const std::vector<float> final_centroid_x = {1.22, 5.2, 10.22};
  const std::vector<float> final_centroid_y = {1.22, 5.2, 10.22};

  EXPECT_EQ(labels, result.labels);
  for (int i = 0; i < final_centroid_x.size(); i++) {
    EXPECT_FLOAT_EQ(final_centroid_x[i], result.final_centroid_x[i]);
    EXPECT_FLOAT_EQ(final_centroid_y[i], result.final_centroid_y[i]);
  }
}

TEST(KMeansClustering, T2) {
  const std::vector<float> data_x = {1.,  1.2, 1.1, 1.3, 1.5,
                                     1.4, 1.6, 1.2, 1.3, 1.1};
  const std::vector<float> data_y = {1.,  1.2, 1.1, 1.3, 1.5,
                                     1.4, 1.6, 1.2, 1.3, 1.1};
  const std::vector<float> initial_centroid_x = {1., 5., 10.};
  const std::vector<float> initial_centroid_y = {1., 5., 10.};
  constexpr int max_iterations = 10;
  auto result = KMeansClustering(data_x, data_y, initial_centroid_x,
                                 initial_centroid_y, max_iterations);

  const std::vector<int> labels = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const std::vector<float> final_centroid_x = {1.27, 5., 10.};
  const std::vector<float> final_centroid_y = {1.27, 5., 10.};

  EXPECT_EQ(labels, result.labels);
  for (int i = 0; i < final_centroid_x.size(); i++) {
    EXPECT_FLOAT_EQ(final_centroid_x[i], result.final_centroid_x[i]);
    EXPECT_FLOAT_EQ(final_centroid_y[i], result.final_centroid_y[i]);
  }
}

// data_x =
// [ 1.   1.5  1.2  1.3  1.1  5.   5.2  5.1  5.3  5.4 10.1 10.2 10.  10.3
//  10.5]
// data_y =
// [ 1.   1.5  1.2  1.3  1.1  5.   5.2  5.1  5.3  5.4 10.1 10.2 10.  10.3
//  10.5]
// initial_centroid_x = [3.4 7.1 8.5]
// initial_centroid_y = [3.4 7.1 8.5]
// sample_size = 15
// k = 3
// max_iterations = 1
// Output: [0 0 0 0 0 1 1 1 1 1 2 2 2 2 2]
// Expected: [0 0 0 0 0 0 0 0 1 1 2 2 2 2 2]
// Max difference: 1

TEST(KMeansClustering, T3) {
  const std::vector<float> data_x = {1.,  1.5, 1.2,  1.3,  1.1, 5.,   5.2, 5.1,
                                     5.3, 5.4, 10.1, 10.2, 10., 10.3, 10.5};
  const std::vector<float> data_y = {1.,  1.5, 1.2,  1.3,  1.1, 5.,   5.2, 5.1,
                                     5.3, 5.4, 10.1, 10.2, 10., 10.3, 10.5};
  const std::vector<float> initial_centroid_x = {3.4, 7.1, 8.5};
  const std::vector<float> initial_centroid_y = {3.4, 7.1, 8.5};
  constexpr int max_iterations = 1;
  auto result = KMeansClustering(data_x, data_y, initial_centroid_x,
                                 initial_centroid_y, max_iterations);

  const std::vector<int> labels = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2};
  const std::vector<float> final_centroid_x = {1.27, 5., 10.};
  const std::vector<float> final_centroid_y = {1.27, 5., 10.};

  EXPECT_EQ(labels, result.labels);
  for (int i = 0; i < final_centroid_x.size(); i++) {
    EXPECT_FLOAT_EQ(final_centroid_x[i], result.final_centroid_x[i]);
    EXPECT_FLOAT_EQ(final_centroid_y[i], result.final_centroid_y[i]);
  }
}