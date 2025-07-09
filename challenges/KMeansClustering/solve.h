#pragma once
#include <vector>

struct KMeansClusteringRes {
  std::vector<int> labels;
  std::vector<float> final_centroid_x;
  std::vector<float> final_centroid_y;
};

KMeansClusteringRes KMeansClustering(
  const std::vector<float> &data_x,
  const std::vector<float> &data_y,
  const std::vector<float> &initial_centroid_x,
  const std::vector<float> &initial_centroid_y,
  int max_iterations
);
