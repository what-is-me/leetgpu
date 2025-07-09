#include <cuda_runtime.h>

#include "solve.h"
using uint = unsigned int;
__device__ __forceinline__ static float distance2(const float x_1,
                                                  const float y_1,
                                                  const float x_2,
                                                  const float y_2) {
  return (x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2);
}

__device__ __forceinline__ static void block_load_initial_centroid(
    const float *initial_centroid_x, const float *initial_centroid_y,
    const int k, float *s_initial_centroid_x, float *s_initial_centroid_y) {
  for (uint i = threadIdx.x; i < k; i += blockDim.x) {
    s_initial_centroid_x[i] = initial_centroid_x[i];
    s_initial_centroid_y[i] = initial_centroid_y[i];
  }
}

__global__ static void calc_label_kernal(const float *data_x,
                                         const float *data_y,
                                         const float *initial_centroid_x,
                                         const float *initial_centroid_y,
                                         const int sample_size, const int k,
                                         int *labels) {
  __shared__ float s_initial_centroid_x[100];
  __shared__ float s_initial_centroid_y[100];
  block_load_initial_centroid(initial_centroid_x, initial_centroid_y, k,
                              s_initial_centroid_x, s_initial_centroid_y);
  __syncthreads();
  if (const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
      idx < sample_size) {
    const float x = data_x[idx];
    const float y = data_y[idx];
    int label = -1;
    float min_distance = INFINITY;
    for (int i = 0; i < k; ++i) {
      const float distance =
          distance2(x, y, s_initial_centroid_x[i], s_initial_centroid_y[i]);
      if (distance < min_distance) {
        label = i;
        min_distance = distance;
      }
    }
    labels[idx] = label;
  }
}

static void calc_label(const float *data_x, const float *data_y,
                       const float *initial_centroid_x,
                       const float *initial_centroid_y, const int sample_size,
                       const int k, int *labels) {
  constexpr uint threadsPerBlock = 256;
  const uint blocksPerGrid =
      (sample_size + threadsPerBlock - 1) / threadsPerBlock;
  calc_label_kernal<<<blocksPerGrid, threadsPerBlock>>>(
      data_x, data_y, initial_centroid_x, initial_centroid_y, sample_size, k,
      labels);
  cudaDeviceSynchronize();
}

__device__ __forceinline__ static void block_init_final_centroid(
    float *s_final_centroid_x, float *s_final_centroid_y,
    int *s_final_centroid_count, const int k) {
  for (uint i = threadIdx.x; i < k; i += blockDim.x) {
    s_final_centroid_x[i] = 0;
    s_final_centroid_y[i] = 0;
    s_final_centroid_count[i] = 0;
  }
}

__device__ __forceinline__ static void block_add_final_centroid(
    const float *s_final_centroid_x, const float *s_final_centroid_y,
    const int *s_final_centroid_count, float *final_centroid_x,
    float *final_centroid_y, int *final_centroid_count, const int k) {
  for (uint i = threadIdx.x; i < k; i += blockDim.x) {
    atomicAdd(final_centroid_x + i, s_final_centroid_x[i]);
    atomicAdd(final_centroid_y + i, s_final_centroid_y[i]);
    atomicAdd(final_centroid_count + i, s_final_centroid_count[i]);
  }
}

__global__ static void calc_final_centroid_sum_and_cnt_kernal(
    const float *data_x, const float *data_y, const int *labels,
    float *final_centroid_x, float *final_centroid_y, int *final_centroid_count,
    const int sample_size, const int k) {
  __shared__ float s_final_centroid_x[100];
  __shared__ float s_final_centroid_y[100];
  __shared__ int s_final_centroid_count[100];
  block_init_final_centroid(s_final_centroid_x, s_final_centroid_y,
                            s_final_centroid_count, k);
  __syncthreads();
  if (const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
      idx < sample_size) {
    const float x = data_x[idx];
    const float y = data_y[idx];
    const int label = labels[idx];
    atomicAdd(s_final_centroid_x + label, x);
    atomicAdd(s_final_centroid_y + label, y);
    atomicAdd(s_final_centroid_count + label, 1);
  }
  __syncthreads();
  block_add_final_centroid(s_final_centroid_x, s_final_centroid_y,
                           s_final_centroid_count, final_centroid_x,
                           final_centroid_y, final_centroid_count, k);
}

__global__ static void calc_final_centroid_kernal(
    float *final_centroid_x, float *final_centroid_y,
    const float *initial_centroid_x, const float *initial_centroid_y,
    const int *final_centroid_count, const int k) {
  if (const uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < k) {
    const float x = final_centroid_x[idx];
    const float y = final_centroid_y[idx];
    const int count = final_centroid_count[idx];
    if (count > 0) {
      final_centroid_x[idx] = x / count;
      final_centroid_y[idx] = y / count;
    }else {
      final_centroid_x[idx] = initial_centroid_x[idx];
      final_centroid_y[idx] = initial_centroid_y[idx];
    }
  }
}

static void iterate_k_means(const float *data_x, const float *data_y,
                            const float *initial_centroid_x,
                            const float *initial_centroid_y,
                            const int sample_size, const int k,
                            /*output*/ int *labels, int *final_centroid_count,
                            float *final_centroid_x, float *final_centroid_y) {
  // 1. 计算label
  constexpr uint threadsPerBlock = 256;
  const uint blocksPerGrid =
      (sample_size + threadsPerBlock - 1) / threadsPerBlock;
  calc_label(data_x, data_y, initial_centroid_x, initial_centroid_y,
             sample_size, k, labels);
  // 2. 计算centroid
  cudaMemset(final_centroid_x, 0, k * sizeof(float));
  cudaMemset(final_centroid_y, 0, k * sizeof(float));
  cudaMemset(final_centroid_count, 0, k * sizeof(int));
  cudaDeviceSynchronize();
  calc_final_centroid_sum_and_cnt_kernal<<<blocksPerGrid, threadsPerBlock>>>(
      data_x, data_y, labels, final_centroid_x, final_centroid_y,
      final_centroid_count, sample_size, k);
  cudaDeviceSynchronize();
  calc_final_centroid_kernal<<<1, threadsPerBlock>>>(
      final_centroid_x, final_centroid_y, initial_centroid_x,
      initial_centroid_y, final_centroid_count, k);
  cudaDeviceSynchronize();
}

constexpr float epsilon = 0.0001;

__global__ static void need_iterate_kernal(const float *initial_centroid_x,
                                           const float *initial_centroid_y,
                                           const float *final_centroid_x,
                                           const float *final_centroid_y,
                                           const int k, int *res) {
  __shared__ int s_res[1];
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < k) {
    const float diff_x = fabsf(final_centroid_x[idx] - initial_centroid_x[idx]);
    const float diff_y = fabsf(final_centroid_y[idx] - initial_centroid_y[idx]);
    atomicOr(s_res, diff_x > epsilon || diff_y > epsilon);
  }
  __syncthreads();
  if (idx == 0) {
    *res = s_res[0];
  }
}

static bool need_iterate(const float *initial_centroid_x,
                         const float *initial_centroid_y,
                         const float *final_centroid_x,
                         const float *final_centroid_y, const int k,
                         int *d_res) {
  int res;
  need_iterate_kernal<<<1, 256>>>(initial_centroid_x, initial_centroid_y,
                                  final_centroid_x, final_centroid_y, k, d_res);
  cudaDeviceSynchronize();
  cudaMemcpy(&res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
  return res;
}

#ifdef LOCAL_MACHINE
static
#endif
    void
    solve(const float *data_x, const float *data_y, int *labels,
          float *initial_centroid_x, float *initial_centroid_y,
          float *final_centroid_x, float *final_centroid_y, int sample_size,
          int k, int max_iterations) {
  int *final_centroid_count;
  int *d_res;
  cudaMalloc(&final_centroid_count, sizeof(int) * k);
  cudaMalloc(&d_res, sizeof(int));
  for (int i = 0; i < max_iterations; ++i) {
    iterate_k_means(data_x, data_y, initial_centroid_x, initial_centroid_y,
                    sample_size, k, labels, final_centroid_count,
                    final_centroid_x, final_centroid_y);
    std::swap(initial_centroid_x, final_centroid_x);
    std::swap(initial_centroid_y, final_centroid_y);
    if (!need_iterate(initial_centroid_x, initial_centroid_y, final_centroid_x,
                      final_centroid_y, k, d_res)) {
      break;
    }
  }
  cudaMemcpy(final_centroid_x, initial_centroid_x, sizeof(float) * k,
             cudaMemcpyDeviceToDevice);
  cudaMemcpy(final_centroid_y, initial_centroid_y, sizeof(float) * k,
             cudaMemcpyDeviceToDevice);
  // calc_label(data_x, data_y, initial_centroid_x, initial_centroid_y,
  //            sample_size, k, labels);
  cudaFree(d_res);
  cudaFree(final_centroid_count);
}
#include "cuda_common.cuh"
KMeansClusteringRes KMeansClustering(
    const std::vector<float> &data_x, const std::vector<float> &data_y,
    const std::vector<float> &initial_centroid_x,
    const std::vector<float> &initial_centroid_y, int max_iterations) {
  common::device_static_vector<float> d_data_x(data_x);
  common::device_static_vector<float> d_data_y(data_y);
  common::device_static_vector<int> d_labels(data_x.size());
  common::device_static_vector<float> d_initial_centroid_x(initial_centroid_x);
  common::device_static_vector<float> d_initial_centroid_y(initial_centroid_y);
  common::device_static_vector<float> d_final_centroid_x(
      initial_centroid_x.size());
  common::device_static_vector<float> d_final_centroid_y(
      initial_centroid_y.size());
  solve(d_data_x.data(), d_data_y.data(), d_labels.data(),
        d_initial_centroid_x.data(), d_initial_centroid_y.data(),
        d_final_centroid_x.data(), d_final_centroid_y.data(), data_x.size(),
        initial_centroid_x.size(), max_iterations);
  return KMeansClusteringRes{d_labels.to_vector(),
                             d_final_centroid_x.to_vector(),
                             d_final_centroid_y.to_vector()};
}
