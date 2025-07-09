#include "solve.h"
#include <cuda_runtime.h>
__device__ static void memsetSharedZero(int *arr, const int size) {
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    arr[i] = 0;
  }
}
__device__ static void memAddShared(int *global_dest, int *shared_src, const int size) {
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    if (shared_src[i]) {
      atomicAdd(global_dest + i, shared_src[i]);
    }
  }
}
__global__ static void histogramming_kernal(const int *input, int *histogram, int N, int num_bins) {
  if (blockIdx.x == 0) {
    memsetSharedZero(histogram, num_bins);
    __syncthreads();
  }
  __shared__ int histogram_shared[1024];
  memsetSharedZero(histogram_shared, num_bins);
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  __syncthreads();
  if (i < N) {
    atomicAdd(histogram_shared + input[i], 1);
  }
  __syncthreads();
  memAddShared(histogram, histogram_shared, num_bins);
  __syncthreads();
}
// input, histogram are device pointers
#ifdef LOCAL_MACHINE
static
#endif
void solve(const int *input, int *histogram, int N, int num_bins) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  histogramming_kernal<<<blocksPerGrid, threadsPerBlock>>>(input, histogram, N, num_bins);
  cudaDeviceSynchronize();
}
#include "cuda_common.cuh"
std::vector<int> Histogramming(const std::vector<int> &input, int num_bins) {
  common::device_static_vector<int> d_input(input);
  common::device_static_vector<int> d_histogram(num_bins);
  solve(d_input.data(), d_histogram.data(), d_input.size(), num_bins);
  return d_histogram.to_vector();
}
