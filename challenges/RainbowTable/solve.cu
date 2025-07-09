#include "solve.h"
#include <cuda_runtime.h>

__device__ unsigned int fnv1a_hash(int input) {
  constexpr unsigned int FNV_PRIME = 16777619;
  constexpr unsigned int OFFSET_BASIS = 2166136261;

  unsigned int hash = OFFSET_BASIS;
#pragma unroll
  for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
    unsigned char byte = (input >> (byte_pos * 8)) & 0xFF;
    hash = (hash ^ byte) * FNV_PRIME;
  }

  return hash;
}

__global__ void fnv1a_hash_kernel(const int *input, unsigned int *output, int N, int R) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) { return; }
  unsigned int hash = input[idx];
  for (int r = 0; r < R; ++r) {
    hash = fnv1a_hash(hash);
  }
  output[idx] = hash;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
#ifdef LOCAL_MACHINE
static
#endif
void solve(const int *input, unsigned int *output, int N, int R) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  fnv1a_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, R);
  cudaDeviceSynchronize();
}
#include "cuda_common.cuh"
std::vector<unsigned int> RainbowTable(const std::vector<int> &numbers, int R) {
  common::device_static_vector<int> d_numbers(numbers);
  common::device_static_vector<unsigned int> d_hashs(d_numbers.size());
  solve(d_numbers.data(), d_hashs.data(), d_numbers.size(), R);
  return d_hashs.to_vector();
}
