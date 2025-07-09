#include "solve.h"
#include <cuda_runtime.h>

__global__ static void reduction_kernal(const float *input, float *output, const int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  float val = i < N ? input[i] : 0;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    constexpr unsigned FULL_MASK = 0xffffffff;
    val += __shfl_down_sync(FULL_MASK, val, offset);
  }
  if (i % 32 == 0) {
    atomicAdd(output, val);
  }
}

// input, output are device pointers
#ifdef LOCAL_MACHINE
static
#endif
void solve(const float *input, float *output, int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  reduction_kernal<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
  cudaDeviceSynchronize();
}
#include "cuda_common.cuh"
float Reduction(const std::vector<float> &input) {
  common::device_static_vector<float> d_input(input);
  common::device_static_vector<float> d_output(1);
  solve(d_input.data(), d_output.data(), d_input.size());
  return d_output.to_vector().front();
}
