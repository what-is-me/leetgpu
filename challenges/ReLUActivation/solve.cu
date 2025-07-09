#include "solve.h"
#include <cuda_runtime.h>

__global__ void relu_kernel(const float *input, float *output, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    output[i] = input[i] > 0 ? input[i] : 0;
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
#ifdef LOCAL_MACHINE
static
#endif
void solve(const float *input, float *output, int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
  cudaDeviceSynchronize();
}

#include "cuda_common.cuh"
std::vector<float> ReLUActivation(const std::vector<float> &input) {
  common::device_static_vector<float> data(input);
  common::device_static_vector<float> output(input.size());
  solve(input.data(), output.data(), input.size());
  return output.to_vector();
}
