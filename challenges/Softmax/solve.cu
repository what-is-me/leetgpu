#include <cuda_runtime.h>

#include "solve.h"

constexpr unsigned FULL_MASK = 0xffffffff;
__global__ static void max_kernal(const float *input, float *res, const int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  float val = i < N ? input[i] : input[0];
  // if (i < N) {
  //   printf("%d %f\n", i, val);
  // }
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = max(val, __shfl_down_sync(FULL_MASK, val, offset));
    // if (i < N) {
    //   printf("down %d %f\n", i, val);
    // }
  }
  if (i % 32 == 0) {
    res[i / 32] = val;
  }
}

float reduce_max(const float *origin_input, int N) {
  constexpr int threadsPerBlock = 256;

  float *input, *output;
  cudaMalloc(&input, (N + 31) / 32 * sizeof(float));
  cudaMalloc(&output, ((N + 31) / 32 + 31) / 32 * sizeof(float));

  max_kernal<<<(N + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
      origin_input, input, N);
  cudaDeviceSynchronize();
  N = (N + 31) / 32;

  while (N > 1) {
    max_kernal<<<(N + threadsPerBlock - 1) / threadsPerBlock,
                 threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
    N = (N + 31) / 32;
    std::swap(input, output);
  }

  float res;
  cudaMemcpy(&res, input, N * sizeof(float), cudaMemcpyDeviceToHost);
  return res;
}

__global__ static void reduction_kernal(const float *input, const float max_val,
                                        float *output, float *res,
                                        const int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  float val = 0;
  if (i < N) {
    val = expf(input[i] - max_val);
    output[i] = val;
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(FULL_MASK, val, offset);
  }
  if (i % 32 == 0) {
    atomicAdd(res, val);
  }
}

__global__ void softmax_kernel(const float *sum_p, float *output, int N) {
  if (const int i = blockIdx.x * blockDim.x + threadIdx.x; i < N) {
    output[i] /= *sum_p;
  }
}

// input, output are device pointers
#ifdef LOCAL_MACHINE
static
#endif
    void
    solve(const float *input, float *output, int N) {
  float *sum_p;
  cudaMalloc(&sum_p, sizeof(float));
  cudaMemset(sum_p, 0, sizeof(float));

  constexpr int threadsPerBlock = 256;
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  const float max_val = reduce_max(input, N);
  //printf("max_val: %f\n", max_val);
  reduction_kernal<<<blocksPerGrid, threadsPerBlock>>>(input, max_val, output,
                                                       sum_p, N);
  cudaDeviceSynchronize();
  softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(sum_p, output, N);
  cudaDeviceSynchronize();
  cudaFree(sum_p);
}

#include "cuda_common.cuh"
std::vector<float> Softmax(const std::vector<float> &input) {
  common::device_static_vector<float> d_input(input);
  common::device_static_vector<float> d_output(input.size());
  solve(d_input.data(), d_output.data(), d_input.size());
  return d_output.to_vector();
}
