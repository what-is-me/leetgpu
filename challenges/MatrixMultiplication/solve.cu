#include <cuda_runtime.h>

#include "solve.h"

template <typename T>
__forceinline__ __device__ __host__ T& MAT_AT(T* mat, int row_th, int col_th,
                                              int rows, int cols) {
  return mat[row_th * cols + col_th];
}

template <typename T>
__forceinline__ __device__ __host__ T MAT_AT_DEFAULT0(const T* mat, int row_th,
                                                      int col_th, int rows,
                                                      int cols) {
  return row_th < rows && col_th < cols ? mat[row_th * cols + col_th] : 0;
}

__global__ void matrix_multiplication_kernel(const float* A, const float* B,
                                             float* C, int M, int N, int K) {
  const auto base_x = blockIdx.x * blockDim.x;
  const auto base_y = blockIdx.y * blockDim.y;
  const auto dx = threadIdx.x;
  const auto dy = threadIdx.y;
  const auto x = base_x + dx;
  const auto y = base_y + dy;
  __shared__ float a_buf[16][16];
  __shared__ float b_buf[16][16];
  float c_cell = 0;
  for (int i = 0; i < N; i += 16) {
    __syncthreads();
    a_buf[dy][dx] = MAT_AT_DEFAULT0(A, y, i + dx, M, N);
    b_buf[dx][dy] = MAT_AT_DEFAULT0(B, i + dy, x, N, K);
    __syncthreads();

#pragma unroll
    for (int j = 0; j < 16; ++j) {
      c_cell += a_buf[dy][j] * b_buf[dx][j];
    }
  }
  if (y < M && x < K) {
    MAT_AT(C, y, x, M, K) = c_cell;
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
#ifdef LOCAL_MACHINE
static
#endif
    void
    solve(const float* A, const float* B, float* C, int M, int N, int K) {
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
  matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M,
                                                                   N, K);
  cudaDeviceSynchronize();
}


#include "cuda_common.cuh"
std::vector<float> MatrixMultiplication(const std::vector<float>& A,
                                        const std::vector<float>& B, int M,
                                        int N, int K) {
  common::device_static_vector<float> d_A(A);
  common::device_static_vector<float> d_B(B);
  common::device_static_vector<float> d_C(M * K);
  solve(d_A.data(), d_B.data(), d_C.data(), M, N, K);
  return d_C.to_vector();
}
