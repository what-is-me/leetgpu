#include <cuda_runtime.h>

#include "solve.h"

// FNV-1a hash function that takes a byte array and its length as input
// Returns a 32-bit unsigned integer hash value
__device__ __forceinline__ static unsigned int fnv1a_hash_bytes(
    const unsigned char* data, int length) {
  constexpr unsigned int OFFSET_BASIS = 2166136261;
  unsigned int hash = OFFSET_BASIS;
  for (int i = 0; i < length; i++) {
    constexpr unsigned int FNV_PRIME = 16777619;
    hash = (hash ^ data[i]) * FNV_PRIME;
  }
  return hash;
}

__device__ __forceinline__ static void decode(unsigned int code,
                                              const int password_length,
                                              unsigned char* password) {
  for (int i = 0; i < password_length; i++) {
    password[i] = 'a' + code % 26;
    code /= 26;
  }
}

__device__ __forceinline__ static unsigned int fnv1a_hash_rounds(
    const unsigned char* data, int length, int R) {
  unsigned int hash = fnv1a_hash_bytes(data, length);
  if (R == 1) {
    return hash;
  }
  for (int i = 1; i < R; i++) {
    hash = fnv1a_hash_bytes(reinterpret_cast<const unsigned char*>(&hash), 4);
  }
  return hash;
}

__global__ static void password_cracking_kernal(const unsigned int target_hash,
                                                const int password_length,
                                                const int R,
                                                const unsigned int N,
                                                char* output_password) {
  if (const unsigned int code = blockIdx.x * blockDim.x + threadIdx.x;
      code < N) {
    unsigned char password[6];
    decode(code, password_length, password);
    if (fnv1a_hash_rounds(password, password_length, R) == target_hash) {
      memcpy(output_password, password, password_length);
    }
  }
}

// output_password is a device pointer
#ifdef LOCAL_MACHINE
static
#endif
    void
    solve(unsigned int target_hash, int password_length, int R,
          char* output_password) {
  constexpr unsigned int threadsPerBlock = 256;
  unsigned int N = 1;
  for (int i = 0; i < password_length; i++) N *= 26;
  const unsigned int blocksPerGrid =
      (N + threadsPerBlock - 1) / threadsPerBlock;
  password_cracking_kernal<<<blocksPerGrid, threadsPerBlock>>>(
      target_hash, password_length, R, N, output_password);
  cudaDeviceSynchronize();
}

#include "cuda_common.cuh"
std::string PasswordCracking(const unsigned int target_hash,
                             const int password_length, const int R) {
  common::device_static_vector<char> d_output_password(password_length + 1);
  solve(target_hash, password_length, R, d_output_password.data());
  const auto output_password = d_output_password.to_vector();
  return std::string(output_password.begin(), std::prev(output_password.end()));
}
