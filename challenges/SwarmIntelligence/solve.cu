#include <cuda_runtime.h>

#include "solve.h"
struct Vector2 {
  float x;
  float y;
};

__device__ __forceinline__ Vector2 operator+(Vector2 a, const Vector2 b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}

__device__ __forceinline__ Vector2 operator-(Vector2 a, const Vector2 b) {
  a.x -= b.x;
  a.y -= b.y;
  return a;
}

__device__ __forceinline__ Vector2 operator/(Vector2 a, const float f) {
  a.x /= f;
  a.y /= f;
  return a;
}

__device__ __forceinline__ Vector2 operator*(Vector2 a, const float f) {
  a.x *= f;
  a.y *= f;
  return a;
}

struct Agent {
  Vector2 p;
  Vector2 v;
};

__device__ __forceinline__ float pow2(const float x) { return x * x; }

__device__ __forceinline__ bool is_neighbor(const Agent& a, const Agent& b) {
  return pow2(a.p.x - b.p.x) + pow2(a.p.y - b.p.y) < 25.0f;
}

__global__ void calc_v_avg_kernal(const Agent* agents, const int N,
                                  Agent& agent_next, const int agent_idx) {
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  const bool valid_and_neighbor = idx < N && idx != agent_idx &&
                                  is_neighbor(agents[agent_idx], agents[idx]);
  auto [vx, vy] = valid_and_neighbor ? agents[idx].v : Vector2{0.0f, 0.0f};
  float cnt = valid_and_neighbor ? 1.0f : 0.0f;
  for (int offset = 16; offset > 0; offset >>= 1) {
    constexpr unsigned FULL_MASK = 0xffffffff;
    vx += __shfl_down_sync(FULL_MASK, vx, offset);
    vy += __shfl_down_sync(FULL_MASK, vy, offset);
    cnt += __shfl_down_sync(FULL_MASK, cnt, offset);
  }
  if (threadIdx.x % 32 == 0) {
    atomicAdd(&agent_next.p.x, cnt);  // x as cnt
    atomicAdd(&agent_next.v.x, vx);
    atomicAdd(&agent_next.v.y, vy);
  }
}

__global__ void calc_agent_next_kernal(const Agent* agents, Agent* agents_next,
                                       const int N) {
  if (const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N) {
    Vector2 v = agents[idx].v;
    if (const float cnt = agents_next[idx].p.x; cnt > 0.0f) {
      const auto v_avg = agents_next[idx].v / cnt;
      v = v + (v_avg - v) * 0.05;
    }
    agents_next[idx] = {agents[idx].p + v, v};
  }
}

static void solve_agent(const Agent* agents, Agent* agents_next, const int N) {
  constexpr int threadsPerBlock = 256;
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  for (int i = 0; i < N; i++) {
    calc_v_avg_kernal<<<blocksPerGrid, threadsPerBlock>>>(agents, N,
                                                          agents_next[i], i);
  }
  cudaDeviceSynchronize();
  calc_agent_next_kernal<<<blocksPerGrid, threadsPerBlock>>>(agents,
                                                             agents_next, N);
  cudaDeviceSynchronize();
}
// agents, agents_next are device pointers
#ifdef LOCAL_MACHINE
static
#endif
    void
    solve(const float* agents, float* agents_next, int N) {
  solve_agent(reinterpret_cast<const Agent*>(agents),
              reinterpret_cast<Agent*>(agents_next), N);
}
#include "cuda_common.cuh"
std::vector<float> SwarmIntelligence(const std::vector<float>& agents) {
  common::device_static_vector<float> d_agents(agents);
  common::device_static_vector<float> d_agents_next(agents.size());
  solve(d_agents.data(), d_agents_next.data(), d_agents.size() / 4);
  return d_agents_next.to_vector();
}
