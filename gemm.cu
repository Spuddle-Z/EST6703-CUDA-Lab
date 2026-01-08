#include <cuda_runtime.h>
#include <iostream>
#include <memory>

constexpr int M = 20400;
constexpr int N = 2048;
constexpr int K = 8192;
constexpr int TILE_WIDTH = 32;

static float t_base, t_tiling, t_prefetch;
static cudaStream_t stream;

// 将矩阵填充为1.0，单层循环便于编译器展开
inline void fill_ones(float* mat, const size_t count) {
  for (size_t idx = 0; idx < count; ++idx) {
    mat[idx] = 1.f;
  }
}

// 基础版本的GEMM内核
__global__ void gemm_base_kernel(const float* A, const float* B, float* C, const int m, const int n, const int kDim) {
  const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

  if (row >= m || col >= n) return;

  float acc = 0.f;
  for (int k = 0; k < kDim; ++k) {
    acc = fmaf(A[row * kDim + k], B[k * n + col], acc);
  }
  C[row * n + col] = acc;
}

// 分块优化版本的GEMM内核
__global__ void gemm_tiling_kernel(const float* A, const float* B, float* C, const int m, const int n, const int kDim) {
  __shared__ float As[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.y * TILE_WIDTH + ty;
  const int col = blockIdx.x * TILE_WIDTH + tx;

  if (row >= m || col >= n) return;

  float acc = 0.f;
  const int numTiles = kDim / TILE_WIDTH;
  for (int tile = 0; tile < numTiles; ++tile) {
    const int kBase = tile * TILE_WIDTH;
    As[ty][tx] = A[row * kDim + kBase + tx];
    Bs[ty][tx] = B[(kBase + ty) * n + col];
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE_WIDTH; ++k) {
      acc = fmaf(As[ty][k], Bs[k][tx], acc);
    }
    __syncthreads();
  }

  C[row * n + col] = acc;
}

// 分块+双缓冲预取版本的GEMM内核
__global__ void gemm_prefetch_kernel(const float* A, const float* B, float* C, const int m, const int n, const int kDim) {
  __shared__ float As[2][TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bs[2][TILE_WIDTH][TILE_WIDTH];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.y * TILE_WIDTH + ty;
  const int col = blockIdx.x * TILE_WIDTH + tx;

  if (row >= m || col >= n) return;

  const int numTiles = kDim / TILE_WIDTH;
  int curr = 0;
  int next = 1;
  int kBase = 0;

  const int rowBase = row * kDim;
  As[curr][ty][tx] = A[rowBase + kBase + tx];
  Bs[curr][ty][tx] = B[(kBase + ty) * n + col];
  __syncthreads();

  float acc = 0.f;
  for (int tile = 0; tile < numTiles; ++tile) {
    if (tile + 1 < numTiles) {
      kBase = (tile + 1) * TILE_WIDTH;
      As[next][ty][tx] = A[rowBase + kBase + tx];
      Bs[next][ty][tx] = B[(kBase + ty) * n + col];
    }

    #pragma unroll
    for (int k = 0; k < TILE_WIDTH; ++k) {
      acc = fmaf(As[curr][ty][k], Bs[curr][k][tx], acc);
    }

    __syncthreads();
    curr ^= 1;
    next ^= 1;
  }

  C[row * n + col] = acc;
}

using GemmKernel = void(*)(const float*, const float*, float*, const int, const int, const int);

float* run_gemm_kernel(GemmKernel kernel, float& elapsedMs) {
  const size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
  const size_t bytesB = static_cast<size_t>(K) * N * sizeof(float);
  const size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);

  std::unique_ptr<float[]> hostA(new float[M * K]);
  std::unique_ptr<float[]> hostB(new float[K * N]);
  std::unique_ptr<float[]> hostC(new float[M * N]);

  fill_ones(hostA.get(), M * K);
  fill_ones(hostB.get(), K * N);

  float *devA = nullptr, *devB = nullptr, *devC = nullptr;
  cudaMalloc(&devA, bytesA);
  cudaMalloc(&devB, bytesB);
  cudaMalloc(&devC, bytesC);

  cudaMemcpy(devA, hostA.get(), bytesA, cudaMemcpyHostToDevice);
  cudaMemcpy(devB, hostB.get(), bytesB, cudaMemcpyHostToDevice);

  const dim3 block(TILE_WIDTH, TILE_WIDTH);
  const dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, stream);
  kernel<<<grid, block, 0, stream>>>(devA, devB, devC, M, N, K);
  cudaEventRecord(stop, stream);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedMs, start, stop);

  cudaMemcpy(hostC.get(), devC, bytesC, cudaMemcpyDeviceToHost);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);

  return hostC.release();
}

float* gemm_base() {
  return run_gemm_kernel(gemm_base_kernel, t_base);
}

float* gemm_tiling() {
  return run_gemm_kernel(gemm_tiling_kernel, t_tiling);
}

float* gemm_prefetch() {
  return run_gemm_kernel(gemm_prefetch_kernel, t_prefetch);
}

int main() {
  cudaStreamCreate(&stream);

  constexpr size_t totalFlops = 2ULL * static_cast<size_t>(M) * static_cast<size_t>(N) * static_cast<size_t>(K);

  std::cout << "start..." << std::endl;

  float* mat_base = gemm_base();
  free(mat_base);
  std::cout << "kernel base: " << t_base << "ms  TFLOPs: "
            << totalFlops / t_base / 1e9 << std::endl;

  float* mat_tiling = gemm_tiling();
  free(mat_tiling);
  std::cout << "kernel tiling: " << t_tiling << "ms  TFLOPs: "
            << totalFlops / t_tiling / 1e9 << std::endl;

  float* mat_prefetch = gemm_prefetch();
  free(mat_prefetch);
  std::cout << "kernel prefetch: " << t_prefetch << "ms  TFLOPs: "
            << totalFlops / t_prefetch / 1e9 << std::endl;

  return 0;
}