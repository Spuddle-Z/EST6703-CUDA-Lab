#include <iostream>

constexpr int M = 20400;
constexpr int N = 2048;
constexpr int K = 8192;

static float t_base, t_tiling, t_prefetch;
static cudaStream_t stream;

// 所有元素赋值1.0
void fill_mat_1(float* mat, const int m, const int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      mat[i * n + j] = 1.;
    }
  }
}

// 基础版本的GEMM内核
__global__ void gemm_base_kernel(const float* A, const float* B, float* C, const int M, const int N, const int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;  // 计算行索引
  int col = blockIdx.x * blockDim.x + threadIdx.x;  // 计算列索引
  
  if (row < M && col < N) {
    float sum = 0.;
    for (int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

constexpr int TILE_WIDTH = 16;

// 分块优化版本的GEMM内核
__global__ void gemm_tiling_kernel(const float* A, const float* B, float* C, const int M, const int N, const int K) {
  __shared__ float As[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  float pSum = 0.;
  for (int p = 0; p < K / TILE_WIDTH; ++p) {
    As[ty][tx] = A[row * K + p * TILE_WIDTH + tx];
    Bs[ty][tx] = B[(ty + p * TILE_WIDTH) * N + col];
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k) {
      pSum += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
  }
  C[row * N + col] = pSum;
}

// 分块+双缓冲预取版本的GEMM内核
__global__ void gemm_prefetch_kernel(const float* A, const float* B, float* C, const int M, const int N, const int K) {
  // 假设维度均为TILE_WIDTH的整数倍以去掉分支，提高吞吐
  __shared__ float As[2][TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bs[2][TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;
  int aRowBase = row * K;

  const int numTiles = K / TILE_WIDTH;  // K整除TILE_WIDTH
  int curr = 0;
  int next = 1;

  // 预取第0块
  int kBase = 0;
  As[curr][ty][tx] = A[aRowBase + kBase + tx];
  Bs[curr][ty][tx] = B[(kBase + ty) * N + col];
  __syncthreads();

  float pSum = 0.f;
  #pragma unroll 1
  for (int tile = 0; tile < numTiles; ++tile) {
    #pragma unroll
    for (int k = 0; k < TILE_WIDTH; ++k) {
      pSum += As[curr][ty][k] * Bs[curr][k][tx];
    }

    // 预取下一块
    if (tile + 1 < numTiles) {
      kBase = (tile + 1) * TILE_WIDTH;
      As[next][ty][tx] = A[aRowBase + kBase + tx];
      Bs[next][ty][tx] = B[(kBase + ty) * N + col];
    }
    __syncthreads();
    curr ^= 1;
    next ^= 1;
  }

  C[row * N + col] = pSum;
}

float* gemm_base() {
  // 分配主机内存
  float* A = (float*)malloc(M * K * sizeof(float));
  float* B = (float*)malloc(K * N * sizeof(float));
  float* C = (float*)malloc(M * N * sizeof(float));
  
  // 初始化矩阵
  fill_mat_1(A, M, K);
  fill_mat_1(B, K, N);
  
  // 分配设备内存
  float *cuA, *cuB, *cuC;
  cudaMalloc((void **)&cuA, M * K * sizeof(float));
  cudaMalloc((void **)&cuB, K * N * sizeof(float));
  cudaMalloc((void **)&cuC, M * N * sizeof(float));
  
  // 拷贝数据到设备
  cudaMemcpy(cuA, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cuB, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
  
  // 配置内核参数
  dim3 blockDim(32, 32);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
  
  // 测量执行时间
  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);
  
  cudaEventRecord(begin, stream);
  gemm_base_kernel<<<gridDim, blockDim, 0, stream>>>(cuA, cuB, cuC, M, N, K);
  cudaEventRecord(end, stream);
  
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&t_base, begin, end);
  
  // 拷贝结果回主机
  cudaMemcpy(C, cuC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  
  // 释放内存
  cudaFree(cuA);
  cudaFree(cuB);
  cudaFree(cuC);
  free(A);
  free(B);
  return C;
}

float* gemm_tiling() {
  float* A = (float*)malloc(M * K * sizeof(float));
  float* B = (float*)malloc(K * N * sizeof(float));
  float* C = (float*)malloc(M * N * sizeof(float));

  fill_mat_1(A, M, K);
  fill_mat_1(B, K, N);

  float *cuA, *cuB, *cuC;
  cudaMalloc((void **)&cuA, M * K * sizeof(float));
  cudaMalloc((void **)&cuB, K * N * sizeof(float));
  cudaMalloc((void **)&cuC, M * N * sizeof(float));

  cudaMemcpy(cuA, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cuB, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaEventRecord(begin, stream);
  gemm_tiling_kernel<<<gridDim, blockDim, 0, stream>>>(cuA, cuB, cuC, M, N, K);
  cudaEventRecord(end, stream);

  cudaEventSynchronize(end);
  cudaEventElapsedTime(&t_tiling, begin, end);

  cudaMemcpy(C, cuC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(cuA);
  cudaFree(cuB);
  cudaFree(cuC);
  free(A);
  free(B);
  return C;
}

float* gemm_prefetch() {
  float* A = (float*)malloc(M * K * sizeof(float));
  float* B = (float*)malloc(K * N * sizeof(float));
  float* C = (float*)malloc(M * N * sizeof(float));

  fill_mat_1(A, M, K);
  fill_mat_1(B, K, N);

  float *cuA, *cuB, *cuC;
  cudaMalloc((void **)&cuA, M * K * sizeof(float));
  cudaMalloc((void **)&cuB, K * N * sizeof(float));
  cudaMalloc((void **)&cuC, M * N * sizeof(float));

  cudaMemcpy(cuA, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cuB, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaEventRecord(begin, stream);
  gemm_prefetch_kernel<<<gridDim, blockDim, 0, stream>>>(cuA, cuB, cuC, M, N, K);
  cudaEventRecord(end, stream);

  cudaEventSynchronize(end);
  cudaEventElapsedTime(&t_prefetch, begin, end);

  cudaMemcpy(C, cuC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(cuA);
  cudaFree(cuB);
  cudaFree(cuC);
  free(A);
  free(B);
  return C;
}

// 主函数
int main()
{
  cudaStreamCreate(&stream);

  // 计算理论计算量
  constexpr size_t TFLOP = 2 * (size_t)M * (size_t)N * (size_t)K;

  std::cout << "start..." << std::endl;
  float *mat_base = gemm_base();
  // check_mat(mat_base, M, N, K);
  free(mat_base);
  std::cout << "kernel base: " << t_base << "ms  TFLOPs: "
            << TFLOP / t_base / 1e9
            << std::endl;  
  float *mat1 = gemm_tiling();
  // check_mat(mat1, M, N, K);
  free(mat1);
  std::cout << "kernel tiling: " << t_tiling << "ms  TFLOPs: "
            << TFLOP / t_tiling / 1e9
            << std::endl; 
  float *mat_prefetch = gemm_prefetch();
  // check_mat(mat_prefetch, M, N, K);
  free(mat_prefetch);
  std::cout << "kernel prefetch: " << t_prefetch << "ms  TFLOPs: "
            << TFLOP / t_prefetch / 1e9
            << std::endl; 
  return 0;
}