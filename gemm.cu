#include <iostream>

constexpr int M = 20400;
constexpr int N = 2048;
constexpr int K = 8192;

static float t_base, t_block;
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
  
  if (row < M && col < N) {  // 边界检查
    float sum = 0.;
    // 每个线程计算C的一个元素，需要K次乘加运算
    for (int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

// 

float* gemm0() {
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

// 主函数
int main()
{
  cudaStreamCreate(&stream);

  // 计算理论计算量
  constexpr size_t TFLOP = 2 * (size_t)M * (size_t)N * (size_t)K;

  std::cout << "start..." << std::endl;
  float *mat_base = gemm0();
  // check_mat(mat_base, M, N, K);
  free(mat_base);
  std::cout << "kernel0: " << t_base << "ms  TFLOPs: "
            << TFLOP / t_base / 1e9
            << std::endl;  
  // float *mat1 = gemm1();
  // // check_mat(mat1, M, N, K);
  // free(mat1);
  // std::cout << "kernel1: " << t1 << "ms  TFLOPs: "
  //           << TFLOP / t1 / 1e9
  //           << std::endl; 
  // float* mat2 = gemm2();
  // check_mat(mat2, M, N, K);
  // free(mat2);
  // std::cout << "kernel2: " << t2 << "ms  TFLOPs: "
  //           << TFLOP / t2 / 1e9
  //           << std::endl; 
  return 0;
}