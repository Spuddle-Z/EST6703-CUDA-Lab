nvcc -arch=sm_86 gemm.cu -o gemm
gemm
ncu -f -o gemm_rep --cache-control none --set full --target-processes all gemm