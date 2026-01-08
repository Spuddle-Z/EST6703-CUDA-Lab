# CUDA GEMM Lab Report

## 1. 实验目的
- 熟悉 CUDA 编程与内核性能分析流程。
- 在 GPU 上实现并优化通用矩阵乘法 $C = A \times B$（FP32）。
- 比较 CPU 与 GPU（多种优化）的性能，给出加速比并用 `ncu` 进行瓶颈分析。

## 2. 实验环境
请在完成测试后填写：
- 硬件：GPU 型号、SM 数量、显存；CPU 型号与核心数；内存大小。
- 软件：OS、CUDA Toolkit 版本、NVIDIA 驱动版本、nvcc 版本、`ncu` 版本。
- 编译命令：`nvcc -O3 gemm.cu -o gemm`（如有额外编译选项请注明，例如 `-lineinfo` 方便分析）。

## 3. 问题规模与数据类型
- 矩阵维度：当前代码使用 $M = 20400,\ N = 2048,\ K = 8192$（FP32）。
- 理论浮点操作量：$2 \times M \times N \times K$ 次 FMA，对应 TFLOPs 计算公式：
  $$\text{TFLOPs} = \frac{2MNK}{\text{time(ms)} \times 10^9}$$
- 备注：作业要求中的 20480 可按需调整代码中的 `M`，并复测所有数据。

## 4. 理论与代码实现（对应 `gemm.cu`）
**4.1 理论思路**
- GEMM 算法：$C_{ij} = \sum_k A_{ik} B_{kj}$，计算复杂度 $O(MNK)$，数据复用对带宽友好。
- 分块/共享内存：将 A、B 子块搬入 `__shared__`，提升缓存命中并降低全局访存；块大小受寄存器、共享内存与占用率约束。
- 预取/双缓冲：在计算当前 tile 时加载下一 tile，尝试重叠内存访问与计算，减少访存 stall。
- TFLOPs 评估：利用 FMA 计为 2 次浮点操作，结合计时衡量吞吐，对比不同优化策略的提升。

**4.2 代码实现要点**
- `gemm_base_kernel`：直接 global 访问的 i-j-k 循环，`fmaf` 累加，含边界保护。
- `gemm_tiling_kernel`：
   - `TILE_WIDTH=32`，`As`/`Bs` 为共享内存 tile。
   - `numTiles = K / TILE_WIDTH`，每次加载一块并在块内展开计算。
- `gemm_prefetch_kernel`：
   - 双缓冲 `As[2]`、`Bs[2]` 交替使用；先装载 tile0，循环中预取 tile(t+1)。
   - `#pragma unroll` 展开内层乘加，减少循环开销。
- 宿主端：
   - `fill_ones` 填充输入；`run_gemm_kernel` 统一封装分配/拷贝/计时/回收，复用 launch 逻辑。
   - 主函数依次调用三种内核，输出时间与 TFLOPs，便于对比。

## 5. 测试方法
1. **CPU 基线**（需补充实现或使用 BLAS）：
   - 可用单线程或 OpenMP BLAS (如 OpenBLAS) 计算同规模 GEMM；记录时间 `t_cpu`。
2. **GPU 运行**：
   - 命令：`./gemm`（或在 Windows 下 `gemm.exe`）。
   - 记录输出中的 `t_base`、`t_tiling`、`t_prefetch`（单位 ms）。
3. **性能计数器（ncu）**：
   - 示例：`ncu --set full --kernel-name regex:gemm_.* ./gemm`。
   - 关注指标：SM 利用率、achieved occupancy、DRAM Throughput、L2/TEX Cache Hit Rate、Instruction Throughput、Replay/STALL 原因等。

## 6. 实验结果（请填入实测数据）
| 方案 | 运行时间 (ms) | TFLOPs | 相对基线加速比 |
| --- | --- | --- | --- |
| CPU (参考) |  |  |  |
| GPU 基线 `gemm_base` |  |  | 1.00x |
| GPU 分块 `gemm_tiling` |  |  |  |
| GPU 分块+预取 `gemm_prefetch` |  |  |  |

- 若将 M 调整为 20480，请增加一组同表格数据或替换上表。

## 7. 性能分析与调优效果
- **共享内存分块**：减少全局访存次数，提升数据重用；预期带宽受限程度下降，SM 效率提高。
- **双缓冲预取**：访存与计算重叠，降低 `ld/st` 引起的 stall；对大 K 尺寸效果更明显。
- **潜在瓶颈**（依据 ncu 报告填写）：
  - 是否受 DRAM 带宽限制？L2 命中率如何？
  - Occupancy 是否被寄存器或共享内存限制？
  - 指令层面是否存在 warp replay / branch divergence？

## 8. 进一步优化思路（可选）
- 调整 `TILE_WIDTH`（如 16/32/64）与 block 配置，寻找寄存器占用与 occupancy 的平衡。
- 使用 `cudaMemcpyAsync` + 流并行覆盖 H2D/D2H（当前分次分配不可并发）。
- 考虑 Tensor Core（WMMA / cuBLAS）实现对比 FP32 FMA 基线。
- 引入矩阵对齐与向量化加载（`float4`）以减少访存指令数。

## 9. 结论
- 概述各方案的性能提升幅度与主要原因。
- 结合 `ncu` 证据说明瓶颈与改进效果。
- 给出最终推荐的配置与后续可探索方向。
