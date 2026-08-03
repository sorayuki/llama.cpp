# STQ1_0 Vulkan 优化实验汇总

## 1. 目标与当前结论

目标是缩小 STQ1_0 在 NVIDIA GeForce RTX 4090 Laptop GPU 上 Vulkan 与
CUDA 的推理速度差距，同时不降低提示词处理速度，也不破坏正确性。

当前保留的实现是：

- 基础提交：`38dea724df1c6c574697e7d5d5a628c3f9764ac3`
- 最新代码提交：`dcc281144`，`vulkan: optimize STQ1_0 packed loads`
- 提示词批处理继续使用 STQ1_0 全量反量化后执行 F16 矩阵乘。
- 单 token MMVQ 使用 STQ1_0 的 16 位打包只读视图。
- 每个 workgroup 计算 8 行输出。
- 没有改变 GGUF 中的物理数据格式、块大小或模型文件。

最终真实代码改动只有：

```text
ggml/src/ggml-vulkan/vulkan-shaders/types.glsl
ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq_funcs.glsl
```

最终 STQ1_0 大 workgroup pipeline 资源：

| 项目 | 数值 |
| --- | ---: |
| 每线程寄存器 | 48 |
| 最终机器码大小 | 355712 bytes |
| 共享内存 | 4224 bytes |
| 正确性 | 4/4 通过 |

最可信的结论：

- 提示词处理的原始优化明确有效，满功率 PP2048 约从 8015 提升到 9242 t/s。
- 16 位打包视图在低功率交错测试中约提升 STQ TG 4%，PP2048 无可测回退。
- 当前 35 W 环境的一次最终相邻对照为 CUDA 56.38 t/s、Vulkan 53.96 t/s，
  差约 4.3%。
- 35 W 下 GPU 时钟会在约 630 到 840 MHz 间明显漂移，不能把跨时段的
  39.8 到 54 t/s 全部视为代码收益。

## 2. 测试环境

### 模型和程序

- STQ1_0 模型：`D:\tools\llm\Hy-MT2-1.8B-1.25Bit.gguf`
- Q4 对照模型：`D:\tools\llm\Hy-MT2-1.8B-Q4_K_M.gguf`
- CUDA 程序：`D:\tools\llm\llama-hunyuan`
- Vulkan 构建目录：`D:\devspace\llama.cpp\build-vulkan-reldbg`
- GPU：NVIDIA GeForce RTX 4090 Laptop GPU
- 当前构建类型：RelWithDebInfo，benchmark 会报告 asserts enabled

### 功耗环境

满功率环境的默认功率限制为 80 W。后续主要实验使用低功率电源，
`nvidia-smi` 确认：

```text
Current Power Limit: 35.00 W
Default Power Limit: 80.00 W
SW Power Cap: Active
```

在线视频播放会占用 GPU，并明显污染推理测试。测试时应关闭视频、浏览器硬件
加速任务及其他 GPU 程序。

尝试使用 `nvidia-smi` 把 graphics clock 锁定到 690 MHz，但当前 Windows
用户没有修改 GPU 时钟的权限。锁频失败后没有执行 benchmark，也没有留下时钟
锁定状态。

## 3. 基准结果

### 满功率历史结果

| 后端 | PP2048 | TG128 |
| --- | ---: | ---: |
| Vulkan，提示词优化后 | 约 9242 t/s | 约 134.1 t/s |
| CUDA | 约 9644 t/s | 约 174.5 t/s |

提示词优化前 Vulkan PP2048 约为 8015 t/s。优化后输入速度差距从约 1000 t/s
缩小到约 400 t/s。

### 低功率早期基准

| 后端或实现 | 模型 | TG128 |
| --- | --- | ---: |
| CUDA | STQ1_0 | 58.746 +/- 0.662 t/s |
| Vulkan 原始提交 | STQ1_0 | 39.800 +/- 0.897 t/s |
| CUDA，第一轮 | Q4_K_M | 34.16 +/- 1.23 t/s |
| Vulkan | Q4_K_M | 34.83 +/- 1.28 t/s |
| CUDA，第二轮 | Q4_K_M | 32.05 +/- 1.11 t/s |

Q4 中 CUDA 和 Vulkan 基本处于同一范围，说明早期较大的 STQ 差距主要是
STQ1_0 实现问题，不是 Vulkan 后端天然比 CUDA 慢同样比例。

### 最终相邻低功率对照

| 后端或实现 | TG128 平均 | 最后 5 次平均 |
| --- | ---: | ---: |
| CUDA STQ1_0 | 56.38 t/s | 56.03 t/s |
| Vulkan packed16，8 行 | 53.96 t/s | 53.45 t/s |

平均差约 4.3%，最后 5 次差约 4.6%。这是当前低功率环境中的相邻结果，不能
直接替代满功率结论。

## 4. 明确有效并保留的方案

### 4.1 大批量提示词走反量化加 F16 矩阵乘

实现内容：

- batch 大于 4 时，不再强制所有 STQ1_0 运算走单 token MMVQ 路径。
- 先把 STQ1_0 反量化成 F16，再使用 Vulkan F16 矩阵乘路径。
- 修复 STQ1_0 反量化 shader 越界索引。
- 增加 `n=16` 的 STQ1_0 MUL_MAT 正确性覆盖。

效果：

- 满功率 PP2048 从约 8015 提升到约 9242 t/s。
- 与 CUDA 约 9644 t/s 的差距缩小到约 4%。
- 这是输入速度方面最明确、幅度最大的优化。

### 4.2 STQ1_0 16 位打包只读视图

增加与原物理布局等长的 shader 视图：

```glsl
struct block_stq1_0_packed16 {
    uint16_t qs[QUANT_K_STQ1_0/16];
    uint16_t sign[QUANT_K_STQ1_0/64];
    float16_t d;
};
```

每次 `stq1_0_pack_4()` 使用一次 16 位 `qs` 读取取得 4 个连续 code，并使用
一次 16 位 `sign` 读取取得对应 sign bit。原来按 `uint8_t` 数组动态索引的方式
需要更多窄读取和索引运算。

高状态交错结果：

| 顺序 | 实现 | TG128 |
| --- | --- | ---: |
| B1 | packed16 | 40.400 t/s |
| A | 原始实现 | 39.625 t/s |
| B2 | packed16 | 39.715 t/s |

低状态稳定交错结果：

| 顺序 | 实现 | TG128 |
| --- | --- | ---: |
| A1 | 原始实现 | 33.570 t/s |
| B1 | packed16 | 34.815 t/s |
| A2 | 原始实现 | 35.842 t/s |
| B2 | packed16 | 37.272 t/s |
| A3 | 原始实现 | 35.717 t/s |
| B3 | packed16 | 37.148 t/s |

结论：

- 稳定段约快 4%。
- packed16 PP2048 为 2479.1 t/s，原始实现为 2481.7 t/s，无可测 PP 回退。
- 大 workgroup 为 48 registers、355712-byte binary、4224-byte shared。
- 4/4 正确性通过。
- 最终保留并提交。

## 5. 基本无效、效果太小或无法复现的方案

### 5.1 Plane LUT

目的：根据 `p` 或 plane 调整表结构，减少动态提取。

结果：

- 原始实现 39.800 t/s。
- Plane LUT 39.943 t/s。
- 表面只快 0.36%，机器码和寄存器形态更差。

结论：差异低于功耗噪声，按无效处理，已撤销。

### 5.2 禁止外层 8 行循环展开

只对 STQ 的行循环加 `dont_unroll`，没有改变 K 循环。

静态结果：

- 大 workgroup 从 40 增到 48 registers。
- binary 从 446336 缩小到 125312 bytes。

动态结果：

- 实验 B1 为 38.039 t/s。
- 邻接原版 A4 为 35.235 t/s。
- 实验 B2 为 35.458 t/s。

结论：受功耗漂移影响，整体接近中性。机器码显著缩小但没有稳定吞吐收益。
未保留，可在满功率环境作为低优先级候选复测。

### 5.3 预加载两个 qs byte 和一个 sign byte

早期单次测量约快 0.7%，但后续 A/B/A 没有确认：

| 顺序 | 实现 | TG128 |
| --- | --- | ---: |
| A3 | 原始实现 | 35.19 +/- 0.42 t/s |
| B | preload | 36.07 +/- 1.11 t/s |
| A4 | 原始实现 | 37.55 +/- 0.88 t/s |

相对 A3/A4 插值基线约慢 0.8%；去掉每轮第一个样本后约慢 1.2%。

结论：早期正效果不能复现，已撤销。

### 5.4 Packed uint[8] codebook

把四个 8 位 qpack 打包进一个 `uint`，用 `uint[8]` 加变量移位代替
`uint[32]` 动态索引。

首次结果看起来有效：

| 顺序 | 实现 | TG128 |
| --- | --- | ---: |
| A4 | 原始实现 | 37.55 +/- 0.88 t/s |
| B | packed uint[8] | 38.16 +/- 0.54 t/s |
| A5 | 原始实现 | 37.51 +/- 0.57 t/s |

但后续不能复现：

- 把表放在 `mul_mat_vecq_funcs.glsl` 时只有 35.21 t/s，随后原版 38.47 t/s。
- 再次使用原来的 `types.glsl` 放置方式时为 38.03 t/s，约比邻接原版慢 1.1%。
- 遥测时 GPU utilization 约 94.7%，power 33.65 W，graphics clock 465 MHz。
- shader 37740 bytes、2207 行 SPIR-V、212 个 `OpAccessChain`。

结论：4/4 正确，但正效果不能复现，已撤销。

### 5.5 只读 storage buffer codebook

给 STQ MMVQ 增加单独的 32-byte `uint8_t` SSBO descriptor，把 codebook
作为只读 storage buffer 传入。

静态结果：

- 大 workgroup 48 registers。
- binary 453504 bytes。

交错结果：

| 顺序 | 实现 | TG128 |
| --- | --- | ---: |
| A7 | 原始实现 | 39.633 t/s |
| B | SSBO | 38.717 t/s |
| A8 | 原始实现 | 37.704 t/s |

结论：对两侧基线插值后基本中性，静态资源更差，接口改动较侵入。4/4 正确，
已完全撤销。

### 5.6 一次解码同时生成相邻两个 DP4A 输入

把原先分别生成 `v0` 和 `v1` 的两次函数调用合并，希望共享 code、sign 和
codebook 查询。

静态结果：

- subgroup SPIR-V 从 2136 行降到 2006 行。
- 16 行大 workgroup binary 从 655360 降到 652416 bytes。
- 寄存器仍为 71。

动态结果：

| 顺序 | 实现 | TG128 |
| --- | --- | ---: |
| B1 | 合并解码 | 56.428 t/s |
| A1 | 原始解码 | 56.715 t/s |
| B2 | 合并解码 | 54.745 t/s |
| A2 | 原始解码 | 54.713 t/s |

结论：中性到略慢。缩小 SPIR-V 没有转化为吞吐提升，说明 NVIDIA 驱动已经
消除或隐藏了大部分表面重复工作。已撤销。

### 5.7 仅外层固定 4 次循环强制展开

给生成 4 组 DP4A 输入的外层循环添加 `[[unroll]]`，内层 code 解码不强制展开。

静态结果：

- 大 workgroup 56 registers。
- binary 330240 bytes。

热态 A/B 为 55.409 对 55.395 t/s，完全中性。

结论：已撤销。

### 5.8 仅内层固定 4 次 code 解码强制展开

静态结果：

- 大 workgroup 56 registers。
- binary 329088 bytes。

有些遥测样本按 graphics clock 归一后快约 0.8% 到 1.0%，但原始吞吐会反转：

- 一组为 inner 55.635、baseline 52.006 t/s，期间时钟发生明显变化。
- 最后一组为 inner 56.04、baseline 56.59 t/s，baseline 反而更快。

结论：不能稳定复现，不足以保留，已撤销。

## 6. 明确负效果的方案

### 6.1 专用 128 线程单行 shader

目的：更直接对齐 CUDA。CUDA 的 `ncols=1` MMVQ 使用 4 个 warp 协作计算一行。

结果：满功率约 105 t/s，相对原始 Vulkan 约 134 t/s 明显变慢。

结论：Vulkan 当前按多个输出行复用输入向量更有价值，不能简单复制 CUDA 的
单行 workgroup 映射。

### 6.2 每个 workgroup 计算 4 行

结果约 108 t/s，明显慢于 8 行基线。

结论：输入向量复用不足，已撤销。

### 6.3 每个 workgroup 计算 12 或 16 行

静态资源：

| 行数 | 大 workgroup 寄存器 | Binary | Shared memory |
| ---: | ---: | ---: | ---: |
| 8 | 48 | 355712 bytes | 4224 bytes |
| 12 | 64 | 511104 bytes | 6336 bytes |
| 16 | 71 | 655360 bytes | 8448 bytes |

早期无遥测 8/16/8/16 序列：

```text
55.287, 56.215, 54.960, 55.190 t/s
```

这组数据看似支持 16 行约快 1.05%，但 35 W 遥测稳定段不支持：

| 行数 | TG128 | 平均 graphics clock | 平均功耗 |
| ---: | ---: | ---: | ---: |
| 8 | 53.883 t/s | 约 742 MHz | 约 34.3 W |
| 12 | 53.551 t/s | 约 766 MHz | 约 34.6 W |
| 16 | 50.921 t/s | 约 738 MHz | 约 34.6 W |

结论：当前 35 W 下 12 行没有优势，16 行明显更差。更高寄存器和共享内存使
功耗效率恶化。最终保留 8 行。16 行只值得在 80 W 环境复测约 1% 的潜在收益。

### 6.4 8 位元素 codebook

把 codebook 改成 8 位元素表。结果约 106 t/s，明显变慢。

结论：窄类型表没有带来更好的 NVIDIA 机器码，已撤销。

### 6.5 共享内存 codebook

把 codebook 放进 workgroup shared memory。结果约 107 t/s。

结论：初始化、同步和 shared 访问成本高于可能的表访问收益，已撤销。

### 6.6 每 lane 直接算术解码

不用动态 codebook，直接根据 code 和 sign 计算目标 ternary 值。

结果约 106 t/s，明显慢于查表。

结论：额外位运算和选择开销过大，已撤销。

### 6.7 Branchless full-qpack 算术公式

把完整 qpack 用无分支公式生成。公式已对全部 32 个 codebook 项做穷举验证，
结果正确。

10 个 TG 样本：

```text
114.751, 131.471, 124.849, 125.041, 124.948,
126.977, 125.497, 126.525, 128.890, 131.693
```

平均 126.064 t/s，低于约 134 t/s 基线。

结论：正确但较慢，已撤销。

### 6.8 16 项正号 codebook 加 sign flip

只保留 16 项正号表，用下式翻转全部非零 ternary lane：

```text
qpack_neg = qpack_pos ^ (((~qpack_pos) & 0x55) << 1)
```

静态结果：

- 40 registers。
- binary 493056 bytes。

动态结果：实验 38.523 t/s，紧邻原版 39.602 t/s。

结论：4/4 正确，但额外 sign flip 运算抵消了表减半收益，已撤销。

### 6.9 禁用手工 K-loop 展开

结果约 130 t/s，低于约 134 t/s 基线。

结论：更小的 SPIR-V 和机器码不代表更快；原实现的指令级并行更重要。

### 6.10 K-loop 只保留 2x 展开

静态结果：

- 48 registers。
- binary 199808 bytes。

动态结果：

- K2 B2 为 35.198 t/s。
- 两侧原版 A9/A10 为 38.055、38.127 t/s。
- K2 B3 为 36.270 t/s。

结论：稳定慢约 5% 到 8%，已撤销。

### 6.11 Per-qpack subgroup shuffle

让 lane 间共享单个 qpack 的解码结果。结果约 99 t/s。

结论：subgroup shuffle 和重排成本远高于节省的读取，已撤销。

### 6.12 四 qpack 打包 subgroup shuffle

把四个 qpack 一起打包后通过 subgroup 共享。结果约 127 t/s。

结论：比逐 qpack shuffle 好，但仍慢于约 134 t/s 基线，已撤销。

### 6.13 Local paired decode without subgroup operations

不用 subgroup 操作，在本地成对生成或复用解码结果。

低功率实验为 36.94 +/- 0.72 t/s，紧邻原版为 38.78 +/- 0.41 t/s，约慢
4.75%。

结论：已撤销。该实验与第 5.6 节后续测试的“一次生成相邻两个 DP4A 输入”
不是完全相同的代码版本，但两者都没有带来稳定收益。

### 6.14 内外两个 STQ 固定循环都强制展开

这是最直接模仿 CUDA 两处 `#pragma unroll` 的版本。

静态结果：

- subgroup SPIR-V 增至 76660 bytes、4149 行。
- 大 workgroup 64 registers。
- binary 330112 bytes。

动态结果：

- 冷状态曾出现 61.220 对原版 56.932 t/s，但主要来自更高 GPU 时钟。
- 升温后的展开版为 55.277 t/s，原版为 56.760 t/s。

结论：更高寄存器和功耗抵消展开收益，热态明确较慢，已撤销。

## 7. CUDA 与 Vulkan 实现对比结论

内积算法已经基本对齐：

- `VDR_STQ1_0_Q8_1_MMVQ` 都为 1。
- 都每次处理 4 个 group。
- 都为相邻位置构造两个 packed int8 向量。
- 都执行 8 次 DP4A。
- 都使用相同 32 项 codebook 和 scale 乘法。

workgroup 映射并不相同：

- CUDA `ncols=1` 通用参数使用 4 个 warp，也就是 128 线程协作计算 1 行。
- Vulkan 当前大 workgroup 也是 128 线程，但每个 workgroup 计算 8 行，以复用
  输入向量数据。
- Vulkan 专用 128 线程单行实验明显变慢，所以不能只按 CUDA launch 参数照搬。

CUDA `cuobjdump` 资源：

| CUDA kernel | Registers/thread | Shared memory |
| --- | ---: | ---: |
| STQ1_0，`ggml_type=42` | 80 | 384 bytes |
| Q4_K，`ggml_type=12` | 39 | 384 bytes |

最终 Vulkan 大 workgroup 报告 48 registers/thread，subgroup 版本报告 80。
Vulkan executable properties 中的 `Local Memory Size=68719476736` 明显不是
可用的 spill 指标，不能据此判断 local-memory spill。

当前最明显的编译器差异仍是动态 GLSL codebook 的表示方式。原始 SPIR-V 中
能看到多份函数局部 `uint[32]` 表，但缩小或搬移这些表的多个实验都没有稳定
提高最终吞吐。下一步应使用 Nsight Graphics 等工具查看实际指令、cache miss、
issue stall 和 achieved occupancy，而不是继续只根据 SPIR-V 行数猜测。

## 8. 尚未测试或值得以后测试的方向

### 8.1 满功率重新验证

最优先事项：

1. 接回高功率电源，确认功率限制恢复到 80 W。
2. 先跑 CUDA STQ 和 Q4 基线。
3. 使用 original/packed16/original 或 packed16/original/packed16 快速交错。
4. 重测 TG128 和 PP2048。
5. 再决定 packed16 的约 4% 收益在满功率下是否保持。

### 8.2 Release 构建

当前使用 RelWithDebInfo，benchmark 报告 asserts enabled。尚未在纯 Release
构建中重复最终 STQ 基准。

### 8.3 指令级 profiling

尚未完成：

- Vulkan 最终机器码的完整反汇编。
- 每条 Vulkan STQ dispatch 的 instruction issue、stall reason、L1/L2 命中率。
- 实际 achieved occupancy。
- codebook 查询是否产生真正的 local-memory spill。

### 8.4 p0 编译期特化

当前 Vulkan 路径中 `K_PER_ITER=32`，传入 STQ 内积的 `iqs` 为 0，`p0` 根据
`ib_a` 奇偶为 0 或 2。相邻 lane 会交替奇偶，所以运行时直接按 `p0` 分支会产生
最差形式的 subgroup divergence。

仍可研究在 dispatch 或 shader 生成阶段构造两个无分支专用 pipeline，但需要
同时解决 lane 映射和额外 dispatch 成本，尚未测试。

### 8.5 一次性 GPU 权重重排

可以考虑在模型上传后建立 Vulkan 专用持久缓冲，把每个 5-bit code/sign 索引
扩展成已解码 qpack byte，以彻底去掉动态 codebook 查询。

潜在问题：

- STQ 权重部分会明显增大。
- 需要 backend 特定的额外持久 buffer。
- 要修改 tensor buffer 生命周期和内存规划。
- 实现侵入性远高于当前 view-only packed16。

只有在 profiling 明确证明 codebook 是主要瓶颈时才值得实施。

## 9. 正确性、构建和测试命令

### 正确性

```powershell
$env:GGML_VK_FORCE_MMVQ = '1'
$env:GGML_VK_PIPELINE_STATS = 'mul_mat_vec_stq1_0_q8_1_f32'

& '.\build-vulkan-reldbg\bin\test-backend-ops.exe' test `
  -b Vulkan1 -o MUL_MAT -p 'stq1_0'
```

预期 4/4：

- `m=256, n=1, k=256`
- `m=256, n=4, k=256`
- `m=256, n=16, k=256`
- `m=512, n=1, k=1024`

### 构建

```powershell
cmake --build build-vulkan-reldbg --config RelWithDebInfo `
  --target llama-bench test-backend-ops -j 12
```

受限 agent sandbox 中 Ninja 启动子进程可能卡在 `Re-checking globbed directories`，
需要在允许子进程的 shell 中构建。普通用户终端可以直接使用 Ninja。

### Vulkan TG128

```powershell
& '.\build-vulkan-reldbg\bin\llama-bench.exe' `
  -m 'D:\tools\llm\Hy-MT2-1.8B-1.25Bit.gguf' `
  -dev Vulkan1 -p 0 -n 128 -b 2048 -ub 512 `
  -fa on -ngl 99 -r 10 -o json
```

### CUDA TG128

```powershell
& 'D:\tools\llm\llama-hunyuan\llama-bench.exe' `
  -m 'D:\tools\llm\Hy-MT2-1.8B-1.25Bit.gguf' `
  -dev CUDA0 -p 0 -n 128 -b 2048 -ub 512 `
  -fa on -ngl 99 -r 10 -o json
```

## 10. 最终建议

- 当前继续使用已提交的 packed16 + 8 行实现。
- 不要重复已经明确变慢的 codebook、subgroup shuffle、算术解码和 K2 实验。
- 不要只根据 SPIR-V 或 binary 大小判断性能。
- 在 35 W 环境只接受快速交错和遥测支持的结论。
- 接回 80 W 电源后，第一件事应是重新测 CUDA/Vulkan TG 与 PP，而不是继续改代码。
- 下一轮真正有价值的技术工作是指令级 profiling，或在 profiling 支持下研究持久
  权重重排。
