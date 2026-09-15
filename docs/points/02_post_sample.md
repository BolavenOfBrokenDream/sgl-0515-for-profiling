# 02. 异步 exponential-race 采样与小算子链优化（post_sample）

- 实验序：2（0517 `34064f2` → `e667f3f`，交付态 v3.2）
- 开关：
  - `SGLANG_NPU_POST_SAMPLE_OPT`（本仓补丁 `0002` 新增，默认 1）——常驻部分：全 1.0 温度跳过 RealDiv、softmax 不写回、logprob clamp 下沉 + gather
  - `SGLANG_NPU_ASYNC_EXPONENTIAL`（默认 0）——异步采样总开关，仅 910C
  - `SGLANG_NPU_EXP_RACE_TRITON`（默认 1）——Triton 融合消费 kernel，仅在异步采样实际启用时生效
- 适用：9B + 35B

## 改动摘要

噪声生成提前到 forward 前、放独立 NPU stream（DSA 引擎，不争 AIV/AIC）；消费链 7 次全矩阵遍历融合为两段式 partials Triton kernel；全 1.0 温度跳过全词表 RealDiv；softmax 不写回省 TensorMove；logprob clamp 下沉到提取后小张量、sampled-token 改 gather。配套修复 fla/cumsum autotune key 7→6（启动崩溃修复，不门控）。

## A/B 方法

建议分两步（同属实验序 2）：

1. 常驻部分：A `SGLANG_NPU_POST_SAMPLE_OPT=0` vs B `=1`（异步采样保持关）。
2. 异步采样：A 仅常驻开 vs B 再开 `SGLANG_NPU_ASYNC_EXPONENTIAL=1`（`SGLANG_NPU_EXP_RACE_TRITON` 保持 1）。
   - 注意：该替换保持采样分布不变但**改变随机序列**，对比输出文本会有差异，属预期。

生效条件：decode、非投机/verify、无 grammar、非全 greedy、无 per-request seed、无 top-k/p/min-p；非 910C 回退。

## 已知收益（0517 环境）

8die / bs32 / tp8 per engine：0.8ms per decode step。

## 结果

### Qwen3.5-9B

| 臂 | decode ms/step | tok/s | 备注 |
|---|---|---|---|
| A0（全关） | | | |
| B1（常驻） | | | |
| B2（常驻+异步） | | | |

profiling 归档：`results/qwen3.5-9b/profiling/02_post_sample/{a,b1,b2}/`

### Qwen3.5-35B-A3B

| 臂 | decode ms/step | tok/s | 备注 |
|---|---|---|---|
| A0（全关） | | | |
| B1（常驻） | | | |
| B2（常驻+异步） | | | |

profiling 归档：`results/qwen3.5-35b-a3b/profiling/02_post_sample/{a,b1,b2}/`

## 备注

无对其他点的依赖。采样窗口收益随 batch/vocab 增大而增大（bs128/vocab248k 场景 stock multinomial 可达 3ms/step）。
