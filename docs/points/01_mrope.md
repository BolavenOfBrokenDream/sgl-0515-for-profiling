# 01. mRoPE device positions 快路径（mrope）

- 实验序：1（0517 `a9c7bc4`）
- 开关：`SGLANG_NPU_MROPE_FASTPATH`（本仓补丁 `0001` 新增，默认 1；基线为总是开启无开关）
- 适用：9B + 35B（纯文本 batch；含真实多模态自动回退）

## 改动摘要

`forward_batch_info.py::_compute_mrope_positions`：纯文本场景三行 mRoPE 坐标与 device 上 `self.positions` 逐元素相等，快路径直接 `positions.to(int64).unsqueeze(0).repeat(3,1)`，消除每 step 逐 request host 构造 + `torch.cat` + 阻塞 H2D 拷贝。

## A/B 方法

- A：`SGLANG_NPU_MROPE_FASTPATH=0`（= 全关基线）；B：`=1`。
- 计量：decode step CPU 侧耗时 + 端到端。**注意**：消除阻塞 H2D 后设备等待会迁移到下游同步点（如 replay 中的 `.cpu()`），单函数变快不等于端到端变快，以完整 decode step 为准。

## 已知收益（0517 环境）

CPU 侧 1.5ms → 0.1ms per decode step（端到端待本仓实测）。

## 结果

### Qwen3.5-9B（TP_ = _，bs = _）

| 臂 | decode ms/step | tok/s | 备注 |
|---|---|---|---|
| A（关） | | | |
| B（开） | | | |

profiling 归档：`results/qwen3.5-9b/profiling/01_mrope/{a,b}/`

### Qwen3.5-35B-A3B（TP_ = _，bs = _）

| 臂 | decode ms/step | tok/s | 备注 |
|---|---|---|---|
| A（关） | | | |
| B（开） | | | |

profiling 归档：`results/qwen3.5-35b-a3b/profiling/01_mrope/{a,b}/`

## 备注

无开关交互。生效条件：非 MTP、非扩散 LM、`rl_on_policy_target` 已设置或全 batch 无多模态输入。
