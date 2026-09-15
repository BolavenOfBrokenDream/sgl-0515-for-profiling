# 09. decode 图内边角 cast 消除（cast_elimination）

- 实验序：9（0517 `6972d2d`）
- 开关：`SGLANG_NPU_CAST_ELIMINATION`（本仓补丁 `0004` 新增，默认 1；基线为合入即生效无开关）
- 适用：9B + 35B（GDN 段）；MoE gating 部分仅 35B 且以 #5 开为前提

## 改动摘要

1. GDN conv 前：decode 图元数据 4 个静态 buffer 上提 int64，conv host 侧 `.to(kLong)` 变 no-op，每层省 2 个 cast（`state_indices_list_gdn` 保持 int32——`recurrent_gated_delta_rule` 按 `int32_t*` 裸读，改 dtype 会静默算错）。
2. GDN recurrent 侧：同一组 buffer 喂 conv（int64）与 recurrent（int32）两契约，解法为 int32 影子 buffer 双份——图外每 step 1–2 个小 copy 同步，图内零 cast（把每层 2 个图内 cast 换成每 step 2 个图外小 copy）。
3. MoE gating bf16 直喂：`npu_moe_gating_top_k` 契约支持 BF16（内部 CAST_NONE 加宽、全程 fp32 计算、CAST_RINT 输出，与 host 侧 fp32 化逐位一致），删出入口两处 fp32 化；deepep 线两处 `dispatch_a` 补防御性 `.to(fp32)` 恢复契约。
4. `dt_bias` 源头 fp32 物化：AscendC recurrent wrapper 的 `.float()` 变 no-op（对 #12 生效）。

## A/B 方法

- A：`SGLANG_NPU_CAST_ELIMINATION=0`；B：`=1`。
- GDN 段在 decode 图流程生效，eager/verify 影子恒 None、行为与基线一致；MoE 部分需 #5 已开（累积协议下满足，仅 35B）；dt_bias 部分收益在 #12 开启后才兑现（#12 关时本包该改动零副作用）。
- profiling 对比重点：decode 图内 cast kernel 计数（GDN 每层 2→0，recurrent 侧 3→0，MoE 前段 2→0）。

## 已知收益

（待补充——单个索引/小 tensor cast 约 2–4µs，逐层重复）

## 结果

### Qwen3.5-9B

| 臂 | decode ms/step | 图内 cast 数/层 | 备注 |
|---|---|---|---|
| A（关） | | | |
| B（开） | | | |

profiling 归档：`results/qwen3.5-9b/profiling/09_cast_elimination/{a,b}/`

### Qwen3.5-35B-A3B

| 臂 | decode ms/step | 图内 cast 数/层 | 备注 |
|---|---|---|---|
| A（关） | | | |
| B（开） | | | |

profiling 归档：`results/qwen3.5-35b-a3b/profiling/09_cast_elimination/{a,b}/`

## 备注

纯调度开销消除，无数值变化。影子 buffer 与主 buffer 同源镜像维护，eager/verify 显式置 None 兜底。
