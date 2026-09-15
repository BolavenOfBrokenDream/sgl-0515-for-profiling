# 实验计划：逐优化点 A/B 与 profiling

## 实验顺序（0517 开发仓上仓顺序）

以 `sglang-npu-opt-stack` 提交历史为准（基线导入与 PR 合入不计）：

| 实验序 | 点 | 0517 commit | 备注 |
|---|---|---|---|
| 1 | mrope | `a9c7bc4` | |
| 2 | post_sample | `34064f2` → `e667f3f`（v3.2 升级） | 交付态为 v3.2 终态 |
| 3 | gmm | `a7a1c0d` | 仅 35B |
| 4 | gdn_mamba | `fe997a5` | |
| 5 | moe_front_fusion | `f91b9b5` | 仅 35B |
| 6 | full_attention | `95461e4` → `d62cb3b`（守卫放宽 q∈{1,2,4}） | 交付态含升级版 |
| 7 | ~~moe_weight_prefetch~~ | `4501149` | **不做，恒关**（最优实践裁决：tp8+8die 下 w13/w2 难以全量预取且抢带宽） |
| 8 | gdn_recurrent_decode_opt | `c2a4e31` | 与 #12 同一调用点，分别实验 |
| 9 | cast_elimination | `6972d2d` | |
| 10 | gdn_qkvzba_pack | `1e04211`（tp v4 拆分） | |
| 11 | fused_qkvzba_conv1d | `1e04211` | |
| 12 | recurrent_ascendc | `1e04211` | |
| 13 | vgmm1 | `1e04211` | 0515 侧另经 `2eb3bc8` 移植，未上机验证过 |
| 14 | moe_tail_fusion | `1e04211` | 0515 侧 `2b0034e` 重实现 + `2eb3bc8` v4.1，未上机验证过 |

## A/B 协议

每个优化点做一组 A/B：

- **A（对照）**：截止该点之前的累计开启状态（实验序 < 当前点的点全开，prefetch 恒关）。
- **B（实验）**：A 的基础上再开当前点。

即第 1 点的 A 是"全关基线"；此后逐点累积。两臂除当前点开关外其余 env、模型、TP、batch、数据集、随机种子完全一致。

执行要点：

1. 开关在 capture 期固化——**每臂一次完整服务重启**，改 env 不重启无效。
2. 每臂先 warmup（≥ 2× capture batch 覆盖的请求量），再正式计量。
3. 计量口径：decode step 耗时（ms/step）与端到端吞吐（tok/s）。**必须用图模式口径**——eager 计时大胜是 launch 消除假象（历史教训）。
4. 每臂各采一份 profiling（见下），命名 `<point>/{a,b}` 归档。
5. 9B dense 无 MoE：实验序 3、5、13、14 跳过（7 本就不做）；mrope 快路径对纯文本 batch 生效，两模型均测。
6. 组合交互点按 `docs/switch-matrix.md` 末尾清单处理：#3 的完整收益测法要求 #5 已开（累积协议天然满足）；#13 只在与 #5 组合下验 C1；#14 需双流环境。

## Profiling 协议

- 采集窗口：稳态 decode 段（避开 prefill 与启动期），每臂固定相同请求数/时长。
- 工具：torch_npu profiler（`torch_npu.profiler.profile`，`experimental_config` 开 `aic_metrics` + `l2_cache`）或 msprof，与既有 probe 实验口径保持一致。
- 每臂产物：kernel 耗时表（按 name 聚合）、decode step 时间轴、目标算子前后窗口对比。
- 归档：`results/<model>/profiling/<NN>_<point>/{a,b}/`，人工从服务器回拷；汇总读数填入 `docs/points/NN_<point>.md` 的结果表。
- 判定收益时警惕两类假象：热态单测读数（vgmm1 教训：生产冷态 memory bound 下标量开销被 DMA 掩盖）与阻塞消除后等待迁移到下游同步点（mrope 教训：单函数变快 ≠ 端到端等量变快，收益以完整 decode step 为准）。

## 结果记录

- 每点一篇笔记：`docs/points/NN_<name>.md`，含开关、A/B 方法、9B/35B 结果表、profiling 摘要与归档链接。
- 原始数据 csv 放 `results/<model>/`，格式：每测试块一行标题行 + csv 数据，块间空行。
