#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 relax e2e rollout 日志提取三组性能指标，产出多组实验对比 csv。

指标定义：
1. rollout_time_avg_s
   每 rollout step 打印一次的完成行：
     Rollout 0 generation: 100%|██████████| 64/64 [03:18<00:00,  3.11s/it]
   取其中的耗时（03:18 -> 198s）。step 0 偏长不计，取 step 1..last 的平均；
   可用 --trim-rollout-top N 再剔除耗时最大的 N 个 step。

2. max_gen_throughput_tok_s
   仅 Decode batch 行的 "gen throughput (token/s)"（Prefill 不计）。按引擎区分
   （行首 SGLangEngine pid=xxx，2 engine 实验即 2 个 pid），每个引擎取全日志最大
   值（出现位置不定，引擎预热期第一次打印可能很慢，须全量比对），再跨引擎求和。

3. total_gen_throughput_tok_s
   每个 rollout step（>=1）内：各引擎 "Decode batch" 行的 #full token 最大值跨引擎
   求和（先每引擎取 max 再相加），除以该 step 的 rollout time（s）；取各 step 平均。
   step 归属：Rollout N 完成行打印于 step N 末尾，其后的 decode 行属于 step N+1。
   --trim-rollout-top N 剔除耗时最大的 N 个 step 同样作用于本指标与指标 1。

实验顺序：按文件名中的时间戳（非 mtime）升序，最早 = 00_baseline，依次对应
14 组实验（00_baseline .. 13_moe_tail_fusion）。

用法：
  extract_rollout_metrics.py <log_dir> [-o OUT_CSV] [--glob GLOB] [--per-step]

输出 csv（exp 规约：一行标题行 + csv 数据）。
"""

import argparse
import re
import sys
from pathlib import Path

EXPERIMENT_NAMES = [
    "00_baseline",
    "01_mrope",
    "02_post_sample",
    "03_gmm",
    "04_gdn_mamba",
    "05_moe_front_fusion",
    "06_full_attention",
    "07_gdn_recurrent_decode_opt",
    "08_cast_elimination",
    "09_gdn_qkvzba_pack",
    "10_fused_qkvzba_conv1d",
    "11_recurrent_ascendc",
    "12_vgmm1",
    "13_moe_tail_fusion",
]

# 文件名时间戳，形如 2026-09-16-01:16:44（tee 日志名）。回拷到 Windows 时冒号
# 可能被改写为 -、_、：或私有区字符 \uf03a，一并兼容
RE_TS = re.compile(r"(\d{4})-(\d{2})-(\d{2})[-_ ](\d{2})[:：_\-](\d{2})[:：_\-](\d{2})")
# rollout step 完成行（要求带速率的完整 tqdm 后缀，排除 `?it/s` 的半途更新）
RE_ROLLOUT = re.compile(
    r"Rollout\s+(\d+)\s+generation:.*\[((?:\d+:)?\d{1,2}:\d{2})<[^]\[]*\d+(?:\.\d+)?s/it\]"
)
RE_PID = re.compile(r"\(SGLangEngine pid=(\d+)\)")
RE_DECODE = re.compile(
    r"Decode batch.*?#full token:\s*(\d+).*?gen throughput \(token/s\):\s*([\d.]+)"
)


def parse_elapsed(s: str) -> float:
    parts = [int(p) for p in s.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def parse_log(path: Path):
    """返回 (step_times dict, max_tps_per_pid dict, step_full_token dict, pids set)"""
    step_times = {}          # step -> elapsed s（last wins，取完成行）
    max_tps = {}             # pid -> max gen throughput（全日志，不分 step）
    step_ft = {}             # step -> {pid -> max #full token}
    pids = set()
    current_step = 0         # 首个 Rollout 0 完成行之前的 decode 行归 step 0

    with open(path, "r", errors="replace") as f:
        for line in f:
            # tqdm 用 \r 刷新，按段拆开处理
            for seg in line.split("\r"):
                m = RE_ROLLOUT.search(seg)
                if m:
                    n = int(m.group(1))
                    step_times[n] = parse_elapsed(m.group(2))
                    current_step = n + 1
                    continue
                if "Decode batch" not in seg:
                    continue
                d = RE_DECODE.search(seg)
                if not d:
                    continue
                full_token, tps = int(d.group(1)), float(d.group(2))
                p = RE_PID.search(seg)
                pid = p.group(1) if p else "unknown"
                pids.add(pid)
                if tps > max_tps.get(pid, 0.0):
                    max_tps[pid] = tps
                if current_step >= 1:
                    slot = step_ft.setdefault(current_step, {})
                    if full_token > slot.get(pid, 0):
                        slot[pid] = full_token
    return step_times, max_tps, step_ft, pids


def metrics_for(path: Path, trim_rollout_top: int = 0):
    step_times, max_tps, step_ft, pids = parse_log(path)

    steps = sorted(s for s in step_times if s >= 1)
    # 剔除耗时最大的 trim_rollout_top 个 step，指标 1 与指标 3 都只用剩余 step
    steps_used = sorted(steps, key=lambda s: step_times[s])[: len(steps) - trim_rollout_top]
    rollout_avg = None
    if steps_used:
        rollout_avg = sum(step_times[s] for s in steps_used) / len(steps_used)
    elif steps:
        print(f"WARN: {path.name} trim_rollout_top={trim_rollout_top} >= 可用 step 数 "
              f"{len(steps)}，rollout_time_avg_s / total_gen_throughput_tok_s 置空",
              file=sys.stderr)

    max_gen = sum(max_tps.values()) if max_tps else None

    totals = []
    for s in steps_used:
        if s in step_ft and s in step_times:
            totals.append(sum(step_ft[s].values()) / step_times[s])
    total_gen = sum(totals) / len(totals) if totals else None

    per_step = [
        (s, step_times[s], sum(step_ft.get(s, {}).values()),
         sum(step_ft[s].values()) / step_times[s] if s in step_ft else None)
        for s in steps
    ]
    return rollout_avg, max_gen, total_gen, len(steps_used), len(pids), per_step


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_dir", help="存放 14 个实验日志的目录")
    ap.add_argument("-o", "--output", help="输出 csv 路径（默认 <log_dir>/rollout_metrics.csv）")
    ap.add_argument("--glob", default="*.log", help="日志文件 glob（默认 *.log）")
    ap.add_argument("--per-step", action="store_true",
                    help="附带输出每 step 明细 csv（<out>.per_step.csv）")
    ap.add_argument("--trim-rollout-top", type=int, default=0, metavar="N",
                    help="剔除耗时最大的 N 个 step 后再计算 rollout_time_avg_s 与 "
                         "total_gen_throughput_tok_s（默认 0；max_gen_throughput 不受影响）")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    files = sorted(p for p in log_dir.glob(args.glob) if p.is_file())
    if not files:
        sys.exit(f"no log files matched {args.glob} in {log_dir}")

    def ts_key(p: Path):
        m = RE_TS.search(p.name)
        if not m:
            print(f"WARN: {p.name} 文件名无时间戳，排到最后", file=sys.stderr)
            return (9999, 99, 99, 99, 99, 99, p.name)
        return tuple(int(g) for g in m.groups()) + (p.name,)

    files.sort(key=ts_key)
    if len(files) != len(EXPERIMENT_NAMES):
        print(f"WARN: 日志数 {len(files)} != {len(EXPERIMENT_NAMES)}，仍按时间戳顺序编号",
              file=sys.stderr)

    out = Path(args.output) if args.output else log_dir / "rollout_metrics.csv"
    header = ["experiment", "log_file", "n_engines", "n_steps_used",
              "rollout_time_avg_s", "max_gen_throughput_tok_s",
              "total_gen_throughput_tok_s"]
    rows, per_step_rows = [], []

    for i, p in enumerate(files):
        name = EXPERIMENT_NAMES[i] if i < len(EXPERIMENT_NAMES) else f"exp_{i:02d}"
        rollout_avg, max_gen, total_gen, n_steps, n_eng, per_step = metrics_for(
            p, trim_rollout_top=args.trim_rollout_top)
        if rollout_avg is None:
            print(f"WARN: {p.name} 未解析到可用 rollout step 完成行", file=sys.stderr)
        if max_gen is None:
            print(f"WARN: {p.name} 未解析到 Decode batch 行", file=sys.stderr)
        fmt = lambda v: "" if v is None else f"{v:.2f}"
        rows.append([name, p.name, n_eng, n_steps,
                     fmt(rollout_avg), fmt(max_gen), fmt(total_gen)])
        for s, t, ft_sum, tg in per_step:
            per_step_rows.append([name, s, f"{t:.0f}", ft_sum,
                                  "" if tg is None else f"{tg:.2f}"])

    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write(f"# rollout metrics comparison, dir={log_dir}, ordered by filename timestamp, "
                f"trim_rollout_top={args.trim_rollout_top}\n")
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(out)

    if args.per_step:
        ps_out = out.with_suffix(".per_step.csv")
        with open(ps_out, "w", encoding="utf-8", newline="") as f:
            f.write("# per-step detail\n")
            f.write("experiment,step,rollout_time_s,full_token_sum_all_engines,total_gen_throughput_tok_s\n")
            for r in per_step_rows:
                f.write(",".join(str(x) for x in r) + "\n")
        print(ps_out)


if __name__ == "__main__":
    main()
