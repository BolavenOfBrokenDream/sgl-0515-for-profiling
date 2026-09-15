#!/bin/bash
# decode 压测：向已启动的 sglang 服务发压测流量，结果按 exp 规约写 csv。
# 用法：scripts/bench_decode.sh <model_tag> <NN_point> <arm>
# 依赖环境变量（按部署修改，或用环境覆盖）：
#   SGL_HOST（默认 127.0.0.1）SGL_PORT（默认 30000）
#   BENCH_NUM_PROMPTS（默认 200）BENCH_INPUT_LEN（默认 1024）BENCH_OUTPUT_LEN（默认 1024）
#   BENCH_MAX_CONCURRENCY（默认 32，对齐生产 decode bs）
# 输出：results/<model_tag>/<NN_point>_<arm>.csv
#   格式：一行标题行 + csv 数据（块间空行），直接可读/可粘贴。

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_TAG="${1:?usage: bench_decode.sh <model_tag> <NN_point> <arm>}"
POINT="${2:?}"
ARM="${3:?}"

HOST="${SGL_HOST:-127.0.0.1}"
PORT="${SGL_PORT:-30000}"
NUM_PROMPTS="${BENCH_NUM_PROMPTS:-200}"
INPUT_LEN="${BENCH_INPUT_LEN:-1024}"
OUTPUT_LEN="${BENCH_OUTPUT_LEN:-1024}"
CONCURRENCY="${BENCH_MAX_CONCURRENCY:-32}"

OUT_DIR="$REPO_DIR/results/$MODEL_TAG"
mkdir -p "$OUT_DIR"
RAW_JSON="$OUT_DIR/${POINT}_${ARM}_bench.json"
CSV="$OUT_DIR/${POINT}_${ARM}.csv"

python3 -m sglang.bench_serving \
  --backend sglang \
  --host "$HOST" --port "$PORT" \
  --dataset-name random \
  --random-input-len "$INPUT_LEN" \
  --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$NUM_PROMPTS" \
  --max-concurrency "$CONCURRENCY" \
  --output-file "$RAW_JSON"

# 转 csv：标题行 + 关键指标（decode 口径为主）
python3 - "$RAW_JSON" "$CSV" "$POINT" "$ARM" "$MODEL_TAG" <<'PY'
import json, sys

raw, csv_path, point, arm, model = sys.argv[1:6]
with open(raw) as f:
    d = json.load(f)

keys = [
    "model", "point", "arm",
    "completed", "total_throughput_tok_s", "output_throughput_tok_s",
    "mean_ttft_ms", "mean_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "p99_itl_ms", "mean_e2e_latency_ms",
]
vals = {
    "model": model, "point": point, "arm": arm,
    "completed": d.get("completed"),
    "total_throughput_tok_s": d.get("total_throughput"),
    "output_throughput_tok_s": d.get("output_throughput"),
    "mean_ttft_ms": d.get("mean_ttft_ms"),
    "mean_tpot_ms": d.get("mean_tpot_ms"),
    "p99_tpot_ms": d.get("p99_tpot_ms"),
    "mean_itl_ms": d.get("mean_itl_ms"),
    "p99_itl_ms": d.get("p99_itl_ms"),
    "mean_e2e_latency_ms": d.get("mean_e2e_latency_ms"),
}
with open(csv_path, "w") as f:
    f.write(f"# {model} {point} arm={arm} decode bench\n")
    f.write(",".join(keys) + "\n")
    f.write(",".join("" if vals[k] is None else str(vals[k]) for k in keys) + "\n")
print(csv_path)
PY
