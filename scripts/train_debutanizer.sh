#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
用法：
  bash scripts/train_debutanizer.sh CSV路径 [训练参数...]

示例：
  bash scripts/train_debutanizer.sh data/debutanizer.csv
  bash scripts/train_debutanizer.sh data/debutanizer.csv --epochs 1 --batch-size 8
  bash scripts/train_debutanizer.sh data/debutanizer.csv --mode finetune --gradient-checkpointing

默认使用 MOMENT-Small、process 特征、probe 模式及自动设备选择。
输出默认保存到仓库 outputs/debutanizer/ 下的新目录，可用 --output 指定。
训练结束后自动加载 best.pt 复核测试集推理，保存指标、预测结果与图表。
相对数据路径和自定义输出路径均相对于当前工作目录。

请先安装依赖（Linux 选择一种）：
  uv sync --locked --extra cu128
  uv sync --locked --extra cpu

完整训练参数：
  uv run --no-sync python scripts/debutanizer.py train --help
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi

DATA_PATH="$1"
shift
if [[ ! -f "$DATA_PATH" ]]; then
  printf '数据文件不存在：%s\n' "$DATA_PATH" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  printf '未找到 uv，请安装 uv 并按 --help 中的说明安装依赖。\n' >&2
  exit 1
fi

OUTPUT_DIR="$ROOT_DIR/outputs/debutanizer/run_$(date +%Y%m%d_%H%M%S)_$$"
exec uv run --project "$ROOT_DIR" --no-sync python "$ROOT_DIR/scripts/debutanizer.py" train \
  --data "$DATA_PATH" \
  --output "$OUTPUT_DIR" \
  "$@"
