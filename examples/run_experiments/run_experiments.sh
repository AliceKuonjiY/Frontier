#!/usr/bin/env bash
# Qwen3-4B PD-AF 一键实验入口。
#
# 用法：
#   ./run_experiments.sh                         # core 实验集
#   ./run_experiments.sh --profile full          # 全量（含 AF2/AF4 对照）
#   ./run_experiments.sh --stage gpu_scale_burst # 只跑配比
#   ./run_experiments.sh --list                  # 只看配置
#   ./run_experiments.sh --dry-run               # 只打印命令
#
# 环境变量：
#   PYTHON_BIN   指定 Python 解释器
#   OMP_NUM_THREADS / OPENBLAS_NUM_THREADS  默认 2
#   PROFILE      默认 profile（等价于 --profile）

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# examples/run_experiments -> examples -> Frontier
FRONTIER_ROOT="$(cd "$HERE/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$FRONTIER_ROOT/.." && pwd)"

if [ ! -d "$FRONTIER_ROOT/frontier" ]; then
  echo "ERROR: 找不到 Frontier 工作树：$FRONTIER_ROOT" >&2
  exit 2
fi

find_python() {
  local candidates=()
  [ -n "${PYTHON_BIN:-}" ] && candidates+=("$PYTHON_BIN")
  candidates+=(
    "$WORKSPACE_ROOT/.venv/bin/python"
    "$FRONTIER_ROOT/.venv/bin/python"
  )
  if command -v python3 >/dev/null 2>&1; then
    candidates+=("$(command -v python3)")
  fi

  local py
  for py in "${candidates[@]}"; do
    [ -n "$py" ] || continue
    [ -x "$py" ] || continue
    if "$py" -c 'import sklearn, pandas, numpy, scipy' >/dev/null 2>&1; then
      echo "$py"
      return 0
    fi
  done
  return 1
}

PY="$(find_python || true)"
if [ -z "$PY" ]; then
  cat >&2 <<'EOF'
ERROR: 没有找到带依赖的 Python。

请先创建 uv 虚拟环境：
  cd Frontier
  uv venv .venv
  uv pip install --python .venv/bin/python -e .

或指定已有解释器：
  PYTHON_BIN=/path/to/python ./run_experiments.sh
EOF
  exit 2
fi

export PYTHONPATH="$FRONTIER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DISABLED=true
export VIDUR_DISABLE_WANDB=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

EXTRA_ARGS=()
if [ -n "${PROFILE:-}" ]; then
  EXTRA_ARGS+=(--profile "$PROFILE")
fi

echo "Python   : $PY"
echo "Frontier : $FRONTIER_ROOT"
echo

exec "$PY" "$HERE/run_pdaf_experiments.py" "${EXTRA_ARGS[@]}" "$@"