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
#   PYTHON_BIN   显式指定 Python 解释器或 PATH 中的命令名
#   OMP_NUM_THREADS / OPENBLAS_NUM_THREADS  默认 2
#   PROFILE      默认 profile（等价于 --profile）

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# examples/run_experiments -> examples -> Frontier
FRONTIER_ROOT="$(cd "$HERE/../.." && pwd)"

if [ ! -d "$FRONTIER_ROOT/frontier" ]; then
  echo "ERROR: 找不到 Frontier 工作树：$FRONTIER_ROOT" >&2
  exit 2
fi

check_python() {
  local out
  out="$("$1" -c 'import sys, ddsketch, fasteners, numpy, pandas, plotly, yaml, sklearn, scipy, tqdm; assert sys.version_info >= (3, 10); print("ok")' 2>/dev/null)" || return 1
  [ "$out" = "ok" ]
}

if [ -n "${PYTHON_BIN:-}" ]; then
  # An explicit PYTHON_BIN is authoritative: resolve PATH names, validate it,
  # and fail loudly instead of silently falling back to another interpreter.
  if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PY="$(command -v "$PYTHON_BIN")"
  elif [ -x "$PYTHON_BIN" ]; then
    PY="$PYTHON_BIN"
  else
    echo "ERROR: PYTHON_BIN 不是可执行命令或文件：$PYTHON_BIN" >&2
    exit 2
  fi
  if ! check_python "$PY"; then
    echo "ERROR: PYTHON_BIN=$PYTHON_BIN 缺少依赖（ddsketch/fasteners/numpy/pandas/plotly/yaml/sklearn/scipy/tqdm）" >&2
    exit 2
  fi
else
  # Only the repository's own venv and the system python3 are auto-probed.
  # Do not execute a .venv from an external parent directory.
  PY=""
  for candidate in "$FRONTIER_ROOT/.venv/bin/python" "$(command -v python3 || true)"; do
    [ -n "$candidate" ] || continue
    [ -x "$candidate" ] || continue
    if check_python "$candidate"; then
      PY="$candidate"
      break
    fi
  done
  if [ -z "$PY" ]; then
    cat >&2 <<'EOF'
ERROR: 没有找到带依赖的 Python。

请先创建仓库内的 uv 虚拟环境：
  cd Frontier
  uv venv .venv
  uv pip install --python .venv/bin/python -e .

或通过 PYTHON_BIN 显式指定一个已安装依赖的解释器：
  PYTHON_BIN=/path/to/python ./run_experiments.sh
EOF
    exit 2
  fi
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