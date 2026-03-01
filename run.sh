#!/bin/bash
# 一键运行 AI 指标分析流水线
# 用法: ./run.sh [--skip-llm]  或  ./run.sh --step 1|2|3

cd "$(dirname "$0")"

# 使用项目 venv
if [ ! -d ".venv" ]; then
  echo "❌ 未找到 .venv，请先运行: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

# 避免 HuggingFace 镜像 404
unset HF_ENDPOINT HF_MIRROR

# 避免 Matplotlib 字体缓存写入权限问题
export MPLCONFIGDIR="${PWD}/.matplotlib_cache"
mkdir -p "$MPLCONFIGDIR" 2>/dev/null || true

# 运行
.venv/bin/python main.py "$@"
