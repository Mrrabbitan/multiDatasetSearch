#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$ROOT/poc_app.pid"
LOG_FILE="$ROOT/poc_app.log"

if [[ -f "$PID_FILE" ]]; then
  echo "PID 文件已存在: $PID_FILE，可能已经在运行。如果确认没有运行，可以手动删除该文件。"
  exit 0
fi

cd "$ROOT"

# 初始化 conda（解决 conda activate 问题）
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/root/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "警告: 找不到 conda.sh，假设已在外部激活环境"
fi

# 激活 conda 环境（如果 conda 可用）
if command -v conda &> /dev/null; then
    conda activate multimodal || echo "警告: 无法激活 multimodal 环境，使用当前环境"
fi

echo "当前 Python: $(which python)"
python -m streamlit run poc/app/app.py >>"$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" >"$PID_FILE"
echo "已启动多模态视联 POC，PID=$PID，日志文件: $LOG_FILE"
