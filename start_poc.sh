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

# 假设已在外部执行过: conda activate multimodal
python -m streamlit run poc/app/app.py >>"$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" >"$PID_FILE"
echo "已启动多模态视联 POC，PID=$PID，日志文件: $LOG_FILE"
