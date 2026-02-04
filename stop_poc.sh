#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$ROOT/poc_app.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "未找到 PID 文件: $PID_FILE，可能服务未启动。"
  exit 0
fi

PID="$(tr -d '\n' <"$PID_FILE")"
if [[ -z "$PID" ]]; then
  echo "PID 文件为空，自动删除。"
  rm -f "$PID_FILE"
  exit 0
fi

if ! [[ "$PID" =~ ^[0-9]+$ ]]; then
  echo "PID 文件内容非法: $PID，自动删除。"
  rm -f "$PID_FILE"
  exit 0
fi

# 优先尝试使用 kill
if kill "$PID" 2>/dev/null; then
  echo "已发送 SIGTERM 到进程 PID=$PID"
  sleep 1
fi

# 如果还在，可以尝试强制杀掉（在 Windows Git Bash 中通常可用）
if kill -0 "$PID" 2>/dev/null; then
  echo "进程仍在运行，尝试强制结束 PID=$PID"
  kill -9 "$PID" 2>/dev/null || true
fi

rm -f "$PID_FILE"
echo "停止完成（如果 PID 存在的话）。"
