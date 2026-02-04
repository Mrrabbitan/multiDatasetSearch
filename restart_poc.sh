#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT"

echo "先尝试停止已有服务（如果存在）..."
if [[ -f "$ROOT/stop_poc.sh" ]]; then
  bash "$ROOT/stop_poc.sh" || true
fi

sleep 1

echo "启动服务..."
if [[ ! -f "$ROOT/start_poc.sh" ]]; then
  echo "未找到 $ROOT/start_poc.sh"
  exit 1
fi

bash "$ROOT/start_poc.sh"
