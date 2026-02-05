#!/usr/bin/env bash
set -euo pipefail

# 获取脚本所在目录
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# 初始化 conda（解决 conda activate 问题）
# 方法1：source conda.sh
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/root/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "错误: 找不到 conda.sh，请手动指定 conda 路径"
    exit 1
fi

# 激活 conda 环境
conda activate multimodal

echo "=== 开始重新入库 ==="
echo "当前环境: $(which python)"
echo "Python 版本: $(python --version)"

# 1. 清理旧数据
echo "步骤 1/3: 清理旧数据..."
rm -rf poc/data/embeddings/*
rm -rf poc/data/index/*
rm -f poc/data/metadata.db

# 2. 重新导入结构化数据
echo "步骤 2/3: 导入结构化数据..."
python -m poc.pipeline.ingest

# 3. 生成向量索引
echo "步骤 3/3: 生成向量索引..."
python -m poc.pipeline.embed

echo "=== 重新入库完成 ==="
