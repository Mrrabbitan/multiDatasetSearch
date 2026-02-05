#!/usr/bin/env bash

# 获取脚本所在目录
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# 初始化 conda（解决 conda activate 问题）
echo "=== 初始化 Conda 环境 ==="
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    echo "✓ 找到 conda.sh: $HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/root/miniconda3/etc/profile.d/conda.sh"
    echo "✓ 找到 conda.sh: /root/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    echo "✓ 找到 conda.sh: $HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "✗ 错误: 找不到 conda.sh"
    echo "请手动运行: conda activate multimodal && bash $0"
    exit 1
fi

# 激活 conda 环境
echo "正在激活 multimodal 环境..."
conda activate multimodal

# 验证环境是否激活成功
if [ "$CONDA_DEFAULT_ENV" != "multimodal" ]; then
    echo "✗ 错误: conda 环境激活失败，当前环境是: $CONDA_DEFAULT_ENV"
    echo "请手动运行: conda activate multimodal && bash $0"
    exit 1
fi

# 设置严格模式（在 conda 激活后设置，避免 conda 命令触发 pipefail）
set -euo pipefail

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

# 3. 生成向量嵌入
echo "步骤 3/4: 生成向量嵌入..."
python -m poc.pipeline.embed

# 4. 构建向量索引
echo "步骤 4/4: 构建向量索引..."
python -m poc.search.index

echo "=== 重新入库完成 ==="
