#!/usr/bin/env bash
# 重新入库脚本 - 清理并重新生成所有数据和索引
# 使用方法: conda activate multimodal && bash 重新入库.sh

set -euo pipefail

# 检查是否在正确的环境中
if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "$CONDA_DEFAULT_ENV" != "multimodal" ]; then
    echo "✗ 错误: 当前在 $CONDA_DEFAULT_ENV 环境，请切换到 multimodal 环境"
    echo "运行: conda activate multimodal && bash $0"
    exit 1
fi

# 如果没有激活任何 conda 环境，尝试自动激活
if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "检测到未激活 conda 环境，尝试自动激活..."

    # 查找并初始化 conda
    if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
        source "/root/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo "✗ 错误: 找不到 conda，请手动激活环境"
        echo "运行: conda activate multimodal && bash $0"
        exit 1
    fi

    # 激活环境
    conda activate multimodal

    # 再次检查
    if [ "$CONDA_DEFAULT_ENV" != "multimodal" ]; then
        echo "✗ 错误: 自动激活失败，请手动激活环境"
        echo "运行: conda activate multimodal && bash $0"
        exit 1
    fi
fi

echo "✓ 当前环境: $CONDA_DEFAULT_ENV"
echo "✓ Python 路径: $(which python)"
echo ""

# 获取脚本所在目录
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "=== 开始重新入库 ==="

# 1. 清理旧数据
echo "步骤 1/4: 清理旧数据..."
rm -rf poc/data/embeddings/*
rm -rf poc/data/index/*
rm -f poc/data/metadata.db
echo "✓ 清理完成"

# 2. 重新导入结构化数据
echo ""
echo "步骤 2/4: 导入结构化数据..."
python -m poc.pipeline.ingest
echo "✓ 导入完成"

# 3. 生成向量嵌入
echo ""
echo "步骤 3/4: 生成向量嵌入（这可能需要几分钟）..."
python -m poc.pipeline.embed
echo "✓ 嵌入完成"

# 4. 构建向量索引
echo ""
echo "步骤 4/4: 构建向量索引..."
python -m poc.search.index
echo "✓ 索引完成"

echo ""
echo "=== 重新入库完成 ==="
echo "现在可以重启服务: bash restart_poc.sh"
