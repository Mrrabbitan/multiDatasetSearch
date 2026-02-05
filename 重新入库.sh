#!/bin/bash
# 重新入库脚本 - 清理并重新生成所有数据和索引
# 使用方法: bash 重新入库.sh

set -e

# 获取脚本所在目录
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# 查找并初始化 conda
CONDA_SH=""
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
fi

# 如果找到了 conda.sh 并且不在 multimodal 环境中，则重新执行脚本
if [ -n "$CONDA_SH" ] && [ "${CONDA_DEFAULT_ENV:-}" != "multimodal" ]; then
    echo "检测到当前环境: ${CONDA_DEFAULT_ENV:-none}"
    echo "正在切换到 multimodal 环境并重新执行脚本..."
    # 在新的 bash 中 source conda.sh，激活环境，然后执行脚本
    exec bash -c "source '$CONDA_SH' && conda activate multimodal && exec bash '$0' --already-activated"
fi

# 检查是否成功激活
if [ "${CONDA_DEFAULT_ENV:-}" != "multimodal" ]; then
    echo "✗ 错误: 无法自动切换到 multimodal 环境"
    echo "请手动运行: conda activate multimodal && bash 重新入库.sh"
    exit 1
fi

echo "✓ 当前环境: $CONDA_DEFAULT_ENV"
echo "✓ Python 路径: $(which python)"
echo ""

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
