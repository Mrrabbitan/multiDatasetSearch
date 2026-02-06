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
rm -rf poc/data/lancedb/*
rm -f poc/data/metadata.db
echo "✓ 清理完成"

# 2. 重新创建数据库结构
echo ""
echo "步骤 2/4: 创建数据库结构..."
python -m poc.pipeline.ingest --config poc/config/poc.yaml
echo "✓ 数据库结构创建完成"

# 3. 导入告警数据（包含 summary 字段）
echo ""
echo "步骤 3/4: 导入告警数据（包含图像理解字段）..."
python import_warning_data.py
echo "✓ 告警数据导入完成"

# 4. 生成向量嵌入并写入 LanceDB
echo ""
echo "步骤 4/4: 生成向量嵌入并写入 LanceDB（这可能需要几分钟）..."
python -m poc.pipeline.embed --config poc/config/poc.yaml
echo "✓ 向量嵌入完成"

echo ""
echo "=== 重新入库完成 ==="
echo "数据统计："
python -c "import sqlite3; conn=sqlite3.connect('poc/data/metadata.db'); print(f'  - Assets: {conn.execute(\"SELECT COUNT(*) FROM assets\").fetchone()[0]}'); print(f'  - Events: {conn.execute(\"SELECT COUNT(*) FROM events\").fetchone()[0]}'); print(f'  - 包含图像理解: {conn.execute(\"SELECT COUNT(*) FROM events WHERE summary IS NOT NULL AND summary != \\\"\\\"\").fetchone()[0]}'); conn.close()"
echo ""
echo "现在可以启动应用: streamlit run poc/app/app_v2.py"
