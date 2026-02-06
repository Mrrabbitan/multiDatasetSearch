#!/usr/bin/env python
"""
VL模型分析主入口脚本
处理图片和视频，使用VL模型进行分析
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.vl_analyze import (
    process_images,
    process_videos,
    main,
)

if __name__ == "__main__":
    main()
