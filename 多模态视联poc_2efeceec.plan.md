---
name: 多模态视联POC
overview: 基于离线样本的端到端POC，覆盖自动化标注、训练、多模态检索、结构化关联与智能问数，用于客户演示与可行性验证。
todos:
  - id: schema_ingest
    content: 定义统一元数据Schema并完成离线入库管线
    status: completed
  - id: auto_label
    content: 搭建YOLOv8自动标注与审核流
    status: completed
  - id: train_dataset
    content: 构建时空正负样本并完成训练评估
    status: completed
  - id: multimodal_search
    content: 建立CLIP向量索引与多模态检索
    status: completed
  - id: qa_layer
    content: 实现NL2SQL与检索联合问数
    status: completed
  - id: demo_delivery
    content: 完成演示界面与可行性报告
    status: completed
isProject: false
---

# 多模态视联POC开发计划

## 目标与范围

- 覆盖端到端能力：采集/入库 → 自动标注 → 训练 → 多模态检索 → 智能问数 → 演示
- 数据输入为离线批量样本（图片/视频+结构化导出），在本仓库新增独立目录，避免影响现有专利生成器代码

## 关键假设

- 以1-2个重点场景作为演示主线（默认：烟火、违法建设），后续可按客户确认替换
- 离线数据量控制在POC可跑通规模，支持单机GPU或小规模计算资源

## 技术方案与工作分解

1. 统一数据与知识体系

- 定义统一元数据与标签Schema：资产、事件/告警、检测结果、时空字段
- 建立对象存储（本地文件/目录）+ 结构化元数据存储（SQLite/轻量DB）+ 检索索引（FAISS）
- 产出：`[poc/schema/metadata.sql](poc/schema/metadata.sql)`、`[poc/config/poc.yaml](poc/config/poc.yaml)`

1. 数据入库与关联

- 解析离线样本（图片/视频）与结构化导出（CSV/JSON），建立asset_id绑定
- 支持时空字段（lat/lon/time）入库，为检索过滤与统计问数服务
- 产出：`[poc/pipeline/ingest.py](poc/pipeline/ingest.py)`、`[poc/pipeline/validate.py](poc/pipeline/validate.py)`

1. 自动化标注与审核流

- 基于YOLOv8预训练模型进行自动标注，输出COCO/YOLO格式
- 提供人工审核入口（Label Studio/CVAT对接脚本或导入导出流程）
- 产出：`[poc/pipeline/label_auto.py](poc/pipeline/label_auto.py)`、`[poc/pipeline/label_review_io.py](poc/pipeline/label_review_io.py)`、`[poc/docs/labeling_workflow.md](poc/docs/labeling_workflow.md)`

1. 训练与数据集构建

- 基于时空策略构建正负样本（同区域不同时段/同时间不同区域）
- 进行YOLOv8微调训练，输出基础指标（mAP、召回、误报率）
- 产出：`[poc/pipeline/dataset_build.py](poc/pipeline/dataset_build.py)`、`[poc/pipeline/train_yolov8.py](poc/pipeline/train_yolov8.py)`、`[poc/docs/training_report.md](poc/docs/training_report.md)`

1. 多模态检索

- 计算图像/关键帧的CLIP向量并建立FAISS索引
- 支持“文本→图片/视频片段”检索，并结合结构化过滤（场景、时间、位置、告警类型）
- 产出：`[poc/pipeline/embed.py](poc/pipeline/embed.py)`、`[poc/search/index.py](poc/search/index.py)`、`[poc/search/query.py](poc/search/query.py)`

1. 智能问数

- 实现自然语言到结构化查询（SQL/DSL）与检索联合回答
- 输出统计结果+关联样本（证据帧/片段）
- 产出：`[poc/qa/nl2sql.py](poc/qa/nl2sql.py)`、`[poc/qa/answer.py](poc/qa/answer.py)`、`[poc/docs/qa_examples.md](poc/docs/qa_examples.md)`

1. 演示与交付

- 构建轻量演示界面（Streamlit/FastAPI）：
  - 自动标注与审核结果展示
  - 多模态检索（文本/图像）+ 结构化过滤
  - 智能问数结果与证据展示
- 产出：`[poc/app/app.py](poc/app/app.py)`、`[poc/README.md](poc/README.md)`、`[poc/docs/feasibility.md](poc/docs/feasibility.md)`

## 里程碑与验收

- M1：完成数据入库与统一Schema，演示可检索样本
- M2：自动标注+审核流可跑通，形成训练集
- M3：完成训练与基础评估，具备检索与问数能力
- M4：演示界面与讲解脚本可对客户完整展示

