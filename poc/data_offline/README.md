# 离线数据目录（POC）

本目录用于存放离线批量样本，不应提交真实数据到仓库。

目录约定：
- `images/`：图片样本（烟火/违法建设/盗采/火灾等场景）
- `videos/`：视频样本（用于抽帧与视频标注）
- `structured/`：结构化告警导出（CSV/JSON）

推荐做法：
- 样本来源与下载建议见 `poc/docs/sample_assets.md`
- 结构化字段说明见 `poc/data_offline/structured/README.md`
