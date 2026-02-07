# 结构化数据说明（POC）

支持 CSV/JSON 格式，常用字段如下（大小写不敏感）：

- `file_name`：媒体文件名（用于关联图片/视频）
- `event_type`：场景标签（如 烟火/违法建设/盗采/火灾）
- `alarm_time`：告警时间（建议 `YYYY-MM-DD HH:MM:SS`）
- `lat` / `lon`：经纬度
- `alarm_level`：告警等级（可选）
- `alarm_source`：告警来源（可选）
- `region`：区域信息（可选）

建议先准备少量样本验证流程，再逐步扩展数据规模。
