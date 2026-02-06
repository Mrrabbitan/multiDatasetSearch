"""
导入告警明细表数据到数据库

用法：
    python import_warning_data.py
"""

import csv
import json
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime


def create_asset_id(warning_order_id: str, file_name: str) -> str:
    """根据工单ID和文件名生成唯一的 asset_id"""
    unique_key = f"{warning_order_id}_{file_name}"
    return hashlib.sha256(unique_key.encode()).hexdigest()


def import_warning_csv(csv_path: str, db_path: str):
    """导入告警明细表CSV到数据库"""

    # 连接数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 清空现有数据（可选）
    print("清空现有数据...")
    cursor.execute("DELETE FROM events")
    cursor.execute("DELETE FROM assets")
    conn.commit()

    # 读取CSV
    print(f"读取CSV文件: {csv_path}")
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        assets_inserted = 0
        events_inserted = 0

        for row in reader:
            try:
                # 提取关键字段
                warning_order_id = row.get('warning_order_id', '')
                alarm_time = row.get('alarm_time', '')
                warning_type_name = row.get('warning_type_name', '')
                latitude = row.get('latitude', '')
                longitude = row.get('longitude', '')
                address = row.get('address', '')
                channel_name = row.get('channel_name', '')
                device_name = row.get('device_name', '')
                video_url = row.get('video_url', '')
                file_img_url_src = row.get('file_img_url_src', '')
                file_img_url_icon = row.get('file_img_url_icon', '')
                summary = row.get('summary', '')  # 新增：图像理解字段
                description = row.get('description', '')
                confidence_level = row.get('confidence_level_max', row.get('confidence_level', ''))

                # 优先使用原图，如果没有则使用框图
                img_urls = file_img_url_src.split(',') if file_img_url_src else file_img_url_icon.split(',')
                url_path = img_urls[0].strip() if img_urls else ''
                file_name = Path(url_path).name if url_path else ''

                # 如果没有文件名，跳过这条记录
                if not file_name:
                    continue

                # 生成唯一的 asset_id（使用工单ID+文件名）
                asset_id = create_asset_id(warning_order_id, file_name)

                # 将 URL 路径转换为本地路径
                # 图片 URL: /12000000034/ThirdAlarm/pic/xxx.jpg -> warning_img/xxx.jpg
                # 视频 URL: /12000000034/ThirdAlarm/video/xxx.mp4 -> warning_file/xxx.mp4
                if file_name:
                    file_path = f"warning_img/{file_name}"
                else:
                    file_path = ''

                # 转换视频URL为本地路径
                local_video_url = ''
                if video_url:
                    video_filename = Path(video_url).name
                    if video_filename:
                        local_video_url = f"warning_file/{video_filename}"

                # 转换图片URL为本地路径（多个图片用逗号分隔）
                local_img_url_src = ''
                if file_img_url_src:
                    img_filenames = [Path(url.strip()).name for url in file_img_url_src.split(',') if url.strip()]
                    local_img_url_src = ','.join([f"warning_img/{name}" for name in img_filenames if name])

                local_img_url_icon = ''
                if file_img_url_icon:
                    img_filenames = [Path(url.strip()).name for url in file_img_url_icon.split(',') if url.strip()]
                    local_img_url_icon = ','.join([f"warning_img/{name}" for name in img_filenames if name])

                # 构建 extra_json（使用本地路径）
                extra_json = {
                    'warning_order_id': warning_order_id,
                    'warning_type_name': warning_type_name,
                    'address': address,
                    'channel_name': channel_name,
                    'device_name': device_name,
                    'video_url': local_video_url,  # 本地路径
                    'file_img_url_src': local_img_url_src,  # 本地路径
                    'file_img_url_icon': local_img_url_icon,  # 本地路径
                    'tenant_name': row.get('tenant_name', ''),
                    'province_name': row.get('province_name', ''),
                    'city_name': row.get('city_name', ''),
                    'county_name': row.get('county_name', ''),
                }

                # 插入 assets 表
                cursor.execute("""
                    INSERT OR REPLACE INTO assets
                    (asset_id, media_type, file_path, file_name, captured_at, lat, lon, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    asset_id,
                    'image',
                    file_path,
                    file_name,
                    alarm_time,
                    float(latitude) if latitude else None,
                    float(longitude) if longitude else None,
                    'warning_csv'
                ))
                assets_inserted += 1

                # 插入 events 表（使用 asset_id 作为 event_id，确保一对一关系）
                cursor.execute("""
                    INSERT OR REPLACE INTO events
                    (event_id, asset_id, event_type, alarm_level, alarm_source, alarm_time,
                     lat, lon, region, extra_json, summary, description, address, device_name, confidence_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    asset_id,  # 使用 asset_id 作为 event_id，确保唯一性
                    asset_id,
                    warning_type_name,
                    row.get('emergency_level', 'medium'),
                    row.get('warning_source_name', 'AI告警'),
                    alarm_time,
                    float(latitude) if latitude else None,
                    float(longitude) if longitude else None,
                    row.get('province_name', ''),
                    json.dumps(extra_json, ensure_ascii=False),
                    summary,  # 新增
                    description,  # 新增
                    address,  # 新增
                    device_name,  # 新增
                    float(confidence_level) if confidence_level else None  # 新增
                ))
                events_inserted += 1

                # 每100条提交一次
                if events_inserted % 100 == 0:
                    conn.commit()
                    print(f"已处理 {events_inserted} 条记录...")

            except Exception as e:
                print(f"处理行时出错: {e}")
                print(f"问题行: {row.get('warning_order_id', 'unknown')}")
                continue

        # 最终提交
        conn.commit()

        print(f"\n导入完成！")
        print(f"- Assets 插入: {assets_inserted} 条")
        print(f"- Events 插入: {events_inserted} 条")

        # 验证数据
        print("\n数据验证:")
        cursor.execute("SELECT COUNT(*) FROM assets")
        print(f"- Assets 总数: {cursor.fetchone()[0]}")

        cursor.execute("SELECT COUNT(*) FROM events")
        print(f"- Events 总数: {cursor.fetchone()[0]}")

        cursor.execute("SELECT event_type, COUNT(*) FROM events GROUP BY event_type")
        print("\n事件类型分布:")
        for row in cursor.fetchall():
            print(f"  - {row[0]}: {row[1]} 条")

        # 验证 summary 字段
        cursor.execute("SELECT COUNT(*) FROM events WHERE summary IS NOT NULL AND summary != ''")
        summary_count = cursor.fetchone()[0]
        print(f"\n包含图像理解的记录: {summary_count} 条")

    conn.close()


if __name__ == "__main__":
    csv_path = "最终标注入库数据.csv"  # 使用新的CSV文件
    db_path = "poc/data/metadata.db"

    print("="*60)
    print("告警明细表数据导入工具")
    print("="*60)
    print()

    if not Path(csv_path).exists():
        print(f"错误: CSV文件不存在: {csv_path}")
        exit(1)

    if not Path(db_path).exists():
        print(f"错误: 数据库文件不存在: {db_path}")
        print("请先运行: python -m poc.pipeline.ingest --config poc/config/poc.yaml")
        exit(1)

    import_warning_csv(csv_path, db_path)

    print()
    print("="*60)
    print("导入完成！现在可以在 Streamlit 中查询了")
    print("="*60)
    print()
    print("测试查询:")
    print('  python -c "import sqlite3; conn=sqlite3.connect(\'poc/data/metadata.db\'); print(conn.execute(\'SELECT COUNT(*) FROM events WHERE event_type LIKE \\\"%车辆%\\\"\').fetchone())"')
