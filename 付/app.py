import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from poc.pipeline.utils import connect_db, load_yaml, resolve_path
from poc.qa.answer import run_retrieval, run_sql
from poc.qa.nl2sql import parse_question
from poc.search.query import (
    apply_filters,
    bbox_filter,
    encode_query,
    fetch_asset_context,
    load_index,
    load_model,
)


def load_config() -> Dict:
    return load_yaml("config/poc.yaml")


def db_stats(db_path: Path) -> Dict[str, int]:
    conn = connect_db(db_path)
    stats = {
        "assets": conn.execute("SELECT COUNT(*) AS cnt FROM assets").fetchone()["cnt"],
        "events": conn.execute("SELECT COUNT(*) AS cnt FROM events").fetchone()["cnt"],
        "detections": conn.execute("SELECT COUNT(*) AS cnt FROM detections").fetchone()["cnt"],
        "annotations": conn.execute("SELECT COUNT(*) AS cnt FROM annotations").fetchone()["cnt"],
        "embeddings": conn.execute("SELECT COUNT(*) AS cnt FROM embeddings").fetchone()["cnt"],
    }
    conn.close()
    return stats


def run_search(config: Dict, text: str, filters: Dict, top_k: int) -> List[Dict]:
    index_dir = resolve_path(config.get("paths", {}).get("index_dir", "data/index"))
    db_path = resolve_path(config.get("paths", {}).get("db_path", "data/metadata.db"))
    model_name = config.get("search", {}).get("clip_model", "clip-ViT-B-32")

    meta, index_obj = load_index(index_dir)
    model = load_model(model_name)
    query_vec = encode_query(model, text, None)

    if meta.get("backend") == "faiss":
        import faiss  # type: ignore

        scores, idx = index_obj.search(query_vec[None, :], max(top_k * 3, top_k))
        pairs = list(zip(idx[0].tolist(), scores[0].tolist()))
    else:
        import numpy as np  # type: ignore

        vectors = index_obj
        scores = np.dot(vectors, query_vec)
        idx = np.argsort(-scores)[: max(top_k * 3, top_k)]
        pairs = list(zip(idx.tolist(), scores[idx].tolist()))

    asset_ids = meta.get("asset_ids", [])
    candidate_ids = [asset_ids[i] for i, _ in pairs if i < len(asset_ids)]

    conn = connect_db(db_path)
    assets = fetch_asset_context(conn, candidate_ids)
    conn.close()

    bbox = None
    if filters.get("lat") is not None and filters.get("lon") is not None:
        bbox = bbox_filter(filters.get("lat"), filters.get("lon"), filters.get("radius_km", 5.0))
    filtered = apply_filters(
        assets, filters.get("event_type"), filters.get("start_time"), filters.get("end_time"), bbox
    )

    results = []
    for i, score in pairs:
        if i >= len(asset_ids):
            continue
        asset_id = asset_ids[i]
        if asset_id not in filtered:
            continue
        info = filtered[asset_id]
        results.append({"asset_id": asset_id, "score": float(score), **info})
        if len(results) >= top_k:
            break
    return results


def render_search_interface(config: Dict):
    st.header("数据概览")
    db_path = resolve_path(config.get("paths", {}).get("db_path", "data/metadata.db"))
    stats = db_stats(db_path)
    st.json(stats)

    st.header("多模态检索")
    col1, col2 = st.columns([2, 1])
    with col1:
        query_text = st.text_input("检索文本", value="夜间烟火")
    with col2:
        top_k = st.number_input("返回数量", min_value=1, max_value=50, value=10)

    filter_event = st.text_input("事件类型过滤", value="")
    col3, col4, col5 = st.columns(3)
    with col3:
        start_time = st.text_input("开始时间", value="")
    with col4:
        end_time = st.text_input("结束时间", value="")
    with col5:
        radius_km = st.number_input("半径(公里)", min_value=1.0, max_value=50.0, value=5.0)

    col6, col7 = st.columns(2)
    with col6:
        lat = st.text_input("纬度(lat)", value="")
    with col7:
        lon = st.text_input("经度(lon)", value="")

    if st.button("检索"):
        filters = {
            "event_type": filter_event or None,
            "start_time": start_time or None,
            "end_time": end_time or None,
            "lat": float(lat) if lat else None,
            "lon": float(lon) if lon else None,
            "radius_km": radius_km,
        }
        results = run_search(config, query_text, filters, top_k)
        render_results(results)


def render_qa_interface(config: Dict, db_path: Path):
    st.header("智能问数")
    question = st.text_input("问题", value="近7天烟火告警数量有多少？")
    if st.button("解析并查询"):
        plan = parse_question(question)
        rows = run_sql(db_path, plan.sql, plan.params)
        response = {
            "intent": plan.intent,
            "filters": plan.filters,
            "result": rows[0]["cnt"] if plan.intent == "count" and rows else rows,
        }
        st.json(response)
        st.subheader("证据检索")
        try:
            evidence = run_retrieval(config, question, plan.filters)
            render_results(evidence)
        except Exception as exc:
            st.warning(f"证据检索不可用: {exc}")


def render_results(results: List[Dict]):
    for item in results:
        st.write(item)
        file_path = item.get("file_path")
        if file_path and Path(file_path).exists():
            st.image(file_path, width=360)


def render_labeling_interface_placeholder():
    """渲染自动标注页面"""
    st.header("自动标注")

    st.info("""
    ### 使用说明

    1. 在左侧边栏选择 **自动标注** 功能
    2. 配置模型路径和图像目录
    3. 点击 **加载模型** 加载YOLO模型
    4. 选择图片，点击 **自动检测** 进行标注
    5. 可以手动添加、编辑、删除标注
    6. 点击 **保存** 导出YOLO格式标签

    ### 标签格式

    标签保存为YOLO格式，每行:
    ```
    class_id x_center y_center width height
    ```

    ### 运行命令

    ```bash
    # 启动标注界面
    streamlit run poc/app/labeling_interface.py

    # 或从主界面进入
    streamlit run poc/app/app.py
    ```
    """)


def main() -> None:
    st.set_page_config(page_title="多模态视联POC", layout="wide")
    st.title("多模态视联POC演示")

    # 侧边栏导航
    st.sidebar.header("功能导航")
    page = st.sidebar.radio(
        "选择页面",
        ["多模态检索", "智能问数", "自动标注"]
    )

    config = load_config()

    if page == "多模态检索":
        render_search_interface(config)
    elif page == "智能问数":
        db_path = resolve_path(config.get("paths", {}).get("db_path", "data/metadata.db"))
        render_qa_interface(config, db_path)
    elif page == "自动标注":
        from pipeline.labeling_interface import render_labeling_interface
        render_labeling_interface()

    # # 底部操作指引
    # st.header("操作指引")
    # st.code(
    #     "\n".join(
    #         [
    #             "python -m poc.pipeline.ingest --config poc/config/poc.yaml",
    #             "python -m poc.pipeline.label_auto --config poc/config/poc.yaml",
    #             "python -m poc.pipeline.dataset_build --config poc/config/poc.yaml",
    #             "python -m poc.pipeline.train_yolov8 --config poc/config/poc.yaml",
    #             "python -m poc.pipeline.embed --config poc/config/poc.yaml",
    #             "python -m poc.search.index --config poc/config/poc.yaml",
    #             "",
    #             "# 启动自动标注界面",
    #             "streamlit run poc/pipeline/labeling_interface.py",
    #         ]
    #     )
    # )


if __name__ == "__main__":
    main()
