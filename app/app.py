import json
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime, time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from poc.pipeline.utils import connect_db, load_yaml, resolve_path
from poc.qa.answer import run_retrieval, run_sql

try:
    from poc.qa.nl2sql import build_query_plan  # type: ignore
except ImportError:  # pragma: no cover - backward compatibility with older nl2sql
    from poc.qa.nl2sql import parse_question as _parse_question  # type: ignore

    def build_query_plan(text: str, config: Dict):
        return _parse_question(text)
from poc.search.query import (
    apply_filters,
    bbox_filter,
    encode_query,
    fetch_asset_context,
    load_index,
    load_model,
)


@st.cache_resource
def get_cached_model(model_name: str):
    """缓存CLIP模型，避免重复加载"""
    return load_model(model_name)


@st.cache_resource
def get_cached_index(index_dir: Path):
    """缓存向量索引，避免重复加载"""
    return load_index(index_dir)


def load_config() -> Dict:
    return load_yaml("poc/config/poc.yaml")


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
    """
    混合检索策略：先SQL预过滤，再向量检索排序
    优化点：
    1. 使用缓存的模型和索引，避免重复加载
    2. 先用SQL过滤出符合条件的asset_id集合（Pre-filtering）
    3. 再在这个子集上做向量检索和排序
    """
    index_dir = resolve_path(config.get("paths", {}).get("index_dir", "poc/data/index"))
    db_path = resolve_path(config.get("paths", {}).get("db_path", "poc/data/metadata.db"))
    model_name = config.get("search", {}).get("clip_model", "clip-ViT-B-32")

    # 使用缓存的模型和索引（性能优化1）
    meta, index_obj = get_cached_index(index_dir)
    model = get_cached_model(model_name)

    # 编码查询向量
    query_vec = encode_query(model, text, None)

    # === 混合检索策略优化：Pre-filtering ===
    # 第一步：先用SQL过滤出符合条件的asset_id集合
    conn = connect_db(db_path)

    # 构建SQL预过滤条件
    where_clauses = []
    params = []

    if filters.get("event_type"):
        where_clauses.append("e.event_type = ?")
        params.append(filters["event_type"])

    if filters.get("start_time"):
        where_clauses.append("e.alarm_time >= ?")
        params.append(filters["start_time"])

    if filters.get("end_time"):
        where_clauses.append("e.alarm_time <= ?")
        params.append(filters["end_time"])

    # 地理位置过滤
    bbox = None
    if filters.get("lat") is not None and filters.get("lon") is not None:
        bbox = bbox_filter(filters.get("lat"), filters.get("lon"), filters.get("radius_km", 5.0))
        if bbox:
            where_clauses.append("a.lat BETWEEN ? AND ?")
            where_clauses.append("a.lon BETWEEN ? AND ?")
            params.extend([bbox[0], bbox[1], bbox[2], bbox[3]])

    # 构建预过滤SQL
    where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    prefilter_sql = f"""
        SELECT DISTINCT a.asset_id
        FROM assets a
        LEFT JOIN events e ON a.asset_id = e.asset_id
        {where_sql}
    """

    prefiltered_rows = conn.execute(prefilter_sql, params).fetchall()
    prefiltered_asset_ids = {row["asset_id"] for row in prefiltered_rows}

    # 如果没有符合条件的数据，直接返回空
    if not prefiltered_asset_ids:
        conn.close()
        return []

    # 第二步：在预过滤的asset_id集合上做向量检索
    asset_ids = meta.get("asset_ids", [])

    # 找出预过滤集合在向量索引中的位置
    valid_indices = [i for i, aid in enumerate(asset_ids) if aid in prefiltered_asset_ids]

    if not valid_indices:
        conn.close()
        return []

    # 向量检索（只在预过滤的子集上计算相似度）
    if meta.get("backend") == "faiss":
        import faiss  # type: ignore
        # FAISS需要检索全量，然后过滤
        scores, idx = index_obj.search(query_vec[None, :], len(asset_ids))
        pairs = [(i, s) for i, s in zip(idx[0].tolist(), scores[0].tolist()) if i in valid_indices]
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:top_k]
    else:
        import numpy as np  # type: ignore
        vectors = index_obj
        # 只计算预过滤子集的相似度（性能优化）
        subset_vectors = vectors[valid_indices]
        scores = np.dot(subset_vectors, query_vec)
        sorted_idx = np.argsort(-scores)[:top_k]
        pairs = [(valid_indices[i], scores[i]) for i in sorted_idx]

    # 第三步：获取最终结果的详细信息
    candidate_ids = [asset_ids[i] for i, _ in pairs if i < len(asset_ids)]
    assets = fetch_asset_context(conn, candidate_ids)
    conn.close()

    # 构建最终结果
    results = []
    for i, score in pairs:
        if i >= len(asset_ids):
            continue
        asset_id = asset_ids[i]
        if asset_id not in assets:
            continue
        info = assets[asset_id]
        results.append({"asset_id": asset_id, "score": float(score), **info})

    return results


def render_results(results: List[Dict]):
    for item in results:
        st.write(item)
        file_path = item.get("file_path")
        if not file_path:
            continue
        path_obj = Path(file_path)
        if not path_obj.exists():
            continue
        suffix = path_obj.suffix.lower()
        if suffix in {".mp4", ".mov", ".avi", ".mkv"}:
            st.video(str(path_obj))
        else:
            st.image(str(path_obj), width=360)


def main() -> None:
    st.set_page_config(page_title="多模态视联POC", layout="wide")
    st.title("多模态视联POC演示")

    config = load_config()
    db_path = resolve_path(config.get("paths", {}).get("db_path", "poc/data/metadata.db"))

    st.sidebar.header("数据概览")
    stats = db_stats(db_path)
    st.sidebar.json(stats)

    st.header("多模态检索")
    col1, col2 = st.columns([2, 1])
    with col1:
        query_text = st.text_input("检索文本", value="车辆闯入监控告警")
    with col2:
        top_k = st.number_input("返回数量", min_value=1, max_value=50, value=10)

    filter_event = st.text_input("事件类型过滤", value="")
    col3, col4, col5 = st.columns(3)
    with col3:
        enable_time_filter = st.checkbox("启用时间过滤", value=False)
        start_date = st.date_input("开始日期")
        start_time_t = st.time_input("开始时间", value=time(0, 0))
    with col4:
        end_date = st.date_input("结束日期")
        end_time_t = st.time_input("结束时间", value=time(23, 59))
    with col5:
        radius_km = st.number_input("半径(公里)", min_value=1.0, max_value=50.0, value=5.0)

    col6, col7 = st.columns(2)
    with col6:
        lat = st.text_input("纬度(lat)", value="")
    with col7:
        lon = st.text_input("经度(lon)", value="")

    if st.button("检索"):
        start_time_str = None
        end_time_str = None
        if enable_time_filter:
            if start_date:
                start_dt = datetime.combine(start_date, start_time_t)
                start_time_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            if end_date:
                end_dt = datetime.combine(end_date, end_time_t)
                end_time_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")

        filters = {
            "event_type": filter_event or None,
            "start_time": start_time_str,
            "end_time": end_time_str,
            "lat": float(lat) if lat else None,
            "lon": float(lon) if lon else None,
            "radius_km": radius_km,
        }
        results = run_search(config, query_text, filters, top_k)
        render_results(results)

    st.header("智能问数")
    question = st.text_input("问题", value="近7天车辆闯入监控告警有多少条？")
    if st.button("解析并查询"):
        plan = build_query_plan(question, config)
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

    st.header("操作指引")
    st.code(
        "\n".join(
            [
                "python -m poc.pipeline.ingest --config poc/config/poc.yaml",
                "python -m poc.pipeline.label_auto --config poc/config/poc.yaml",
                "python -m poc.pipeline.dataset_build --config poc/config/poc.yaml",
                "python -m poc.pipeline.train_yolov8 --config poc/config/poc.yaml",
                "python -m poc.pipeline.embed --config poc/config/poc.yaml",
                "python -m poc.search.index --config poc/config/poc.yaml",
            ]
        )
    )


if __name__ == "__main__":
    main()
