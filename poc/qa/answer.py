import argparse
import json
import os
from typing import Any, Dict, List

from poc.pipeline.utils import connect_db, load_yaml, resolve_path
from poc.qa.guardrails import SQLGuardrail, SQLSecurityError

try:
    from poc.qa.nl2sql import build_query_plan  # type: ignore
except ImportError:  # pragma: no cover - backward compatibility with older nl2sql
    from poc.qa.nl2sql import parse_question as _parse_question  # type: ignore

    def build_query_plan(text: str, config: Dict):
        return _parse_question(text)


def run_sql(db_path, sql: str, params: List, validate_security: bool = True) -> List[Dict[str, Any]]:
    """
    执行 SQL 查询（带安全护栏）

    Args:
        db_path: 数据库路径
        sql: SQL 语句
        params: 参数列表
        validate_security: 是否启用安全检查（默认开启）

    Returns:
        查询结果列表

    Raises:
        SQLSecurityError: 如果 SQL 不安全
    """
    # 安全检查
    if validate_security:
        SQLGuardrail.validate_sql(sql)
        params = SQLGuardrail.sanitize_params(params)

    conn = connect_db(db_path)
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def run_retrieval(config, question: str, filters: Dict) -> List[Dict[str, Any]]:
    try:
        from poc.search.query import (
            apply_filters,
            bbox_filter,
            encode_query,
            fetch_asset_context,
            load_index,
            load_model,
        )
    except Exception:
        return []

    index_dir = resolve_path(config.get("paths", {}).get("index_dir", "poc/data/index"))
    model_name = config.get("search", {}).get("clip_model", "clip-ViT-B-32")
    db_path = resolve_path(config.get("paths", {}).get("db_path", "poc/data/metadata.db"))

    meta, index_obj = load_index(index_dir)
    mock_mode = os.getenv("POC_QUERY_MOCK") == "1"
    try:
        if mock_mode:
            raise RuntimeError("mock retrieval enabled")
        model = load_model(model_name)
        query_vec = encode_query(model, question, None)
    except Exception:
        try:
            import numpy as np  # type: ignore
        except Exception:
            return []
        dims = meta.get("dims")
        if not dims and hasattr(index_obj, "shape"):
            dims = index_obj.shape[1]
        if not dims:
            return []
        query_vec = np.random.rand(int(dims)).astype("float32")
        query_vec /= max(1e-12, float(np.linalg.norm(query_vec)))

    if meta.get("backend") == "faiss":
        import faiss  # type: ignore

        scores, idx = index_obj.search(query_vec[None, :], max(filters.get("top_k", 10), 10))
        pairs = list(zip(idx[0].tolist(), scores[0].tolist()))
    else:
        import numpy as np  # type: ignore

        vectors = index_obj
        scores = np.dot(vectors, query_vec)
        idx = np.argsort(-scores)[: max(filters.get("top_k", 10), 10)]
        pairs = list(zip(idx.tolist(), scores[idx].tolist()))

    asset_ids = meta.get("asset_ids", [])
    candidate_ids = [asset_ids[i] for i, _ in pairs if i < len(asset_ids)]

    conn = connect_db(db_path)
    assets = fetch_asset_context(conn, candidate_ids)
    conn.close()

    bbox = None
    if filters.get("lat") and filters.get("lon"):
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
        if len(results) >= filters.get("top_k", 10):
            break
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="POC intelligent QA")
    parser.add_argument("--config", default="poc/config/poc.yaml")
    parser.add_argument("--question", required=True)
    parser.add_argument("--with-evidence", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    db_path = resolve_path(config.get("paths", {}).get("db_path", "poc/data/metadata.db"))

    plan = build_query_plan(args.question, config)
    sql_rows = run_sql(db_path, plan.sql, plan.params)

    answer = {
        "question": args.question,
        "intent": plan.intent,
        "filters": plan.filters,
    }

    if plan.intent == "count":
        answer["result"] = sql_rows[0]["cnt"] if sql_rows else 0
    else:
        answer["result"] = sql_rows

    if args.with_evidence:
        answer["evidence"] = run_retrieval(config, args.question, plan.filters)

    print(json.dumps(answer, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
