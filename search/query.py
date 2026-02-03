import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from poc.pipeline.utils import connect_db, load_yaml, resolve_path

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Image = None

def load_index(index_dir: Path):
    meta_path = index_dir / "index_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    backend = meta.get("backend")
    if backend == "faiss" and faiss:
        index = faiss.read_index(str(index_dir / "index.faiss"))
        return meta, index
    vectors = np.load(index_dir / "index.npy")
    return meta, vectors


def load_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("sentence-transformers not installed.") from exc
    return SentenceTransformer(model_name)


def encode_query(model, text: Optional[str], image_path: Optional[Path]):
    if np is None:
        raise RuntimeError("numpy is required.")
    if text:
        return model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    if image_path:
        if Image is None:
            raise RuntimeError("Pillow is required for image query.")
        image = Image.open(image_path).convert("RGB")
        return model.encode(image, convert_to_numpy=True, normalize_embeddings=True)
    raise RuntimeError("Specify text or image for query.")


def bbox_filter(lat: Optional[float], lon: Optional[float], radius_km: float):
    if lat is None or lon is None:
        return None
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / (111.0 * max(0.1, math.cos(math.radians(lat))))
    return (lat - lat_delta, lat + lat_delta, lon - lon_delta, lon + lon_delta)


def fetch_asset_context(conn, asset_ids: List[str]) -> Dict[str, dict]:
    if not asset_ids:
        return {}
    placeholders = ",".join(["?"] * len(asset_ids))
    rows = conn.execute(
        f"""
        SELECT a.asset_id, a.file_path, a.file_name, a.captured_at, a.lat, a.lon,
               (SELECT e.event_type FROM events e WHERE e.asset_id = a.asset_id ORDER BY e.alarm_time DESC LIMIT 1) AS event_type
        FROM assets a
        WHERE a.asset_id IN ({placeholders})
        """,
        asset_ids,
    ).fetchall()
    return {row["asset_id"]: dict(row) for row in rows}


def apply_filters(
    assets: Dict[str, dict],
    event_type: Optional[str],
    start_time: Optional[str],
    end_time: Optional[str],
    bbox: Optional[Tuple[float, float, float, float]],
):
    results = {}
    for asset_id, info in assets.items():
        if event_type and info.get("event_type") != event_type:
            continue
        captured = info.get("captured_at")
        if start_time and captured and captured < start_time:
            continue
        if end_time and captured and captured > end_time:
            continue
        if bbox and info.get("lat") is not None and info.get("lon") is not None:
            if not (bbox[0] <= info["lat"] <= bbox[1] and bbox[2] <= info["lon"] <= bbox[3]):
                continue
        results[asset_id] = info
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="POC multimodal search query")
    parser.add_argument("--config", default="poc/config/poc.yaml")
    parser.add_argument("--index-dir", default="poc/data/index")
    parser.add_argument("--text")
    parser.add_argument("--image")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--event-type")
    parser.add_argument("--start-time")
    parser.add_argument("--end-time")
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--radius-km", type=float, default=5.0)
    args = parser.parse_args()

    if np is None:
        raise RuntimeError("numpy is required. Please pip install numpy.")

    config = load_yaml(args.config)
    model_name = config.get("search", {}).get("clip_model", "clip-ViT-B-32")
    db_path = resolve_path(config.get("paths", {}).get("db_path", "poc/data/metadata.db"))
    index_dir = resolve_path(args.index_dir)

    meta, index_obj = load_index(index_dir)
    if args.mock:
        dims = meta.get("dims")
        if not dims and hasattr(index_obj, "shape"):
            dims = index_obj.shape[1]
        if not dims:
            raise RuntimeError("cannot infer embedding dims for mock query.")
        query_vec = np.random.rand(int(dims)).astype("float32")
        query_vec /= max(1e-12, float(np.linalg.norm(query_vec)))
    else:
        model = load_model(model_name)
        query_vec = encode_query(
            model, args.text, resolve_path(args.image) if args.image else None
        ).astype("float32")

    if meta.get("backend") == "faiss" and faiss:
        scores, idx = index_obj.search(query_vec[None, :], max(args.top_k * 3, args.top_k))
        pairs = list(zip(idx[0].tolist(), scores[0].tolist()))
    else:
        vectors = index_obj
        scores = np.dot(vectors, query_vec)
        idx = np.argsort(-scores)[: max(args.top_k * 3, args.top_k)]
        pairs = list(zip(idx.tolist(), scores[idx].tolist()))

    asset_ids = meta.get("asset_ids", [])
    candidate_ids = [asset_ids[i] for i, _ in pairs if i < len(asset_ids)]

    conn = connect_db(db_path)
    assets = fetch_asset_context(conn, candidate_ids)
    conn.close()

    bbox = bbox_filter(args.lat, args.lon, args.radius_km) if args.lat and args.lon else None
    filtered = apply_filters(assets, args.event_type, args.start_time, args.end_time, bbox)

    results = []
    for i, score in pairs:
        if i >= len(asset_ids):
            continue
        asset_id = asset_ids[i]
        if asset_id not in filtered:
            continue
        info = filtered[asset_id]
        results.append({"asset_id": asset_id, "score": float(score), **info})
        if len(results) >= args.top_k:
            break

    print(json.dumps(results, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
