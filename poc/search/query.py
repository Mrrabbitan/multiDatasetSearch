import argparse
import json
import math
from pathlib import Path
from typing import Optional

from poc.pipeline.utils import load_yaml, resolve_path

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Image = None

try:
    import lancedb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    lancedb = None


def load_model(model_name: str, cache_dir: Optional[str] = None, hf_mirror: Optional[str] = None):
    """
    加载CLIP模型，支持设置缓存目录和镜像源

    Args:
        model_name: 模型名称
        cache_dir: 模型缓存目录
        hf_mirror: HuggingFace镜像源URL
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("sentence-transformers not installed.") from exc

    # 设置HuggingFace镜像源（国内加速）
    if hf_mirror:
        import os
        # 设置多个环境变量以确保兼容性
        os.environ['HF_ENDPOINT'] = hf_mirror
        os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir if cache_dir else os.path.expanduser('~/.cache/huggingface')
        print(f"使用HuggingFace镜像源: {hf_mirror}")

    # 加载模型
    if cache_dir:
        from pathlib import Path
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        model = SentenceTransformer(model_name, cache_folder=str(cache_path))
    else:
        model = SentenceTransformer(model_name)

    return model


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


def build_lance_filter(
    event_type: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: float = 5.0,
) -> Optional[str]:
    """
    构建 LanceDB 过滤条件（SQL WHERE 语法）
    """
    conditions = []

    if event_type:
        conditions.append(f"event_type = '{event_type}'")

    if start_time:
        # 使用 alarm_time 或 captured_at
        conditions.append(f"(alarm_time >= '{start_time}' OR (alarm_time = '' AND captured_at >= '{start_time}'))")

    if end_time:
        conditions.append(f"(alarm_time <= '{end_time}' OR (alarm_time = '' AND captured_at <= '{end_time}'))")

    if lat is not None and lon is not None:
        # 计算边界框
        lat_delta = radius_km / 111.0
        lon_delta = radius_km / (111.0 * max(0.1, math.cos(math.radians(lat))))
        min_lat = lat - lat_delta
        max_lat = lat + lat_delta
        min_lon = lon - lon_delta
        max_lon = lon + lon_delta
        conditions.append(f"lat >= {min_lat} AND lat <= {max_lat} AND lon >= {min_lon} AND lon <= {max_lon}")

    return " AND ".join(conditions) if conditions else None


def main() -> None:
    parser = argparse.ArgumentParser(description="POC multimodal search query with LanceDB")
    parser.add_argument("--config", default="poc/config/poc.yaml")
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

    if lancedb is None:
        raise RuntimeError("lancedb is required. Please pip install lancedb.")

    config = load_yaml(args.config)
    paths_cfg = config.get("paths", {})
    search_cfg = config.get("search", {})
    model_name = search_cfg.get("clip_model", "clip-ViT-B-32")
    cache_dir = search_cfg.get("model_cache_dir")
    hf_mirror = search_cfg.get("hf_mirror")
    lancedb_dir = resolve_path(paths_cfg.get("lancedb_dir", "poc/data/lancedb"))

    # 连接 LanceDB
    db = lancedb.connect(str(lancedb_dir))
    table = db.open_table("embeddings")

    # 生成查询向量
    if args.mock:
        # 获取表的向量维度
        sample = table.to_pandas().head(1)
        dims = len(sample['vector'].iloc[0])
        query_vec = np.random.rand(dims).astype("float32")
        query_vec /= max(1e-12, float(np.linalg.norm(query_vec)))
    else:
        model = load_model(model_name, cache_dir=cache_dir, hf_mirror=hf_mirror)
        query_vec = encode_query(
            model, args.text, resolve_path(args.image) if args.image else None
        ).astype("float32")

    # 构建过滤条件
    filter_str = build_lance_filter(
        event_type=args.event_type,
        start_time=args.start_time,
        end_time=args.end_time,
        lat=args.lat,
        lon=args.lon,
        radius_km=args.radius_km,
    )

    # 执行向量搜索
    query = table.search(query_vec.tolist()).limit(args.top_k)
    if filter_str:
        query = query.where(filter_str)

    results_df = query.to_pandas()

    # 转换为 JSON 格式
    results = []
    for _, row in results_df.iterrows():
        results.append({
            "asset_id": row["asset_id"],
            "score": float(row["_distance"]),  # LanceDB 返回距离，越小越相似
            "file_path": row["file_path"],
            "file_name": row["file_name"],
            "captured_at": row["captured_at"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "event_type": row["event_type"],
            "alarm_time": row["alarm_time"],
            "alarm_level": row["alarm_level"],
        })

    print(json.dumps(results, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
