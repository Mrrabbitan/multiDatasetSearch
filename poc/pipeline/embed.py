import argparse
from pathlib import Path
from typing import List, Optional

from .utils import connect_db, load_yaml, resolve_path

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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_images(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        return []
    return [p for p in dir_path.rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
        raise RuntimeError(
            "sentence-transformers not installed. Please pip install sentence-transformers."
        ) from exc

    # 设置HuggingFace镜像源（国内加速）
    if hf_mirror:
        import os
        os.environ['HF_ENDPOINT'] = hf_mirror
        os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir if cache_dir else os.path.expanduser('~/.cache/huggingface')
        print(f"使用HuggingFace镜像源: {hf_mirror}")

    # 加载模型
    print(f"正在加载模型: {model_name}")
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        model = SentenceTransformer(model_name, cache_folder=str(cache_path))
    else:
        model = SentenceTransformer(model_name)

    # 获取向量维度（通过实际编码获取）
    try:
        dims = model.get_sentence_embedding_dimension()
        if dims is None:
            # 如果返回 None，通过编码一个测试样本获取维度
            test_vec = model.encode("test", convert_to_numpy=True)
            dims = test_vec.shape[0]
    except:
        # 备用方案：编码测试样本
        test_vec = model.encode("test", convert_to_numpy=True)
        dims = test_vec.shape[0]

    print(f"模型加载成功！维度: {dims}")
    return model


def embed_images(model, images: List[Path]) -> List[tuple]:
    if Image is None or np is None:
        raise RuntimeError("Pillow and numpy required for embeddings.")
    outputs = []
    for i, path in enumerate(images):
        if (i + 1) % 100 == 0:
            print(f"  处理进度: {i + 1}/{len(images)}")
        image = Image.open(path).convert("RGB")
        vec = model.encode(image, convert_to_numpy=True, normalize_embeddings=True)
        outputs.append((path, vec))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="POC image embedding with LanceDB")
    parser.add_argument("--config", default="poc/config/poc.yaml")
    parser.add_argument("--mock", action="store_true")
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

    raw_images_dir = resolve_path(paths_cfg.get("raw_images_dir", "poc/data/raw/images"))
    db_path = resolve_path(paths_cfg.get("db_path", "poc/data/metadata.db"))
    lancedb_dir = resolve_path(paths_cfg.get("lancedb_dir", "poc/data/lancedb"))
    ensure_dir(lancedb_dir)

    # 发现图片
    images = discover_images(raw_images_dir)
    print(f"发现 {len(images)} 张图片")

    # 生成向量
    if args.mock:
        embeddings = [(path, np.random.rand(512).astype("float32")) for path in images]
        model_name = "mock"
        dims = 512
    else:
        model = load_model(model_name, cache_dir=cache_dir, hf_mirror=hf_mirror)
        dims = model.get_sentence_embedding_dimension()
        print(f"开始生成向量嵌入...")
        embeddings = embed_images(model, images)

    # 从 SQLite 获取资产元数据
    conn = connect_db(db_path)
    assets_data = {}
    assets_by_filename = {}  # 新增：通过文件名索引
    for row in conn.execute("""
        SELECT a.asset_id, a.file_path, a.file_name, a.captured_at, a.lat, a.lon,
               e.event_type, e.alarm_time, e.alarm_level, e.summary, e.description,
               e.address, e.device_name, e.confidence_level
        FROM assets a
        LEFT JOIN events e ON a.asset_id = e.asset_id
    """).fetchall():
        row_dict = dict(row)
        assets_data[row["file_path"]] = row_dict
        # 同时通过文件名索引（处理路径不匹配的情况）
        if row["file_name"]:
            assets_by_filename[row["file_name"]] = row_dict
    conn.close()

    print(f"数据库中的资产数: {len(assets_data)}")
    print(f"通过文件名索引的资产数: {len(assets_by_filename)}")

    # 准备 LanceDB 数据
    lance_data = []
    matched_count = 0
    unmatched_count = 0

    for path, vec in embeddings:
        # 先尝试完整路径匹配
        asset_info = assets_data.get(str(path))

        # 如果路径不匹配，尝试通过文件名匹配
        if not asset_info:
            filename = Path(path).name
            asset_info = assets_by_filename.get(filename)

        if not asset_info:
            unmatched_count += 1
            if unmatched_count <= 5:  # 只打印前5个未匹配的
                print(f"  未匹配: {Path(path).name}")
            continue

        matched_count += 1

        lance_data.append({
            "asset_id": asset_info["asset_id"],
            "file_path": str(path),
            "file_name": asset_info["file_name"],
            "captured_at": asset_info["captured_at"] or "",
            "lat": float(asset_info["lat"]) if asset_info["lat"] is not None else 0.0,
            "lon": float(asset_info["lon"]) if asset_info["lon"] is not None else 0.0,
            "event_type": asset_info["event_type"] or "",
            "alarm_time": asset_info["alarm_time"] or "",
            "alarm_level": asset_info["alarm_level"] or "",
            "summary": asset_info.get("summary") or "",
            "description": asset_info.get("description") or "",
            "address": asset_info.get("address") or "",
            "device_name": asset_info.get("device_name") or "",
            "confidence_level": float(asset_info["confidence_level"]) if asset_info.get("confidence_level") else 0.0,
            "model_name": model_name,
            "vector": vec.tolist(),  # LanceDB 需要 list 格式
        })

    # 写入 LanceDB
    print(f"\n匹配统计:")
    print(f"  - 成功匹配: {matched_count} 条")
    print(f"  - 未匹配: {unmatched_count} 条")
    print(f"写入 LanceDB: {len(lance_data)} 条记录")
    db = lancedb.connect(str(lancedb_dir))

    # 如果表已存在，删除重建（全量更新模式）
    table_name = "embeddings"
    if table_name in db.table_names():
        db.drop_table(table_name)

    # 创建表并写入数据
    table = db.create_table(table_name, data=lance_data)

    # 创建向量索引（提升查询性能）
    print("创建向量索引...")
    table.create_index(metric="cosine", num_partitions=256, num_sub_vectors=96)

    print(f"✓ 完成！共处理 {len(lance_data)} 条记录")
    print(f"  - 模型: {model_name}")
    print(f"  - 维度: {dims}")
    print(f"  - 存储路径: {lancedb_dir}")


if __name__ == "__main__":
    main()
