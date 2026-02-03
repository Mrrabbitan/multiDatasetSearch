import argparse
import uuid
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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_images(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        return []
    return [p for p in dir_path.rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "sentence-transformers not installed. Please pip install sentence-transformers."
        ) from exc
    return SentenceTransformer(model_name)


def embed_images(model, images: List[Path]) -> List[tuple]:
    if Image is None or np is None:
        raise RuntimeError("Pillow and numpy required for embeddings.")
    outputs = []
    for path in images:
        image = Image.open(path).convert("RGB")
        vec = model.encode(image, convert_to_numpy=True, normalize_embeddings=True)
        outputs.append((path, vec))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="POC image embedding")
    parser.add_argument("--config", default="poc/config/poc.yaml")
    parser.add_argument("--output-dir", default="poc/data/embeddings")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if np is None:
        raise RuntimeError("numpy is required. Please pip install numpy.")

    config = load_yaml(args.config)
    paths_cfg = config.get("paths", {})
    search_cfg = config.get("search", {})
    model_name = search_cfg.get("clip_model", "clip-ViT-B-32")

    raw_images_dir = resolve_path(paths_cfg.get("raw_images_dir", "poc/data/raw/images"))
    db_path = resolve_path(paths_cfg.get("db_path", "poc/data/metadata.db"))
    output_dir = resolve_path(args.output_dir)
    ensure_dir(output_dir)

    images = discover_images(raw_images_dir)
    if args.mock:
        embeddings = [(path, np.random.rand(512).astype("float32")) for path in images]
        model_name = "mock"
    else:
        model = load_model(model_name)
        embeddings = embed_images(model, images)

    conn = connect_db(db_path)
    by_path = {
        row["file_path"]: row["asset_id"]
        for row in conn.execute("SELECT asset_id, file_path FROM assets").fetchall()
    }

    rows = []
    for path, vec in embeddings:
        asset_id = by_path.get(str(path))
        if not asset_id:
            continue
        vector_path = output_dir / f"{asset_id}.npy"
        np.save(vector_path, vec.astype("float32"))
        rows.append((uuid.uuid4().hex, asset_id, model_name, str(vector_path), int(vec.shape[0])))

    conn.executemany(
        """
        INSERT OR REPLACE INTO embeddings (
            embedding_id, asset_id, model_name, vector_path, dims
        ) VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()

    print(f"embedded: {len(rows)}")


if __name__ == "__main__":
    main()
