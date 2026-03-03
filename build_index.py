import json
import os
import re
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


CATALOG_PATH = os.path.join("data", "catalog.json")
INDEX_PATH = os.path.join("data", "faiss.index")
META_PATH = os.path.join("data", "meta.json")

# Keep this stable unless you also change recommender.py to match
MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


def _clean(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


# --- NEW: filter out non-assessment artifacts that pollute retrieval ---
EXCLUDE_NAME_CONTAINS = [
    "report",
    "guide",
    "framework",
    "planner",
    "action planner",
    "interview guide",
    "job profiling",
    "360",
    "multi-rater",
    "feedback system",
    "ucf",  # many UCF items are guides/reports
]


def is_excluded(item: Dict[str, Any]) -> bool:
    name = (item.get("name") or "").lower()
    return any(x in name for x in EXCLUDE_NAME_CONTAINS)


def build_text(item: Dict[str, Any]) -> str:
    """
    Embed text emphasizing assessment identity + job-family signal.
    Keep it semantically rich (name + description) with light metadata.
    """
    name = _clean(item.get("name", ""))
    desc = _clean(item.get("description", ""))

    test_type = item.get("test_type") or []
    if isinstance(test_type, str):
        test_type = [test_type]
    tt = " ".join(test_type)

    duration = int(item.get("duration", 0) or 0)

    # Repeat name + use "Assessment" label to bias toward actual tests
    return (
        f"Assessment: {name}. {name}. "
        f"Test type: {tt}. "
        f"Duration: {duration} minutes. "
        f"{desc}"
    )


def main() -> None:
    if not os.path.exists(CATALOG_PATH):
        raise FileNotFoundError(f"Missing {CATALOG_PATH}. Run scrape_catalog.py first.")

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog: List[Dict[str, Any]] = json.load(f)

    if len(catalog) < 300:
        raise ValueError(f"Catalog too small ({len(catalog)}). Something is wrong with scraping.")

    print(f"[load] catalog items: {len(catalog)}")

    # --- NEW: apply filter ---
    filtered = [x for x in catalog if not is_excluded(x)]
    print(f"[filter] removed={len(catalog) - len(filtered)} kept={len(filtered)}")

    if len(filtered) < 250:
        raise ValueError(f"Filtered catalog too small ({len(filtered)}). Loosen EXCLUDE_NAME_CONTAINS.")

    texts = [build_text(x) for x in filtered]

    model = SentenceTransformer(MODEL_NAME)
    print(f"[embed] model={MODEL_NAME} ...")

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    emb = np.asarray(embeddings, dtype="float32")
    d = emb.shape[1]
    print(f"[embed] shape={emb.shape} dim={d}")

    index = faiss.IndexFlatIP(d)
    index.add(emb)
    print(f"[faiss] total vectors indexed: {index.ntotal}")

    faiss.write_index(index, INDEX_PATH)
    print(f"[save] {INDEX_PATH}")

    # IMPORTANT: meta must match filtered order
    meta = {str(i): filtered[i] for i in range(len(filtered))}
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[save] {META_PATH}")

    print("[done] index build complete")


if __name__ == "__main__":
    main()