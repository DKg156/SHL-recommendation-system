from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class RecoConfig:
    data_dir: str = "data"
    index_file: str = "faiss.index"
    meta_file: str = "meta.json"

    # MUST match build_index.py (or set EMBED_MODEL_NAME env var for both)
    embed_model_name: str = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

    # Bigger retrieval pool (catalog is small)
    retrieve_k: int = 250
    final_k: int = 10


class Recommender:
    """
    Loads FAISS + metadata + embedding model once.
    Provides retrieve -> (optional rerank) -> balance -> return top K.
    """

    def __init__(self, cfg: Optional[RecoConfig] = None):
        self.cfg = cfg or RecoConfig()
        self.index: Optional[faiss.Index] = None
        self.meta: Optional[Dict[str, Dict[str, Any]]] = None
        self.model: Optional[SentenceTransformer] = None

    def load(self) -> None:
        idx_path = os.path.join(self.cfg.data_dir, self.cfg.index_file)
        meta_path = os.path.join(self.cfg.data_dir, self.cfg.meta_file)

        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"Missing FAISS index: {idx_path} (run build_index.py)")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing metadata: {meta_path} (run build_index.py)")

        self.index = faiss.read_index(idx_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.model = SentenceTransformer(self.cfg.embed_model_name)

        # optional warmup
        _ = self.model.encode(["warmup"], normalize_embeddings=True)

        # sanity prints (helpful when debugging model/index mismatch)
        try:
            print(f"[REC] model={self.cfg.embed_model_name}")
            print(f"[REC] index_dim={self.index.d}")
        except Exception:
            pass

    def _require_loaded(self) -> None:
        if self.index is None or self.meta is None or self.model is None:
            raise RuntimeError("Recommender not loaded. Call load() at startup.")

    # ---------- NEW: query cleaning (prevents drift + timeouts on huge JDs) ----------
    @staticmethod
    def clean_query(q: str, max_chars: int = 1200) -> str:
        q = (q or "").strip()
        if not q:
            return ""
        q = re.sub(r"\s+", " ", q)
        if len(q) > max_chars:
            q = q[:max_chars]
        return q

    @lru_cache(maxsize=2048)
    def _embed_cached(self, text: str) -> np.ndarray:
        self._require_loaded()
        assert self.model is not None
        vec = self.model.encode([text], normalize_embeddings=True)
        return vec.astype("float32")

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        self._require_loaded()
        assert self.index is not None and self.meta is not None

        q = Recommender.clean_query(query)
        if not q:
            return []

        # clamp k to catalog size
        k = k or self.cfg.retrieve_k
        k = max(1, min(k, len(self.meta)))

        qvec = self._embed_cached(q)
        _, ids = self.index.search(qvec, k)

        out: List[Dict[str, Any]] = []
        for idx in ids[0]:
            item = self.meta.get(str(idx))
            if item:
                out.append(item)
        return out

    @staticmethod
    def _is_mixed_query(query: str) -> bool:
        q = query.lower()
        tokens = set(re.findall(r"[a-z0-9]+", q))

        tech_words = {
            "java", "python", "sql", "javascript", "js",
            "developer", "engineer", "coding", "programming",
            "backend", "frontend", "api", "spring", "django", "react", "node"
        }

        people_words = {
            "stakeholder", "stakeholders", "communication", "communicate",
            "team", "teamwork", "collaboration", "collaborate",
            "leadership", "lead", "influence", "influencing",
            "customer", "client", "clients", "business",
            "crossfunctional", "cross-functional"
        }

        tech = any(w in tokens for w in tech_words) or any(w in q for w in ["microservice", "microservices"])
        people = any(w in tokens for w in people_words) or any(w in q for w in ["collaborat", "stakeholder"])

        return tech and people

    @staticmethod
    def balance(items: List[Dict[str, Any]], query: str, k: int) -> List[Dict[str, Any]]:
        if len(items) <= k:
            return items

        if not Recommender._is_mixed_query(query):
            return items[:k]

        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for it in items:
            tts = it.get("test_type") or []
            if isinstance(tts, str):
                tts = [tts]
            if not tts:
                tts = ["Other"]

            key = "P" if "P" in tts else ("K" if "K" in tts else tts[0])
            buckets.setdefault(key, []).append(it)

        has_k = len(buckets.get("K", [])) > 0
        has_p = len(buckets.get("P", [])) > 0
        if not (has_k and has_p):
            return items[:k]

        k_quota = max(1, round(k * 0.6))
        p_quota = max(1, k - k_quota)

        out: List[Dict[str, Any]] = []

        for _ in range(k_quota):
            if buckets.get("K"):
                out.append(buckets["K"].pop(0))
            else:
                break

        for _ in range(p_quota):
            if buckets.get("P"):
                out.append(buckets["P"].pop(0))
            else:
                break

        seen = {id(x) for x in out}
        for it in items:
            if len(out) >= k:
                break
            if id(it) not in seen:
                out.append(it)
                seen.add(id(it))

        return out[:k]

    @staticmethod
    def _canonical_url(url: str) -> str:
        if not url:
            return url
        return url.replace(
            "https://www.shl.com/products/product-catalog/view/",
            "https://www.shl.com/solutions/products/product-catalog/view/"
        )

    @staticmethod
    def normalize_item(it: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "url": Recommender._canonical_url(it.get("url", "")),
            "name": it.get("name", ""),
            "adaptive_support": str(it.get("adaptive_support", "No")),
            "description": it.get("description", "") or "",
            "duration": int(it.get("duration", 0) or 0),
            "remote_support": str(it.get("remote_support", "No")),
            "test_type": list(it.get("test_type", [])) if isinstance(it.get("test_type", []), list)
            else [str(it.get("test_type", ""))],
        }