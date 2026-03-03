from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from google import genai


class GeminiReranker:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.model_name = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
        self.enabled = bool(self.api_key)
        self.client = None

    def load(self) -> None:
        if not self.enabled:
            return
        self.client = genai.Client(api_key=self.api_key)

    @staticmethod
    def _compact_candidates(cands: List[Dict[str, Any]], max_desc: int = 240) -> List[Dict[str, Any]]:
        out = []
        for c in cands:
            desc = (c.get("description") or "").replace("\n", " ").strip()
            if len(desc) > max_desc:
                desc = desc[:max_desc] + "…"
            out.append(
                {
                    "url": c.get("url"),
                    "name": c.get("name"),
                    "test_type": c.get("test_type", []),
                    "remote_support": c.get("remote_support"),
                    "adaptive_support": c.get("adaptive_support"),
                    "description": desc,
                }
            )
        return out

    def rerank(self, query: str, candidates: List[Dict[str, Any]], k: int = 10) -> List[Dict[str, Any]]:
        """
        Rerank candidates using Gemini. One call max.
        Safe behavior: NEVER throws. On any failure, returns retrieval order.
        """
        # Hard fallback (always valid)
        fallback = candidates[:k]

        # If not configured, just return fallback
        if not self.enabled or self.client is None:
            return fallback

        compact = self._compact_candidates(candidates, max_desc=240)
        prompt = f"""
You are a ranking function.
Format EXACTLY:
{{"ranked_urls":["url1","url2",...]}}
Rules:
- ranked_urls length must be exactly {k}
- Only use URLs from the candidates list
- No duplicates
QUERY:
{query}
CANDIDATES (JSON):
{json.dumps(compact, ensure_ascii=False)}
"""
        
        try:
            # IMPORTANT: with google.genai, use client.models.generate_content
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={"temperature": 0.0,
                "response_mime_type": "application/json",
                },
            )

            # Helpful debug: confirms LLM path is being hit
            print("[LLM] Gemini rerank invoked")
            print("[LLM] raw text:", repr(resp.text))
            txt = (resp.text or "").strip()
            if not txt:
                print("[LLM] Empty response text, fallback")
                return fallback

            data = json.loads(txt)  # can throw -> caught below
            ranked_urls = data.get("ranked_urls", [])
            if not isinstance(ranked_urls, list) or not ranked_urls:
                print("[LLM] Bad JSON schema, fallback:", data)
                return fallback

            # Validate + map back to full candidate dicts (preserve original objects)
            by_url = {c.get("url"): c for c in candidates if c.get("url")}
            out = []
            seen = set()
            for u in ranked_urls:
                if u in by_url and u not in seen:
                    out.append(by_url[u])
                    seen.add(u)
                if len(out) >= k:
                    break

            # If model returned junk URLs, pad with retrieval order
            if len(out) < k:
                for c in candidates:
                    u = c.get("url")
                    if u and u not in seen:
                        out.append(c)
                        seen.add(u)
                    if len(out) >= k:
                        break

            return out[:k]

        except Exception as e:
            # Never break the API for LLM weirdness / quota / JSON errors
            print("[LLM] Gemini rerank failed, fallback:", repr(e))
            return fallback