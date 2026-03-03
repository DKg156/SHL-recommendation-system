from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.recommender import Recommender
from src.llm_rerank import GeminiReranker
from src.jd_extract import extract_text_from_url


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1)


class Assessment(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]


class RecommendResponse(BaseModel):
    recommended_assessments: List[Assessment]


REC: Optional[Recommender] = None
RERANK: Optional[GeminiReranker] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global REC, RERANK
    REC = Recommender()
    REC.load()

    RERANK = GeminiReranker()
    RERANK.load()

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    global REC, RERANK

    if REC is None:
        raise HTTPException(status_code=500, detail="Server not initialized")

    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must be non-empty")

    try:
        if query.startswith(("http://", "https://")):
            query_text = extract_text_from_url(query)
        else:
            query_text = query
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read JD URL: {e}")

    if not query_text:
        raise HTTPException(status_code=400, detail="Input text is empty after extraction")

    candidates = REC.retrieve(query_text, k=90)

    reranked = candidates
    if RERANK is not None:
        try:
            reranked = RERANK.rerank(query_text, candidates, k=15)
            print("[API] Gemini rerank attempted")
        except Exception as e:
            print("[API] Gemini failed, fallback:", repr(e))
            reranked = candidates[:30]

    final = REC.balance(reranked, query_text, k=10)
    normalized = [REC.normalize_item(x) for x in final]

    return {"recommended_assessments": normalized}