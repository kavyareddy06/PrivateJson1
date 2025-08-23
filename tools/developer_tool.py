# tools/retriever_tool.py
import os
import json
from typing import List, Dict
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
from config import INDEX_DIR
from crewai.tools import tool

# Load once (fast)
_INDEX_PATH = os.path.join(INDEX_DIR, "tfidf.index")
_VECTOR = None
_DOCS = None
_VECTORIZER = None

def _load_index():
    global _VECTOR, _DOCS, _VECTORIZER
    if _VECTOR is None or _DOCS is None or _VECTORIZER is None:
        if not os.path.exists(_INDEX_PATH):
            raise RuntimeError(f"Index not found at {_INDEX_PATH}. Run `python ingest.py` first.")
        data = load(_INDEX_PATH)
        _VECTORIZER = data["vectorizer"]
        _VECTOR = data["matrix"]
        _DOCS = data["docs"]

def _search_tfidf(query: str, k: int = 6) -> List[Dict]:
    _load_index()
    qv = _VECTORIZER.transform([query])
    sims = cosine_similarity(qv, _VECTOR).ravel()
    top_idx = sims.argsort()[::-1][:k]
    results = []
    for i in top_idx:
        d = _DOCS[i]
        results.append({
            "score": float(sims[i]),
            "source": d.get("source", "unknown"),
            "type": d.get("type", "unknown"),
            "content": d.get("content", "")
        })
    return results

@tool("kb_retriever")
def retriever_tool(query: str) -> str:
    """
    Retrieve relevant chunks from the KB (JSON + PDFs) using TF-IDF.
    Returns a JSON string with fields: { "query", "results":[{score, source, type, content}, ...], "raw_text": "..."}
    """
    hits = _search_tfidf(query, k=8)
    raw_text = "\n---\n".join([f"[{h['source']} | {h['type']} | {h['score']:.3f}]\n{h['content']}" for h in hits])
    payload = {
        "query": query,
        "results": hits,
        "raw_text": raw_text
    }
    return json.dumps(payload, ensure_ascii=False)
