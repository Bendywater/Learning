import re
from pathlib import Path
from typing import Dict, List

import faiss
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .index_store import EmbeddingModel, load_metadata


def tokenize(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.lower())
    return [token for token in jieba.lcut(text) if token.strip()]


def min_max_normalize(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low, high = min(values), max(values)
    if abs(high - low) < 1e-9:
        return {idx: 1.0 for idx in scores}
    return {idx: (score - low) / (high - low) for idx, score in scores.items()}


class PDFRAGRetriever:
    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        embedding_model: str,
        reranker_model: str = "BAAI/bge-reranker-base",
    ):
        self.index = faiss.read_index(str(index_path))
        self.metadata = load_metadata(metadata_path)
        self.encoder = EmbeddingModel(embedding_model)
        self.bm25 = BM25Okapi([tokenize(row["text"]) for row in self.metadata])
        self.reranker = CrossEncoder(reranker_model) if reranker_model else None

    def dense_search(self, query: str, top_k: int) -> Dict[int, float]:
        vector = self.encoder.encode([query])
        scores, ids = self.index.search(np.asarray(vector, dtype="float32"), top_k)
        result: Dict[int, float] = {}
        for row_id, score in zip(ids[0], scores[0]):
            if row_id != -1:
                result[int(row_id)] = float(score)
        return result

    def bm25_search(self, query: str, top_k: int) -> Dict[int, float]:
        scores = self.bm25.get_scores(tokenize(query))
        if len(scores) == 0:
            return {}
        top_indices = np.argsort(scores)[::-1][:top_k]
        return {int(idx): float(scores[idx]) for idx in top_indices if scores[idx] > 0}

    def retrieve(
        self,
        query: str,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        rerank_top_k: int = 6,
    ) -> List[dict]:
        dense = min_max_normalize(self.dense_search(query, dense_top_k))
        bm25 = min_max_normalize(self.bm25_search(query, bm25_top_k))

        merged: Dict[int, float] = {}
        for row_id, score in dense.items():
            merged[row_id] = merged.get(row_id, 0.0) + 0.65 * score
        for row_id, score in bm25.items():
            merged[row_id] = merged.get(row_id, 0.0) + 0.35 * score

        candidates = sorted(merged.items(), key=lambda item: item[1], reverse=True)
        candidate_rows = []
        for row_id, hybrid_score in candidates[: max(rerank_top_k * 4, rerank_top_k)]:
            row = dict(self.metadata[row_id])
            row["hybrid_score"] = hybrid_score
            candidate_rows.append(row)

        if self.reranker and candidate_rows:
            pairs = [(query, row["text"]) for row in candidate_rows]
            rerank_scores = self.reranker.predict(pairs)
            for row, score in zip(candidate_rows, rerank_scores):
                row["rerank_score"] = float(score)
            candidate_rows.sort(key=lambda row: row["rerank_score"], reverse=True)
        else:
            for row in candidate_rows:
                row["rerank_score"] = row["hybrid_score"]

        return candidate_rows[:rerank_top_k]
