from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bm25 import tokenize
from .chunking import Chunk
from .config import SETTINGS, Settings
from .embeddings import cosine_scores, get_backend
from .store import Index

RRF_K = 60


@dataclass
class Result:
    chunk: Chunk
    score: float
    bm25_score: float
    dense_score: float | None

    @property
    def citation(self) -> str:
        return self.chunk.display_source


def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if scores.size == 0:
        return np.empty(0, dtype=np.int64)
    k = min(k, scores.size)
    partial = np.argpartition(-scores, k - 1)[:k]
    return partial[np.argsort(-scores[partial])]


def _reciprocal_rank_fusion(rankings: list[np.ndarray], n_items: int) -> np.ndarray:
    fused = np.zeros(n_items, dtype=np.float32)
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            fused[item] += 1.0 / (RRF_K + rank + 1)
    return fused


def _select(
    candidates: np.ndarray,
    relevance: np.ndarray,
    vectors: np.ndarray | None,
    top_k: int,
    duplicate_threshold: float,
) -> list[int]:
    """Take candidates in relevance order, skipping near-duplicates of what's chosen."""
    selected: list[int] = []
    for index in sorted(candidates, key=lambda i: -relevance[i]):
        if len(selected) >= top_k:
            break
        if selected and vectors is not None:
            if float(np.max(vectors[selected] @ vectors[index])) >= duplicate_threshold:
                continue
        selected.append(int(index))
    return selected


class Retriever:
    def __init__(self, index: Index, backend=None, settings: Settings = SETTINGS):
        self.index = index
        self.settings = settings
        self._backend = backend
        self._backend_resolved = backend is not None

    @property
    def backend(self):
        if not self._backend_resolved:
            self._backend = (
                get_backend(self.settings) if self.index.embeddings is not None else None
            )
            self._backend_resolved = True
        return self._backend

    @property
    def dense_available(self) -> bool:
        return self.index.embeddings is not None and self.backend is not None

    def search(
        self,
        query: str,
        top_k: int | None = None,
        labels: list[str] | None = None,
    ) -> list[Result]:
        top_k = top_k or self.settings.top_k
        chunks = self.index.chunks
        if not chunks:
            return []

        bm25_scores = self.index.bm25.score(query, self.settings)
        dense_scores: np.ndarray | None = None
        if self.dense_available:
            query_vector = self.backend.encode([query])[0]
            dense_scores = cosine_scores(query_vector, self.index.embeddings)

        mask = np.ones(len(chunks), dtype=bool)
        if labels:
            wanted = set(labels)
            mask = np.asarray([c.label in wanted for c in chunks], dtype=bool)
            if not mask.any():
                return []

        # Thresholds must be applied to the masked scores: a label-excluded chunk sits at
        # -inf there, but still carries its real score in the unmasked array.
        masked_bm25 = np.where(mask, bm25_scores, -np.inf)
        bm25_ranking = _top_indices(masked_bm25, self.settings.candidate_k)
        rankings = [bm25_ranking[masked_bm25[bm25_ranking] > 0]]
        if dense_scores is not None:
            masked_dense = np.where(mask, dense_scores, -np.inf)
            dense_ranking = _top_indices(masked_dense, self.settings.candidate_k)
            rankings.append(dense_ranking[masked_dense[dense_ranking] >= self.settings.dense_floor])

        fused = _reciprocal_rank_fusion(rankings, len(chunks))
        candidate_ids = sorted({int(i) for ranking in rankings for i in ranking}, key=lambda i: -fused[i])
        if not candidate_ids:
            return []
        candidates = np.asarray(candidate_ids, dtype=np.int64)

        ordered = _select(
            candidates,
            fused,
            self.index.embeddings,
            top_k,
            self.settings.duplicate_threshold,
        )
        cutoff = fused[ordered[0]] * self.settings.score_ratio_cutoff
        ordered = [i for i in ordered if fused[i] >= cutoff]
        return [
            Result(
                chunk=chunks[i],
                score=float(fused[i]),
                bm25_score=float(bm25_scores[i]),
                dense_score=float(dense_scores[i]) if dense_scores is not None else None,
            )
            for i in ordered
        ]


def keyword_overlap(query: str, text: str) -> float:
    query_terms = set(tokenize(query))
    if not query_terms:
        return 0.0
    text_terms = set(tokenize(text))
    return len(query_terms & text_terms) / len(query_terms)
