from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .config import SETTINGS, Settings

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")

STOPWORDS = frozenset(
    """a an and are as at be by for from has have how i if in into is it its of on or
    that the their then there these this to was were what when where which who why with
    will would can could should do does did not no you your""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, plus sub-tokens of snake_case and camelCase identifiers.

    Emitting both the whole identifier and its parts lets `ppe_detector` match a query
    for "PPE detection" without losing exact-symbol matches.
    """
    tokens: list[str] = []
    for word in _WORD_RE.findall(text):
        lowered = word.lower()
        if lowered not in STOPWORDS and len(lowered) > 1:
            tokens.append(lowered)
        parts = [p.lower() for chunk in word.split("_") for p in _CAMEL_RE.findall(chunk)]
        if len(parts) > 1:
            tokens.extend(p for p in parts if len(p) > 1 and p not in STOPWORDS)
    return tokens


@dataclass
class BM25Index:
    vocab: dict[str, int]
    term_offsets: np.ndarray
    posting_docs: np.ndarray
    posting_tfs: np.ndarray
    idf: np.ndarray
    doc_len: np.ndarray
    avgdl: float

    @property
    def n_docs(self) -> int:
        return int(self.doc_len.shape[0])

    @classmethod
    def build(cls, documents: list[list[str]]) -> "BM25Index":
        vocab: dict[str, int] = {}
        doc_ids: list[int] = []
        term_ids: list[int] = []
        tfs: list[int] = []
        doc_len = np.zeros(len(documents), dtype=np.float32)

        for doc_id, tokens in enumerate(documents):
            doc_len[doc_id] = len(tokens)
            counts: dict[int, int] = {}
            for token in tokens:
                term_id = vocab.setdefault(token, len(vocab))
                counts[term_id] = counts.get(term_id, 0) + 1
            for term_id, count in counts.items():
                doc_ids.append(doc_id)
                term_ids.append(term_id)
                tfs.append(count)

        n_terms = len(vocab)
        term_array = np.asarray(term_ids, dtype=np.int64)
        order = np.argsort(term_array, kind="stable")
        posting_docs = np.asarray(doc_ids, dtype=np.int32)[order]
        posting_tfs = np.asarray(tfs, dtype=np.float32)[order]
        term_sorted = term_array[order]

        counts_per_term = np.bincount(term_sorted, minlength=n_terms)
        term_offsets = np.zeros(n_terms + 1, dtype=np.int64)
        np.cumsum(counts_per_term, out=term_offsets[1:])

        n_docs = max(1, len(documents))
        df = counts_per_term.astype(np.float32)
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        return cls(
            vocab=vocab,
            term_offsets=term_offsets,
            posting_docs=posting_docs,
            posting_tfs=posting_tfs,
            idf=idf,
            doc_len=doc_len,
            avgdl=float(doc_len.mean()) if len(documents) else 1.0,
        )

    def score(self, query: str, settings: Settings = SETTINGS) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        avgdl = self.avgdl or 1.0
        k1, b = settings.bm25_k1, settings.bm25_b
        for token in tokenize(query):
            term_id = self.vocab.get(token)
            if term_id is None:
                continue
            start, end = self.term_offsets[term_id], self.term_offsets[term_id + 1]
            docs = self.posting_docs[start:end]
            tf = self.posting_tfs[start:end]
            norm = k1 * (1.0 - b + b * self.doc_len[docs] / avgdl)
            scores[docs] += self.idf[term_id] * (tf * (k1 + 1.0)) / (tf + norm)
        return scores

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "term_offsets": self.term_offsets,
            "posting_docs": self.posting_docs,
            "posting_tfs": self.posting_tfs,
            "idf": self.idf,
            "doc_len": self.doc_len,
            "avgdl": np.asarray([self.avgdl], dtype=np.float64),
        }

    @classmethod
    def from_arrays(cls, vocab: dict[str, int], arrays) -> "BM25Index":
        return cls(
            vocab=vocab,
            term_offsets=arrays["term_offsets"],
            posting_docs=arrays["posting_docs"],
            posting_tfs=arrays["posting_tfs"],
            idf=arrays["idf"],
            doc_len=arrays["doc_len"],
            avgdl=float(arrays["avgdl"][0]),
        )
