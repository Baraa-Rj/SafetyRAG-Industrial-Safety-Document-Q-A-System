from __future__ import annotations

import sys

import numpy as np

from .config import SETTINGS, Settings


class SentenceTransformerBackend:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 256,
        )
        return vectors.astype(np.float32)


def get_backend(settings: Settings = SETTINGS, quiet: bool = False):
    """Return a dense backend, or None when dense retrieval is unavailable.

    Absence is not an error: retrieval degrades to lexical-only.
    """
    if not settings.dense_enabled:
        return None
    try:
        return SentenceTransformerBackend(settings.dense_model)
    except Exception as exc:
        if not quiet:
            print(
                f"[safety-rag] dense retrieval unavailable ({type(exc).__name__}: {exc}); "
                "falling back to lexical-only",
                file=sys.stderr,
            )
        return None


def cosine_scores(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return matrix @ query_vector.astype(np.float32)
