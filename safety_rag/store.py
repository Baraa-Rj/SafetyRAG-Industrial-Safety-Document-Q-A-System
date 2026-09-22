from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .bm25 import BM25Index
from .chunking import Chunk


@dataclass
class Index:
    chunks: list[Chunk]
    bm25: BM25Index
    embeddings: np.ndarray | None
    meta: dict

    @property
    def dense_model(self) -> str | None:
        return self.meta.get("dense_model")

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        np.savez(directory / "bm25.npz", **self.bm25.to_arrays())
        (directory / "vocab.json").write_text(json.dumps(self.bm25.vocab), encoding="utf-8")
        if self.embeddings is not None:
            np.save(directory / "embeddings.npy", self.embeddings)
        elif (directory / "embeddings.npy").exists():
            (directory / "embeddings.npy").unlink()
        (directory / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "Index":
        if not (directory / "meta.json").exists():
            raise FileNotFoundError(
                f"no index at {directory} — build one with `safety-rag index`"
            )
        chunks = [
            Chunk(**json.loads(line))
            for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
        with np.load(directory / "bm25.npz") as arrays:
            bm25 = BM25Index.from_arrays(vocab, {k: arrays[k] for k in arrays.files})
        embeddings_path = directory / "embeddings.npy"
        embeddings = np.load(embeddings_path) if embeddings_path.exists() else None
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        return cls(chunks=chunks, bm25=bm25, embeddings=embeddings, meta=meta)


def build_meta(
    chunks: list[Chunk],
    fingerprints: dict[str, str],
    dense_model: str | None,
    failures: list[tuple[str, str]],
    n_documents: int,
) -> dict:
    labels: dict[str, int] = {}
    for chunk in chunks:
        labels[chunk.label] = labels.get(chunk.label, 0) + 1
    return {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_chunks": len(chunks),
        "n_documents": n_documents,
        "chunks_per_source": labels,
        "dense_model": dense_model,
        "fingerprints": fingerprints,
        "load_failures": [{"path": p, "reason": r} for p, r in failures],
    }


def stale_documents(meta: dict, current: dict[str, str]) -> dict[str, list[str]]:
    previous = meta.get("fingerprints", {})
    return {
        "added": sorted(set(current) - set(previous)),
        "removed": sorted(set(previous) - set(current)),
        "changed": sorted(p for p in set(previous) & set(current) if previous[p] != current[p]),
    }
