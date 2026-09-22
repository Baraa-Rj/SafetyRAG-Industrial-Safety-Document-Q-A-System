from __future__ import annotations

import sys

import numpy as np

from .bm25 import BM25Index, tokenize
from .chunking import chunk_documents
from .config import SETTINGS, Settings, Source
from .embeddings import get_backend
from .loaders import load_sources as load_documents
from .store import Index, build_meta


def build_index(
    sources: list[Source],
    settings: Settings = SETTINGS,
    use_dense: bool = True,
    verbose: bool = True,
) -> Index:
    def log(message: str) -> None:
        if verbose:
            print(f"[safety-rag] {message}", file=sys.stderr)

    log("loading documents...")
    documents, failures = load_documents(sources)
    if not documents:
        raise RuntimeError("no documents loaded — check corpus.json source paths")
    log(f"loaded {len(documents)} documents")

    chunks = chunk_documents(documents, settings)
    log(f"built {len(chunks)} chunks")

    bm25 = BM25Index.build([tokenize(f"{c.heading}\n{c.text}") for c in chunks])
    log(f"indexed {len(bm25.vocab)} unique terms")

    embeddings = None
    dense_model = None
    if use_dense:
        backend = get_backend(settings)
        if backend is not None:
            log(f"embedding chunks with {backend.model_name}...")
            embeddings = backend.encode(
                [f"{c.display_source}\n{c.text}" for c in chunks]
            )
            dense_model = backend.model_name
            log(f"embeddings: {embeddings.shape}")

    fingerprints = {str(doc.path): doc.fingerprint for doc in documents}
    meta = build_meta(
        chunks,
        fingerprints,
        dense_model,
        [(str(f.path), f.reason) for f in failures],
    )
    return Index(chunks=chunks, bm25=bm25, embeddings=embeddings, meta=meta)


def current_fingerprints(sources: list[Source]) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for source in sources:
        for path in source.iter_files():
            stat = path.stat()
            fingerprints[str(path)] = f"{int(stat.st_mtime)}:{stat.st_size}"
    return fingerprints
