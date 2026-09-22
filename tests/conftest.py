from pathlib import Path

import pytest

from safety_rag.bm25 import BM25Index, tokenize
from safety_rag.chunking import chunk_documents
from safety_rag.loaders import RawDoc
from safety_rag.store import Index, build_meta

CORPUS = {
    "fall.md": (
        "# Fall detection\n\n"
        "A fall is confirmed only after the fallen class is sustained for several "
        "consecutive frames on the same track. Severity escalates from low to medium "
        "to high while the worker stays down.\n"
    ),
    "zones.md": (
        "# Zone monitoring\n\n"
        "Polygon geofences raise a breach when an unauthorised worker enters a "
        "restricted zone. Each zone carries a whitelist of permitted worker ids.\n"
    ),
    "hardware.md": (
        "# Pilot hardware\n\n"
        "The pilot uses four PoE cameras and an edge processing unit for on-site "
        "inference. Estimated total cost is four to six thousand dollars.\n"
    ),
}

CODE = (
    "import cv2\n\n\n"
    "class ZoneMonitor:\n"
    "    def is_allowed(self, worker_id):\n"
    "        return worker_id in self.allowed_workers\n"
)


def make_doc(name: str, text: str, kind: str = "prose", label: str = "docs") -> RawDoc:
    return RawDoc(path=Path(f"/corpus/{name}"), label=label, kind=kind, text=text, fingerprint="1:1")


@pytest.fixture
def documents() -> list[RawDoc]:
    docs = [make_doc(name, text) for name, text in CORPUS.items()]
    docs.append(make_doc("zone_monitor.py", CODE, kind="code", label="code"))
    return docs


@pytest.fixture
def index(documents) -> Index:
    chunks = chunk_documents(documents)
    bm25 = BM25Index.build([tokenize(f"{c.heading}\n{c.text}") for c in chunks])
    meta = build_meta(chunks, {str(d.path): d.fingerprint for d in documents}, None, [], len(documents))
    return Index(chunks=chunks, bm25=bm25, embeddings=None, meta=meta)
