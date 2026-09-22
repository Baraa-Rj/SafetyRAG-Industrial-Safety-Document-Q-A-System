from __future__ import annotations

import fnmatch
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_FILE = PROJECT_ROOT / "corpus.json"
DEFAULT_INDEX_DIR = PROJECT_ROOT / ".index"

MAX_FILE_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class Source:
    label: str
    root: Path
    include: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    kind: str = "prose"

    def iter_files(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        seen: set[Path] = set()
        for pattern in self.include:
            for path in self.root.glob(pattern):
                if not path.is_file() or path in seen:
                    continue
                rel = path.relative_to(self.root).as_posix()
                if any(fnmatch.fnmatch(rel, pat) for pat in self.exclude):
                    continue
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                seen.add(path)
        return sorted(seen)


@dataclass(frozen=True)
class Settings:
    chunk_tokens: int = 420
    chunk_overlap_tokens: int = 60
    min_chunk_chars: int = 120
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    top_k: int = 6
    candidate_k: int = 40
    mmr_lambda: float = 0.7
    dense_model: str = field(
        default_factory=lambda: os.environ.get(
            "SAFETY_RAG_DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    dense_enabled: bool = field(
        default_factory=lambda: os.environ.get("SAFETY_RAG_DENSE", "1") != "0"
    )


SETTINGS = Settings()


def _expand(raw: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(raw)))


def load_sources(corpus_file: Path | None = None) -> list[Source]:
    path = corpus_file or DEFAULT_CORPUS_FILE
    spec = json.loads(path.read_text())
    sources = []
    for entry in spec["sources"]:
        sources.append(
            Source(
                label=entry["label"],
                root=_expand(entry["root"]),
                include=tuple(entry["include"]),
                exclude=tuple(entry.get("exclude", ())),
                kind=entry.get("kind", "prose"),
            )
        )
    return sources
