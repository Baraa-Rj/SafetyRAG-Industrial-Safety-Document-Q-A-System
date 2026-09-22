from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import SETTINGS, Settings
from .loaders import RawDoc

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_CODE_SYMBOL_RE = re.compile(r"^(?:async\s+)?(?:def|class)\s+(\w+)", re.M)
_WORD_RE = re.compile(r"\w+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_path: str
    label: str
    kind: str
    heading: str
    text: str
    tokens: int
    meta: dict = field(default_factory=dict)

    @property
    def display_source(self) -> str:
        name = Path(self.doc_path).name
        return f"{name} › {self.heading}" if self.heading else name


def estimate_tokens(text: str) -> int:
    return max(1, round(len(_WORD_RE.findall(text)) * 1.3))


def _pack(blocks: list[str], settings: Settings) -> list[str]:
    """Greedily pack blocks into windows, carrying tail blocks over as overlap."""
    windows: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for block in blocks:
        block_tokens = estimate_tokens(block)
        if current and current_tokens + block_tokens > settings.chunk_tokens:
            windows.append("\n\n".join(current))
            overlap: list[str] = []
            overlap_tokens = 0
            for prev in reversed(current):
                prev_tokens = estimate_tokens(prev)
                if overlap_tokens + prev_tokens > settings.chunk_overlap_tokens:
                    break
                overlap.insert(0, prev)
                overlap_tokens += prev_tokens
            current = overlap
            current_tokens = overlap_tokens
        current.append(block)
        current_tokens += block_tokens
    if current:
        windows.append("\n\n".join(current))
    return windows


def _split_oversized(block: str, settings: Settings) -> list[str]:
    if estimate_tokens(block) <= settings.chunk_tokens:
        return [block]
    sentences = re.split(r"(?<=[.!?])\s+", block)
    if len(sentences) == 1:
        words = block.split()
        span = max(1, int(settings.chunk_tokens / 1.3))
        return [" ".join(words[i : i + span]) for i in range(0, len(words), span)]
    return _pack(sentences, settings)


def _prose_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown-ish text into (heading_path, body) sections."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    body: list[str] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            sections.append((" › ".join(name for _, name in stack), content))

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            flush()
            body = []
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2)))
        else:
            body.append(line)
    flush()
    return sections or [("", text.strip())]


def _chunk_prose(doc: RawDoc, settings: Settings) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, body in _prose_sections(doc.text):
        blocks: list[str] = []
        for raw_block in re.split(r"\n\s*\n", body):
            block = raw_block.strip()
            if block:
                blocks.extend(_split_oversized(block, settings))
        for window in _pack(blocks, settings):
            chunks.append(_make_chunk(doc, heading, window, len(chunks)))
    return chunks


def _chunk_code(doc: RawDoc, settings: Settings) -> list[Chunk]:
    lines = doc.text.splitlines()
    boundaries = [0]
    for match in _CODE_SYMBOL_RE.finditer(doc.text):
        line_no = doc.text[: match.start()].count("\n")
        if line_no > boundaries[-1]:
            boundaries.append(line_no)
    boundaries.append(len(lines))

    chunks: list[Chunk] = []
    for start, end in zip(boundaries, boundaries[1:]):
        segment = "\n".join(lines[start:end]).strip("\n")
        if not segment.strip():
            continue
        symbols = _CODE_SYMBOL_RE.findall(segment)
        heading = symbols[0] if symbols else "module"
        for window in _pack(segment.split("\n\n"), settings):
            if window.strip():
                chunks.append(
                    _make_chunk(doc, heading, window, len(chunks), {"start_line": start + 1})
                )
    return chunks


def _make_chunk(doc: RawDoc, heading: str, text: str, ordinal: int, meta: dict | None = None) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.label}:{doc.path.name}#{ordinal:04d}",
        doc_path=str(doc.path),
        label=doc.label,
        kind=doc.kind,
        heading=heading,
        text=text.strip(),
        tokens=estimate_tokens(text),
        meta=meta or {},
    )


def chunk_document(doc: RawDoc, settings: Settings = SETTINGS) -> list[Chunk]:
    chunks = _chunk_code(doc, settings) if doc.kind == "code" else _chunk_prose(doc, settings)
    # A short chunk under a heading is a real section, not a fragment — dropping it on
    # length alone loses whole sections from the index.
    kept = [c for c in chunks if c.heading or len(c.text) >= settings.min_chunk_chars]
    return kept or chunks[:1]


def chunk_documents(docs: list[RawDoc], settings: Settings = SETTINGS) -> list[Chunk]:
    return [chunk for doc in docs for chunk in chunk_document(doc, settings)]
