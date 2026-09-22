from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from .bm25 import STOPWORDS, tokenize
from .retrieve import Result

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
CONTEXT_CHAR_BUDGET = 1800
MAX_EXTRACT_SENTENCES = 6

SYSTEM_PROMPT = (
    "You answer questions about a worksite safety-monitoring system using only the "
    "numbered context provided. Cite every claim with its source number in square "
    "brackets, e.g. [2]. If the context does not contain the answer, say so plainly "
    "instead of guessing."
)


@dataclass
class Answer:
    text: str
    mode: str
    results: list[Result]

    def format_citations(self) -> str:
        lines = []
        for number, result in enumerate(self.results, start=1):
            lines.append(f"  [{number}] {result.citation}  ({result.chunk.doc_path})")
        return "\n".join(lines)


def build_context(results: list[Result]) -> str:
    blocks = []
    for number, result in enumerate(results, start=1):
        text = result.chunk.text
        if len(text) > CONTEXT_CHAR_BUDGET:
            text = text[:CONTEXT_CHAR_BUDGET].rsplit(" ", 1)[0] + " ..."
        blocks.append(f"[{number}] source: {result.citation}\n{text}")
    return "\n\n".join(blocks)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    cleaned = []
    for part in parts:
        sentence = re.sub(r"\s+", " ", part).strip()
        sentence = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", sentence).strip()
        if sentence:
            cleaned.append(sentence)
    return cleaned


def _is_heading_fragment(sentence: str) -> bool:
    return len(sentence.split()) < 7 and not sentence.endswith((".", "!", "?", ":"))


def _extractive(question: str, results: list[Result]) -> str:
    query_terms = {t for t in tokenize(question) if t not in STOPWORDS}
    scored: list[tuple[float, bool, int, int, str]] = []

    for rank, result in enumerate(results):
        if result.chunk.kind == "code":
            summary = f"Implemented in `{result.chunk.display_source}`."
            scored.append((0.4 / (rank + 1), True, rank, 0, summary))
            continue
        for position, sentence in enumerate(_split_sentences(result.chunk.text)):
            words = sentence.split()
            if not 4 <= len(words) <= 80 or _is_heading_fragment(sentence):
                continue
            terms = set(tokenize(sentence))
            if not terms:
                continue
            overlap = len(query_terms & terms) / max(1, len(query_terms))
            if overlap == 0:
                continue
            score = overlap + 0.35 / (rank + 1) + (0.1 if position == 0 else 0.0)
            scored.append((score, False, rank, position, sentence))

    scored.sort(key=lambda item: -item[0])

    chosen: list[tuple[bool, int, int, str]] = []
    seen: set[str] = set()
    for _, is_code, rank, position, sentence in scored:
        key = sentence.lower()[:90]
        if key in seen:
            continue
        seen.add(key)
        chosen.append((is_code, rank, position, sentence))
        if len(chosen) >= MAX_EXTRACT_SENTENCES:
            break

    if not chosen:
        return "The indexed corpus contains no passage matching that question."

    chosen.sort(key=lambda item: (item[0], item[1], item[2]))
    return "\n".join(f"- {sentence} [{rank + 1}]" for _, rank, _, sentence in chosen)


def _anthropic_complete(prompt: str, api_key: str, model: str) -> str:
    payload = json.dumps(
        {
            "model": model,
            "max_tokens": 900,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()
    request = urllib.request.Request(
        ANTHROPIC_URL,
        data=payload,
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read())
    return "".join(block.get("text", "") for block in body.get("content", [])).strip()


def _openai_complete(prompt: str, api_key: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return (completion.choices[0].message.content or "").strip()


def _llm_answer(question: str, results: list[Result]) -> tuple[str, str] | None:
    prompt = f"Context:\n\n{build_context(results)}\n\nQuestion: {question}"
    model_override = os.environ.get("SAFETY_RAG_LLM_MODEL")

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        model = model_override or DEFAULT_ANTHROPIC_MODEL
        return _anthropic_complete(prompt, anthropic_key, model), f"llm:{model}"

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        model = model_override or DEFAULT_OPENAI_MODEL
        return _openai_complete(prompt, openai_key, model), f"llm:{model}"

    return None


def answer(question: str, results: list[Result], mode: str = "auto") -> Answer:
    """Synthesize an answer. `mode` is auto (LLM if a key exists), llm, or extractive."""
    if not results:
        return Answer(
            text="No indexed passage matches that question.",
            mode="empty",
            results=[],
        )

    if mode in ("auto", "llm"):
        try:
            generated = _llm_answer(question, results)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ImportError) as exc:
            if mode == "llm":
                raise RuntimeError(f"LLM generation failed: {exc}") from exc
            generated = None
        if generated is not None:
            return Answer(text=generated[0], mode=generated[1], results=results)
        if mode == "llm":
            raise RuntimeError(
                "no ANTHROPIC_API_KEY or OPENAI_API_KEY set — use --mode extractive"
            )

    return Answer(text=_extractive(question, results), mode="extractive", results=results)
