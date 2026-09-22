from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .answer import answer as synthesize
from .config import DEFAULT_CORPUS_FILE, DEFAULT_INDEX_DIR, SETTINGS, load_sources
from .indexer import build_index, current_fingerprints
from .retrieve import Retriever
from .store import Index, stale_documents


def _load_retriever(index_dir: Path) -> Retriever:
    return Retriever(Index.load(index_dir))


def cmd_index(args: argparse.Namespace) -> int:
    sources = load_sources(args.corpus)

    if args.check:
        try:
            meta = Index.load(args.index_dir).meta
        except FileNotFoundError as exc:
            print(exc)
            return 1
        diff = stale_documents(meta, current_fingerprints(sources))
        total = sum(len(v) for v in diff.values())
        print(f"index built {meta['built_at']} — {meta['n_chunks']} chunks")
        if not total:
            print("index is up to date")
            return 0
        for kind, paths in diff.items():
            for path in paths:
                print(f"  {kind}: {path}")
        print(f"\n{total} document(s) changed — rebuild with `safety-rag index`")
        return 1

    index = build_index(sources, SETTINGS, use_dense=not args.no_dense)
    index.save(args.index_dir)

    print(f"indexed {index.meta['n_documents']} documents -> {index.meta['n_chunks']} chunks")
    for label, count in sorted(index.meta["chunks_per_source"].items()):
        print(f"  {label}: {count} chunks")
    print(f"dense model: {index.dense_model or 'disabled (lexical-only)'}")
    failures = [f for f in index.meta["load_failures"] if "no extractable text" not in f["reason"]]
    for failure in failures:
        print(f"  warning: {failure['path']} — {failure['reason']}", file=sys.stderr)
    print(f"saved to {args.index_dir}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    retriever = _load_retriever(args.index_dir)
    results = retriever.search(args.query, top_k=args.top_k, labels=args.label)
    if not results:
        print("no matches")
        return 1
    for number, result in enumerate(results, start=1):
        dense = f"{result.dense_score:.2f}" if result.dense_score is not None else "n/a"
        print(f"[{number}] {result.citation}")
        print(f"    score={result.score:.4f}  bm25={result.bm25_score:.2f}  dense={dense}")
        print(f"    {result.chunk.doc_path}")
        snippet = " ".join(result.chunk.text.split())[:240]
        print(f"    {snippet}...\n")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    retriever = _load_retriever(args.index_dir)
    results = retriever.search(args.question, top_k=args.top_k, labels=args.label)
    result = synthesize(args.question, results, mode=args.mode)

    print(result.text)
    if result.results:
        print(f"\nSources ({result.mode}):")
        print(result.format_citations())
    if args.show_context:
        print("\n--- retrieved context ---")
        for number, item in enumerate(result.results, start=1):
            print(f"\n[{number}] {item.citation}\n{item.chunk.text}")
    return 0 if results else 1


def cmd_stats(args: argparse.Namespace) -> int:
    meta = Index.load(args.index_dir).meta
    print(f"built at:     {meta['built_at']}")
    print(f"documents:    {meta['n_documents']}")
    print(f"chunks:       {meta['n_chunks']}")
    print(f"dense model:  {meta['dense_model'] or 'none (lexical-only)'}")
    print("chunks per source:")
    for label, count in sorted(meta["chunks_per_source"].items()):
        print(f"  {label}: {count}")
    if meta["load_failures"]:
        print(f"load failures: {len(meta['load_failures'])}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    import os

    def report(name: str, ok: bool, detail: str) -> bool:
        print(f"  [{'ok' if ok else '--'}] {name}: {detail}")
        return ok

    print("ingestion:")
    report("pdftotext", shutil.which("pdftotext") is not None, shutil.which("pdftotext") or "missing — PDFs will be skipped")
    try:
        import docx  # noqa: F401

        report("python-docx", True, "available")
    except ImportError:
        report("python-docx", False, "missing — .docx will be skipped")

    print("retrieval:")
    try:
        import sentence_transformers  # noqa: F401

        report("sentence-transformers", True, f"available ({SETTINGS.dense_model})")
    except ImportError:
        report("sentence-transformers", False, "missing — lexical-only retrieval")

    print("generation:")
    key = "ANTHROPIC_API_KEY" if os.environ.get("ANTHROPIC_API_KEY") else (
        "OPENAI_API_KEY" if os.environ.get("OPENAI_API_KEY") else None
    )
    report("llm key", key is not None, f"{key} set" if key else "none set — extractive answers")

    print("index:")
    try:
        index = Index.load(args.index_dir)
    except FileNotFoundError:
        report("index", False, f"absent — run `safety-rag index`")
        return 1
    report("index", True, f"{index.meta['n_chunks']} chunks, built {index.meta['built_at']}")
    diff = stale_documents(index.meta, current_fingerprints(load_sources(args.corpus)))
    total = sum(len(v) for v in diff.values())
    report("freshness", total == 0, "up to date" if total == 0 else f"{total} document(s) changed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="safety-rag",
        description="Retrieval-augmented QA over the safety-vision corpus.",
    )
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_FILE)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_index = subparsers.add_parser("index", help="build the index from corpus.json")
    p_index.add_argument("--no-dense", action="store_true", help="skip embeddings")
    p_index.add_argument("--check", action="store_true", help="report staleness, do not build")
    p_index.set_defaults(func=cmd_index)

    p_search = subparsers.add_parser("search", help="show ranked passages")
    p_search.add_argument("query")
    p_search.add_argument("-k", "--top-k", type=int, default=SETTINGS.top_k)
    p_search.add_argument("--label", action="append", help="restrict to a corpus source")
    p_search.set_defaults(func=cmd_search)

    p_ask = subparsers.add_parser("ask", help="answer a question with citations")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", "--top-k", type=int, default=SETTINGS.top_k)
    p_ask.add_argument("--label", action="append", help="restrict to a corpus source")
    p_ask.add_argument("--mode", choices=["auto", "llm", "extractive"], default="auto")
    p_ask.add_argument("--show-context", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    subparsers.add_parser("stats", help="summarize the built index").set_defaults(func=cmd_stats)
    subparsers.add_parser("doctor", help="check environment and index health").set_defaults(
        func=cmd_doctor
    )

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
