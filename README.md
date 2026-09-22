# Safety RAG

Retrieval-augmented question answering over the safety-vision corpus: the detection
source code, the graduation report, the pilot SOPs, and the requirements documents.
Ask a question in natural language, get an answer with a citation for every claim.

```
$ safety-rag ask "how does fall detection avoid false positives?"
- Fall detection is a temporal gate over the detector's dedicated fallen class, so a
  worker on the ground is detected directly without a separate classifier. [1]
- FallDetector confirms a fall only after the fallen class is sustained for several
  consecutive frames on the same track. [1]
- Implemented in `settings.py › FallDetectionConfig`. [3]

Sources (extractive):
  [1] implementation.tex › Implementation of the Vision Pipeline › Fall Detection
  [2] README.md › Safety Vision › Features
  [3] settings.py › FallDetectionConfig
```

## Corpus

Sources are declared in [`corpus.json`](corpus.json) — add a path there, not in code.
The default set spans five sources and ~450 chunks:

| Source | Content |
|---|---|
| `safety-vision-code` | `src/`, `config/`, `scripts/`, `tests/` of the safety-vision repo |
| `safety-vision-docs` | its README and `docs/` |
| `graduation-report` | the COMP4200 report LaTeX chapters |
| `requirements-and-sops` | requirements specs (.docx) and pilot SOPs (.pdf) |
| `system-writeups` | system overview PDFs |

`.py`, `.md`, `.tex`, `.pdf`, and `.docx` are ingested. PDFs go through poppler's
`pdftotext`; LaTeX is flattened to headings so chapters chunk by section instead of
arriving as one blob.

## Setup

Requires **Python 3.10+** and `pdftotext` (`sudo apt install poppler-utils`) for PDFs.

```bash
python3 -m venv --system-site-packages .venv
./.venv/bin/pip install -e ".[dense,docx,dev]"
./.venv/bin/python -m safety_rag.cli index
```

Check what's available before trusting a degraded answer:

```bash
./.venv/bin/python -m safety_rag.cli doctor
```

## Usage

```bash
safety-rag index                 # build the index from corpus.json
safety-rag index --check         # report which documents changed since the build
safety-rag index --no-dense      # lexical-only, no embedding model
safety-rag search "QR badge scanning" -k 5
safety-rag search "zone permissions" --label safety-vision-code
safety-rag ask "what is unfinished in the system?"
safety-rag ask "what hardware does the pilot need?" --show-context
safety-rag stats
```

## How it works

```
corpus.json ─▶ loaders ─▶ chunking ─▶ ┬─ BM25 index ──┐
                                      └─ embeddings ──┴─▶ RRF fusion ─▶ answer + citations
```

**Retrieval is hybrid.** BM25 handles exact terms — identifiers, `mAP@50`, `SR1.2` —
and its tokenizer emits both whole identifiers and their parts, so a query for "PPE
detection" reaches `ppe_detector.py`. Dense embeddings handle paraphrase, where the
question and the document share no vocabulary. The two ranked lists are fused with
reciprocal rank fusion.

Two guards keep a small corpus honest: a cosine floor, below which a chunk is
unrelated rather than weakly related, and a relevance cutoff that returns *fewer*
results rather than padding to `top_k` with noise. Only near-duplicates are
suppressed — broader diversity forcing was tried and removed, because it demoted the
correct evidence cluster.

**Answering degrades instead of failing.** With `ANTHROPIC_API_KEY` or
`OPENAI_API_KEY` set, an LLM writes the answer from the retrieved context, prompted to
cite sources and to refuse rather than guess. With no key — the default state of this
machine — answers are extractive: sentences selected from the retrieved chunks by
query overlap, each carrying its source number. Retrieval quality is identical either
way; only the prose differs.

The same applies to ingestion and retrieval. No `sentence-transformers` means
lexical-only. No `pdftotext` means PDFs are skipped. Each of these is reported by
`doctor` rather than failing at query time.

## Tests

```bash
./.venv/bin/python -m pytest
```

42 tests covering tokenization, BM25 ranking, chunk structure, LaTeX cleanup, loader
failure handling, index round-tripping, retrieval filters, and citation integrity.

## Limitations

- **Answer quality without an LLM key is extractive**, not generative: it quotes the
  corpus rather than synthesizing across sources. Set a key for prose answers.
- **No incremental indexing.** `index --check` detects staleness; the rebuild is
  full (~30s on this corpus).
- Scanned PDFs without a text layer yield nothing — there is no OCR step.
- Tables in PDFs flatten to text and lose their column structure.
