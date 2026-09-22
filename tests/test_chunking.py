from safety_rag.chunking import chunk_document, chunk_documents, estimate_tokens
from safety_rag.config import Settings

from conftest import CODE, make_doc


def test_heading_path_is_nested():
    doc = make_doc(
        "report.md",
        "# Chapter\n\nIntro paragraph that is long enough to survive the minimum "
        "chunk filter applied during chunking.\n\n## Section\n\nSection body text "
        "that is also comfortably long enough to be retained as its own chunk.\n",
    )
    headings = [c.heading for c in chunk_document(doc)]
    assert "Chapter" in headings
    assert "Chapter › Section" in headings


def test_code_chunks_are_named_after_their_symbol():
    chunks = chunk_document(make_doc("zone_monitor.py", CODE, kind="code"))
    assert any(c.heading == "ZoneMonitor" for c in chunks)
    assert all(c.kind == "code" for c in chunks)


def test_oversized_section_is_split_under_the_token_budget():
    settings = Settings(chunk_tokens=40, chunk_overlap_tokens=5, min_chunk_chars=10)
    body = " ".join(f"sentence number {i} about worksite safety." for i in range(60))
    chunks = chunk_document(make_doc("long.md", f"# Long\n\n{body}\n"), settings)
    assert len(chunks) > 1
    assert max(c.tokens for c in chunks) <= settings.chunk_tokens * 1.5


def test_consecutive_chunks_overlap():
    settings = Settings(chunk_tokens=30, chunk_overlap_tokens=15, min_chunk_chars=10)
    paragraphs = "\n\n".join(f"Paragraph {i} describes a safety rule in detail." for i in range(12))
    chunks = chunk_document(make_doc("overlap.md", paragraphs), settings)
    assert len(chunks) > 1
    first_tail = set(chunks[0].text.split())
    assert first_tail & set(chunks[1].text.split())


def test_every_chunk_id_is_unique(documents):
    chunks = chunk_documents(documents)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_short_document_still_yields_a_chunk():
    chunks = chunk_document(make_doc("tiny.md", "Short note."))
    assert len(chunks) == 1
    assert chunks[0].text == "Short note."


def test_token_estimate_scales_with_length():
    assert estimate_tokens("one two three") < estimate_tokens("one two three four five six")
