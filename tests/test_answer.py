import pytest

from safety_rag.answer import answer, build_context
from safety_rag.retrieve import Retriever


@pytest.fixture
def results(index):
    return Retriever(index, backend=None).search("how are falls confirmed?", top_k=3)


def test_extractive_answer_cites_a_source(results):
    result = answer("how are falls confirmed?", results, mode="extractive")
    assert result.mode == "extractive"
    assert "[1]" in result.text or "[2]" in result.text
    assert "sustained" in result.text.lower()


def test_every_citation_maps_to_a_retrieved_source(results):
    import re

    result = answer("how are falls confirmed?", results, mode="extractive")
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", result.text)}
    assert cited
    assert max(cited) <= len(result.results)


def test_no_results_yields_an_explicit_refusal():
    result = answer("anything", [], mode="extractive")
    assert result.mode == "empty"
    assert "No indexed passage" in result.text


def test_question_unsupported_by_context_is_not_answered(index):
    results = Retriever(index, backend=None).search("cameras cost")
    result = answer("what is the airspeed of a swallow?", results, mode="extractive")
    assert "no passage" in result.text.lower()


def test_list_markers_are_stripped_from_extracted_sentences(index):
    from conftest import make_doc
    from safety_rag.chunking import chunk_document
    from safety_rag.retrieve import Result

    doc = make_doc("bullets.md", "# Rules\n\n- Workers must wear a helmet at all times on site.\n")
    chunk = chunk_document(doc)[0]
    result = answer(
        "helmet rules",
        [Result(chunk=chunk, score=1.0, bm25_score=1.0, dense_score=None)],
        mode="extractive",
    )
    assert "- - " not in result.text


def test_llm_mode_without_a_key_fails_loudly(results, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="no ANTHROPIC_API_KEY"):
        answer("question", results, mode="llm")


def test_auto_mode_falls_back_to_extractive_without_a_key(results, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert answer("how are falls confirmed?", results, mode="auto").mode == "extractive"


def test_context_blocks_are_numbered_for_citation(results):
    context = build_context(results)
    assert context.startswith("[1] source:")
    for number in range(1, len(results) + 1):
        assert f"[{number}] source:" in context
