from safety_rag.retrieve import Retriever, keyword_overlap


def test_finds_the_relevant_chunk(index):
    results = Retriever(index, backend=None).search("how are falls confirmed?")
    assert results
    assert "fall" in results[0].chunk.text.lower()


def test_label_filter_restricts_results(index):
    results = Retriever(index, backend=None).search("zone worker", labels=["code"])
    assert results
    assert {r.chunk.label for r in results} == {"code"}


def test_unmatched_label_returns_nothing(index):
    assert Retriever(index, backend=None).search("zone", labels=["absent"]) == []


def test_query_with_no_lexical_match_returns_nothing(index):
    assert Retriever(index, backend=None).search("xylophone quantum tuba") == []


def test_results_are_ordered_by_score(index):
    results = Retriever(index, backend=None).search("zone worker permitted", top_k=5)
    assert results == sorted(results, key=lambda r: -r.score)


def test_top_k_caps_result_count(index):
    assert len(Retriever(index, backend=None).search("worker", top_k=1)) <= 1


def test_dense_is_reported_unavailable_without_embeddings(index):
    assert Retriever(index, backend=None).dense_available is False
    assert Retriever(index, backend=None).search("zone")[0].dense_score is None


def test_citation_names_the_source_document(index):
    result = Retriever(index, backend=None).search("pilot cameras cost")[0]
    assert "hardware.md" in result.citation


def test_keyword_overlap_is_bounded():
    assert keyword_overlap("helmet vest", "helmet and vest detected") == 1.0
    assert keyword_overlap("helmet vest", "nothing relevant here") == 0.0
    assert keyword_overlap("", "anything") == 0.0
