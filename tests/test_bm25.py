from safety_rag.bm25 import BM25Index, tokenize


def test_identifier_is_split_into_parts_and_kept_whole():
    tokens = tokenize("PPEDetector reads wet_floor_detector")
    assert "ppedetector" in tokens
    assert "ppe" in tokens and "detector" in tokens
    assert "wet_floor_detector" in tokens
    assert "wet" in tokens and "floor" in tokens


def test_stopwords_and_single_characters_are_dropped():
    assert tokenize("the a of x system") == ["system"]


def test_ranks_the_matching_document_first():
    docs = [
        tokenize("fall detection uses a temporal gate over the fallen class"),
        tokenize("QR badge scanning maps a tracked person to a worker id"),
    ]
    scores = BM25Index.build(docs).score("how does fall detection work")
    assert scores.argmax() == 0
    assert scores[1] == 0.0


def test_unknown_terms_score_zero_everywhere():
    index = BM25Index.build([tokenize("helmet and vest compliance")])
    assert index.score("quantum entanglement").sum() == 0.0


def test_rarer_term_outranks_common_term():
    docs = [
        tokenize("worker worker worker helmet"),
        tokenize("worker worker worker worker"),
    ]
    index = BM25Index.build(docs)
    assert index.score("helmet")[0] > index.score("helmet")[1]


def test_round_trips_through_arrays():
    index = BM25Index.build([tokenize("zone breach alert"), tokenize("helmet missing")])
    restored = BM25Index.from_arrays(index.vocab, index.to_arrays())
    assert restored.n_docs == index.n_docs
    assert restored.score("zone breach").tolist() == index.score("zone breach").tolist()
