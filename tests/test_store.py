import numpy as np
import pytest

from safety_rag.store import Index, stale_documents


def test_index_round_trips_through_disk(index, tmp_path):
    index.save(tmp_path)
    restored = Index.load(tmp_path)

    assert len(restored.chunks) == len(index.chunks)
    assert restored.chunks[0].chunk_id == index.chunks[0].chunk_id
    assert restored.chunks[0].heading == index.chunks[0].heading
    assert restored.meta["n_chunks"] == index.meta["n_chunks"]
    assert restored.embeddings is None
    query = "zone breach worker"
    assert restored.bm25.score(query).tolist() == index.bm25.score(query).tolist()


def test_embeddings_round_trip(index, tmp_path):
    index.embeddings = np.ones((len(index.chunks), 4), dtype=np.float32)
    index.save(tmp_path)
    assert Index.load(tmp_path).embeddings.shape == (len(index.chunks), 4)


def test_stale_embeddings_are_removed_on_rebuild(index, tmp_path):
    index.embeddings = np.ones((len(index.chunks), 4), dtype=np.float32)
    index.save(tmp_path)
    index.embeddings = None
    index.save(tmp_path)
    assert Index.load(tmp_path).embeddings is None


def test_loading_a_missing_index_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="safety-rag index"):
        Index.load(tmp_path / "absent")


def test_staleness_classifies_each_change():
    meta = {"fingerprints": {"a": "1:1", "b": "1:1"}}
    diff = stale_documents(meta, {"a": "1:1", "b": "2:2", "c": "3:3"})
    assert diff == {"added": ["c"], "removed": [], "changed": ["b"]}


def test_unchanged_corpus_is_not_stale():
    meta = {"fingerprints": {"a": "1:1"}}
    assert not any(stale_documents(meta, {"a": "1:1"}).values())
