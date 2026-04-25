import pytest

from chunk_norris.retrieval.bm25 import BM25Retriever, _tokenize


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_chunk(text: str, index: int = 0) -> dict:
    return {"text": text, "metadata": {"chunk_index": index}}


CHUNKS = [
    make_chunk("The refund policy allows returns within 30 days.", 0),
    make_chunk("Contact support at support@example.com for help.", 1),
    make_chunk("The company was founded in 2010 in San Francisco.", 2),
]


# ── TestTokenize ──────────────────────────────────────────────────────────────

class TestTokenize:

    def test_lowercases_text(self):
        result = _tokenize("Hello World")
        assert result == ["hello", "world"]

    def test_splits_on_whitespace(self):
        result = _tokenize("one two three")
        assert result == ["one", "two", "three"]

    def test_empty_string_returns_empty_list(self):
        result = _tokenize("")
        assert result == []

    def test_single_word(self):
        result = _tokenize("hello")
        assert result == ["hello"]


# ── TestBM25RetrieverInit ─────────────────────────────────────────────────────

class TestBM25RetrieverInit:

    def test_chunks_empty_before_index(self):
        retriever = BM25Retriever()
        assert retriever._chunks == []

    def test_bm25_none_before_index(self):
        retriever = BM25Retriever()
        assert retriever._bm25 is None


# ── TestBM25RetrieverIndex ────────────────────────────────────────────────────

class TestBM25RetrieverIndex:

    def test_empty_chunks_raises(self):
        retriever = BM25Retriever()
        with pytest.raises(ValueError, match="chunks"):
            retriever.index([])

    def test_index_stores_chunks(self):
        retriever = BM25Retriever()
        retriever.index(CHUNKS)
        assert retriever._chunks == CHUNKS

    def test_index_builds_bm25(self):
        retriever = BM25Retriever()
        retriever.index(CHUNKS)
        assert retriever._bm25 is not None

    def test_reindexing_replaces_previous(self):
        retriever = BM25Retriever()
        retriever.index(CHUNKS)
        new_chunks = [make_chunk("completely different content")]
        retriever.index(new_chunks)
        assert retriever._chunks == new_chunks


# ── TestBM25RetrieverRetrieve ─────────────────────────────────────────────────

class TestBM25RetrieverRetrieve:

    def test_raises_if_not_indexed(self):
        retriever = BM25Retriever()
        with pytest.raises(RuntimeError, match="index"):
            retriever.retrieve("query", top_k=3)

    def test_empty_query_raises(self):
        retriever = BM25Retriever()
        retriever.index(CHUNKS)
        with pytest.raises(ValueError, match="query"):
            retriever.retrieve("", top_k=1)

    def test_whitespace_query_raises(self):
        retriever = BM25Retriever()
        retriever.index(CHUNKS)
        with pytest.raises(ValueError, match="query"):
            retriever.retrieve("   ", top_k=1)

    def test_returns_list(self):
        retriever = BM25Retriever()
        retriever.index(CHUNKS)
        results = retriever.retrieve("refund policy", top_k=2)
        assert isinstance(results, list)

    def test_returns_top_k_results(self):
        retriever = BM25Retriever()
        retriever.index(CHUNKS)
        results = retriever.retrieve("refund policy", top_k=2)
        assert len(results) == 2

    def test_returns_fewer_if_not_enough_chunks(self):
        retriever = BM25Retriever()
        retriever.index([make_chunk("only one chunk here")])
        results = retriever.retrieve("chunk", top_k=5)
        assert len(results) == 1

    def test_keyword_score_in_metadata(self):
        retriever = BM25Retriever()
        retriever.index(CHUNKS)
        results = retriever.retrieve("refund policy", top_k=1)
        assert "keyword_score" in results[0]["metadata"]

    def test_keyword_score_between_zero_and_one(self):
        retriever = BM25Retriever()
        retriever.index(CHUNKS)
        results = retriever.retrieve("refund policy", top_k=3)
        for r in results:
            assert 0.0 <= r["metadata"]["keyword_score"] <= 1.0

    def test_best_match_scores_one(self):
        # The best matching chunk always gets normalised score of 1.0
        retriever = BM25Retriever()
        retriever.index(CHUNKS)
        results = retriever.retrieve("refund policy returns days", top_k=3)
        assert results[0]["metadata"]["keyword_score"] == 1.0

    def test_keyword_match_ranks_first(self):
        # Chunk with exact keyword match should rank first
        retriever = BM25Retriever()
        retriever.index(CHUNKS)
        results = retriever.retrieve("refund policy", top_k=3)
        assert "refund" in results[0]["text"].lower()

    def test_no_keyword_overlap_scores_zero(self):
        # Query with no words in common with any chunk
        retriever = BM25Retriever()
        retriever.index(CHUNKS)
        results = retriever.retrieve("zzz qqq xxx", top_k=3)
        for r in results:
            assert r["metadata"]["keyword_score"] == 0.0

    def test_results_sorted_by_score_descending(self):
        retriever = BM25Retriever()
        retriever.index(CHUNKS)
        results = retriever.retrieve("refund policy", top_k=3)
        scores = [r["metadata"]["keyword_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_original_metadata_preserved(self):
        retriever = BM25Retriever()
        retriever.index(CHUNKS)
        results = retriever.retrieve("refund", top_k=1)
        assert "chunk_index" in results[0]["metadata"]

    def test_returned_chunks_are_deep_copies(self):
        retriever = BM25Retriever()
        retriever.index(CHUNKS)
        results = retriever.retrieve("refund", top_k=1)
        original_score = results[0]["metadata"]["keyword_score"]
        results[0]["metadata"]["keyword_score"] = 999
        results2 = retriever.retrieve("refund", top_k=1)
        assert results2[0]["metadata"]["keyword_score"] == original_score
