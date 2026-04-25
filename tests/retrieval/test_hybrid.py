import pytest
from unittest.mock import MagicMock

from chunk_norris.retrieval.hybrid import HybridRetriever, _RRF_K


# ── Helpers ───────────────────────────────────────────────────────────────────

VEC_A = [1.0, 0.0, 0.0]
VEC_B = [0.0, 1.0, 0.0]
VEC_C = [0.0, 0.0, 1.0]
VEC_Q = [1.0, 0.0, 0.0]   # identical to VEC_A


def make_embedder(vectors: list[list[float]]) -> MagicMock:
    embedder = MagicMock()
    embedder.embed.return_value = vectors
    return embedder


def make_chunk(text: str, index: int = 0) -> dict:
    return {"text": text, "metadata": {"chunk_index": index}}


# Real text chunks for BM25 keyword testing
CHUNK_REFUND  = make_chunk("The refund policy allows returns within 30 days.", 0)
CHUNK_SUPPORT = make_chunk("Contact support at support@example.com for help.", 1)
CHUNK_COMPANY = make_chunk("The company was founded in 2010 in San Francisco.", 2)
TEXT_CHUNKS = [CHUNK_REFUND, CHUNK_SUPPORT, CHUNK_COMPANY]


# ── TestHybridRetrieverInit ───────────────────────────────────────────────────

class TestHybridRetrieverInit:

    def test_stores_embedder(self):
        embedder = make_embedder([])
        retriever = HybridRetriever(embedder=embedder)
        assert retriever._embedder is embedder

    def test_dense_retriever_initialised(self):
        retriever = HybridRetriever(embedder=make_embedder([]))
        assert retriever._dense is not None

    def test_bm25_retriever_initialised(self):
        retriever = HybridRetriever(embedder=make_embedder([]))
        assert retriever._bm25 is not None

    def test_chunks_empty_before_index(self):
        retriever = HybridRetriever(embedder=make_embedder([]))
        assert retriever._chunks == []


# ── TestHybridRetrieverIndex ──────────────────────────────────────────────────

class TestHybridRetrieverIndex:

    def test_empty_chunks_raises(self):
        retriever = HybridRetriever(embedder=make_embedder([]))
        with pytest.raises(ValueError, match="chunks"):
            retriever.index([])

    def test_index_stores_chunks(self):
        embedder = MagicMock()
        embedder.embed.return_value = [VEC_A, VEC_B, VEC_C]
        retriever = HybridRetriever(embedder=embedder)
        retriever.index(TEXT_CHUNKS)
        assert retriever._chunks == TEXT_CHUNKS

    def test_index_builds_both_retrievers(self):
        embedder = MagicMock()
        embedder.embed.return_value = [VEC_A, VEC_B, VEC_C]
        retriever = HybridRetriever(embedder=embedder)
        retriever.index(TEXT_CHUNKS)
        assert retriever._dense._chunk_vectors is not None
        assert retriever._bm25._bm25 is not None


# ── TestHybridRetrieverRetrieve ───────────────────────────────────────────────

class TestHybridRetrieverRetrieve:

    def _make_indexed(self, chunks=None):
        """Returns an indexed HybridRetriever with mocked embedder."""
        chunks = chunks or TEXT_CHUNKS
        embedder = MagicMock()
        # Return vectors for index call + one vector per retrieve call
        embedder.embed.side_effect = [
            [VEC_A, VEC_B, VEC_C],  # index
            [VEC_Q],                 # retrieve query
        ]
        retriever = HybridRetriever(embedder=embedder)
        retriever.index(chunks)
        return retriever

    def test_raises_if_not_indexed(self):
        retriever = HybridRetriever(embedder=make_embedder([]))
        with pytest.raises(RuntimeError, match="index"):
            retriever.retrieve("query", top_k=3)

    def test_empty_query_raises(self):
        retriever = self._make_indexed()
        with pytest.raises(ValueError, match="query"):
            retriever.retrieve("", top_k=1)

    def test_whitespace_query_raises(self):
        retriever = self._make_indexed()
        with pytest.raises(ValueError, match="query"):
            retriever.retrieve("   ", top_k=1)

    def test_returns_list(self):
        retriever = self._make_indexed()
        results = retriever.retrieve("refund policy", top_k=2)
        assert isinstance(results, list)

    def test_returns_top_k_results(self):
        retriever = self._make_indexed()
        results = retriever.retrieve("refund policy", top_k=2)
        assert len(results) == 2

    def test_returns_fewer_if_not_enough_chunks(self):
        embedder = MagicMock()
        embedder.embed.side_effect = [[VEC_A], [VEC_Q]]
        retriever = HybridRetriever(embedder=embedder)
        retriever.index([make_chunk("only one chunk")])
        results = retriever.retrieve("query", top_k=5)
        assert len(results) == 1

    def test_semantic_score_in_metadata(self):
        retriever = self._make_indexed()
        results = retriever.retrieve("refund policy", top_k=1)
        assert "semantic_score" in results[0]["metadata"]

    def test_keyword_score_in_metadata(self):
        retriever = self._make_indexed()
        results = retriever.retrieve("refund policy", top_k=1)
        assert "keyword_score" in results[0]["metadata"]

    def test_score_in_metadata(self):
        retriever = self._make_indexed()
        results = retriever.retrieve("refund policy", top_k=1)
        assert "score" in results[0]["metadata"]

    def test_all_three_scores_present(self):
        retriever = self._make_indexed()
        results = retriever.retrieve("refund policy", top_k=3)
        for r in results:
            assert "semantic_score" in r["metadata"]
            assert "keyword_score"  in r["metadata"]
            assert "score"          in r["metadata"]

    def test_rrf_score_positive(self):
        retriever = self._make_indexed()
        results = retriever.retrieve("refund policy", top_k=3)
        for r in results:
            assert r["metadata"]["score"] > 0.0

    def test_results_sorted_by_rrf_score_descending(self):
        retriever = self._make_indexed()
        results = retriever.retrieve("refund policy", top_k=3)
        scores = [r["metadata"]["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_original_metadata_preserved(self):
        retriever = self._make_indexed()
        results = retriever.retrieve("refund", top_k=1)
        assert "chunk_index" in results[0]["metadata"]

    def test_keyword_match_influences_ranking(self):
        # A chunk with exact keyword match should rank highly
        # even if semantic similarity is not the best
        embedder = MagicMock()
        # All chunks get the same semantic vector — BM25 decides ranking
        embedder.embed.side_effect = [
            [VEC_A, VEC_A, VEC_A],  # index — identical semantic vectors
            [VEC_Q],                 # retrieve
        ]
        retriever = HybridRetriever(embedder=embedder)
        retriever.index(TEXT_CHUNKS)
        results = retriever.retrieve("refund policy", top_k=3)
        # With identical semantic scores, BM25 keyword match drives ranking
        assert "refund" in results[0]["text"].lower()

    def test_rrf_constant_is_standard(self):
        # k=60 is the standard RRF constant from the original paper
        assert _RRF_K == 60

    def test_returned_chunks_are_deep_copies(self):
        retriever = self._make_indexed()
        results = retriever.retrieve("refund", top_k=1)
        original_score = results[0]["metadata"]["score"]
        results[0]["metadata"]["score"] = 999
        # Re-retrieve
        retriever._dense._chunk_vectors  # still indexed
        embedder = retriever._embedder
        embedder.embed.side_effect = [[VEC_Q]]
        results2 = retriever.retrieve("refund", top_k=1)
        assert results2[0]["metadata"]["score"] != 999
