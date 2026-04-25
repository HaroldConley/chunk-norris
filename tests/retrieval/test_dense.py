import pytest
from unittest.mock import MagicMock

from chunk_norris.retrieval.dense import DenseRetriever


# ── Helpers ───────────────────────────────────────────────────────────────────

VEC_A = [1.0, 0.0, 0.0]
VEC_B = [0.0, 1.0, 0.0]
VEC_C = [0.0, 0.0, 1.0]
VEC_Q = [1.0, 0.0, 0.0]   # identical to VEC_A → similarity = 1.0


def make_embedder(vectors: list[list[float]]) -> MagicMock:
    embedder = MagicMock()
    embedder.embed.return_value = vectors
    return embedder


def make_chunk(text: str, index: int = 0) -> dict:
    return {"text": text, "metadata": {"chunk_index": index}}


# ── TestDenseRetrieverInit ────────────────────────────────────────────────────

class TestDenseRetrieverInit:

    def test_stores_embedder(self):
        embedder = make_embedder([])
        retriever = DenseRetriever(embedder=embedder)
        assert retriever.embedder is embedder

    def test_chunks_empty_before_index(self):
        retriever = DenseRetriever(embedder=make_embedder([]))
        assert retriever._chunks == []
        assert retriever._chunk_vectors == []


# ── TestDenseRetrieverIndex ───────────────────────────────────────────────────

class TestDenseRetrieverIndex:

    def test_empty_chunks_raises(self):
        retriever = DenseRetriever(embedder=make_embedder([]))
        with pytest.raises(ValueError, match="chunks"):
            retriever.index([])

    def test_index_stores_chunks(self):
        chunks = [make_chunk("a"), make_chunk("b")]
        retriever = DenseRetriever(embedder=make_embedder([VEC_A, VEC_B]))
        retriever.index(chunks)
        assert retriever._chunks == chunks

    def test_index_stores_vectors(self):
        chunks = [make_chunk("a"), make_chunk("b")]
        retriever = DenseRetriever(embedder=make_embedder([VEC_A, VEC_B]))
        retriever.index(chunks)
        assert retriever._chunk_vectors == [VEC_A, VEC_B]

    def test_index_calls_embed_with_chunk_texts(self):
        chunks = [make_chunk("hello"), make_chunk("world")]
        embedder = make_embedder([VEC_A, VEC_B])
        retriever = DenseRetriever(embedder=embedder)
        retriever.index(chunks)
        embedder.embed.assert_called_once_with(["hello", "world"])


# ── TestDenseRetrieverRetrieve ────────────────────────────────────────────────

class TestDenseRetrieverRetrieve:

    def test_raises_if_not_indexed(self):
        retriever = DenseRetriever(embedder=make_embedder([]))
        with pytest.raises(RuntimeError, match="index"):
            retriever.retrieve("query", top_k=3)

    def test_empty_query_raises(self):
        chunks = [make_chunk("a")]
        embedder = MagicMock()
        embedder.embed.side_effect = [[VEC_A], [VEC_Q]]
        retriever = DenseRetriever(embedder=embedder)
        retriever.index(chunks)
        with pytest.raises(ValueError, match="query"):
            retriever.retrieve("", top_k=1)

    def test_returns_top_k_results(self):
        chunks = [make_chunk("a"), make_chunk("b"), make_chunk("c")]
        embedder = MagicMock()
        embedder.embed.side_effect = [[VEC_A, VEC_B, VEC_C], [VEC_Q]]
        retriever = DenseRetriever(embedder=embedder)
        retriever.index(chunks)
        results = retriever.retrieve("query", top_k=2)
        assert len(results) == 2

    def test_returns_fewer_if_not_enough_chunks(self):
        chunks = [make_chunk("a")]
        embedder = MagicMock()
        embedder.embed.side_effect = [[VEC_A], [VEC_Q]]
        retriever = DenseRetriever(embedder=embedder)
        retriever.index(chunks)
        results = retriever.retrieve("query", top_k=5)
        assert len(results) == 1

    def test_results_sorted_by_score_descending(self):
        chunks = [make_chunk("b"), make_chunk("a"), make_chunk("c")]
        embedder = MagicMock()
        embedder.embed.side_effect = [[VEC_B, VEC_A, VEC_C], [VEC_Q]]
        retriever = DenseRetriever(embedder=embedder)
        retriever.index(chunks)
        results = retriever.retrieve("query", top_k=3)
        scores = [r["metadata"]["semantic_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_best_chunk_is_first(self):
        chunks = [make_chunk("b"), make_chunk("a"), make_chunk("c")]
        embedder = MagicMock()
        embedder.embed.side_effect = [[VEC_B, VEC_A, VEC_C], [VEC_Q]]
        retriever = DenseRetriever(embedder=embedder)
        retriever.index(chunks)
        results = retriever.retrieve("query", top_k=3)
        assert results[0]["text"] == "a"

    def test_semantic_score_in_metadata(self):
        chunks = [make_chunk("a")]
        embedder = MagicMock()
        embedder.embed.side_effect = [[VEC_A], [VEC_Q]]
        retriever = DenseRetriever(embedder=embedder)
        retriever.index(chunks)
        results = retriever.retrieve("query", top_k=1)
        assert "semantic_score" in results[0]["metadata"]

    def test_semantic_score_between_zero_and_one(self):
        chunks = [make_chunk("a"), make_chunk("b")]
        embedder = MagicMock()
        embedder.embed.side_effect = [[VEC_A, VEC_B], [VEC_Q]]
        retriever = DenseRetriever(embedder=embedder)
        retriever.index(chunks)
        results = retriever.retrieve("query", top_k=2)
        for r in results:
            assert 0.0 <= r["metadata"]["semantic_score"] <= 1.0

    def test_identical_vectors_score_one(self):
        chunks = [make_chunk("a")]
        embedder = MagicMock()
        embedder.embed.side_effect = [[VEC_A], [VEC_Q]]
        retriever = DenseRetriever(embedder=embedder)
        retriever.index(chunks)
        results = retriever.retrieve("query", top_k=1)
        assert results[0]["metadata"]["semantic_score"] == 1.0

    def test_original_metadata_preserved(self):
        chunks = [make_chunk("a", index=42)]
        embedder = MagicMock()
        embedder.embed.side_effect = [[VEC_A], [VEC_Q]]
        retriever = DenseRetriever(embedder=embedder)
        retriever.index(chunks)
        results = retriever.retrieve("query", top_k=1)
        assert results[0]["metadata"]["chunk_index"] == 42

    def test_returned_chunks_are_deep_copies(self):
        chunks = [make_chunk("a")]
        embedder = MagicMock()
        embedder.embed.side_effect = [[VEC_A], [VEC_Q]]
        retriever = DenseRetriever(embedder=embedder)
        retriever.index(chunks)
        results = retriever.retrieve("query", top_k=1)
        results[0]["metadata"]["semantic_score"] = 999
        embedder.embed.side_effect = [[VEC_Q]]
        results2 = retriever.retrieve("query", top_k=1)
        assert results2[0]["metadata"]["semantic_score"] != 999


# ── TestCosimeSimilarity ──────────────────────────────────────────────────────

class TestCosimeSimilarity:

    def test_identical_vectors_return_one(self):
        assert DenseRetriever._cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_orthogonal_vectors_return_zero(self):
        assert DenseRetriever._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_zero_vector_returns_zero(self):
        assert DenseRetriever._cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_opposite_vectors_clamped_to_zero(self):
        result = DenseRetriever._cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert result == 0.0

    def test_result_always_non_negative(self):
        import random
        random.seed(42)
        for _ in range(20):
            a = [random.uniform(-1, 1) for _ in range(5)]
            b = [random.uniform(-1, 1) for _ in range(5)]
            assert DenseRetriever._cosine_similarity(a, b) >= 0.0
