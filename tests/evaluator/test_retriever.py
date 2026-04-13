import pytest
from unittest.mock import MagicMock

from chunk_norris.evaluator.retriever import Retriever


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_embedder(vectors: list[list[float]]) -> MagicMock:
    """Returns a mock embedder that returns vectors in order."""
    embedder = MagicMock()
    embedder.embed.return_value = vectors
    return embedder


def make_chunk(text: str, index: int = 0) -> dict:
    """Returns a minimal chunk dict."""
    return {"text": text, "metadata": {"chunk_index": index}}


# Three orthogonal unit vectors — easy to reason about cosine similarity
VEC_A = [1.0, 0.0, 0.0]   # points along x
VEC_B = [0.0, 1.0, 0.0]   # points along y
VEC_C = [0.0, 0.0, 1.0]   # points along z
VEC_Q = [1.0, 0.0, 0.0]   # same direction as VEC_A → similarity = 1.0


# ── TestRetrieverInit ─────────────────────────────────────────────────────────

class TestRetrieverInit:

    def test_default_top_k(self):
        embedder = make_embedder([])
        retriever = Retriever(embedder=embedder)
        assert retriever.top_k == 3

    def test_custom_top_k_stored(self):
        embedder = make_embedder([])
        retriever = Retriever(embedder=embedder, top_k=5)
        assert retriever.top_k == 5

    def test_zero_top_k_raises(self):
        with pytest.raises(ValueError, match="top_k"):
            Retriever(embedder=make_embedder([]), top_k=0)

    def test_negative_top_k_raises(self):
        with pytest.raises(ValueError, match="top_k"):
            Retriever(embedder=make_embedder([]), top_k=-1)

    def test_chunks_empty_before_index(self):
        retriever = Retriever(embedder=make_embedder([]))
        assert retriever._chunks == []
        assert retriever._chunk_vectors == []


# ── TestRetrieverIndex ────────────────────────────────────────────────────────

class TestRetrieverIndex:

    def test_empty_chunks_raises(self):
        retriever = Retriever(embedder=make_embedder([]))
        with pytest.raises(ValueError, match="chunks"):
            retriever.index([])

    def test_index_stores_chunks(self):
        chunks = [make_chunk("a"), make_chunk("b")]
        embedder = make_embedder([VEC_A, VEC_B])
        retriever = Retriever(embedder=embedder)
        retriever.index(chunks)
        assert retriever._chunks == chunks

    def test_index_stores_vectors(self):
        chunks = [make_chunk("a"), make_chunk("b")]
        embedder = make_embedder([VEC_A, VEC_B])
        retriever = Retriever(embedder=embedder)
        retriever.index(chunks)
        assert retriever._chunk_vectors == [VEC_A, VEC_B]

    def test_index_calls_embed_with_chunk_texts(self):
        chunks = [make_chunk("hello"), make_chunk("world")]
        embedder = make_embedder([VEC_A, VEC_B])
        retriever = Retriever(embedder=embedder)
        retriever.index(chunks)
        embedder.embed.assert_called_once_with(["hello", "world"])

    def test_reindexing_replaces_previous_chunks(self):
        chunks_a = [make_chunk("first")]
        chunks_b = [make_chunk("second")]
        embedder = MagicMock()
        embedder.embed.side_effect = [[VEC_A], [VEC_B]]
        retriever = Retriever(embedder=embedder)
        retriever.index(chunks_a)
        retriever.index(chunks_b)
        assert retriever._chunks == chunks_b


# ── TestRetrieverRetrieve ─────────────────────────────────────────────────────

class TestRetrieverRetrieve:

    def test_raises_if_not_indexed(self):
        retriever = Retriever(embedder=make_embedder([]))
        with pytest.raises(RuntimeError, match="index"):
            retriever.retrieve("some question")

    def test_empty_question_raises(self):
        chunks = [make_chunk("a")]
        embedder = MagicMock()
        embedder.embed.side_effect = [[VEC_A], [VEC_Q]]
        retriever = Retriever(embedder=embedder)
        retriever.index(chunks)
        with pytest.raises(ValueError, match="question"):
            retriever.retrieve("")

    def test_whitespace_question_raises(self):
        chunks = [make_chunk("a")]
        embedder = MagicMock()
        embedder.embed.side_effect = [[VEC_A], [VEC_Q]]
        retriever = Retriever(embedder=embedder)
        retriever.index(chunks)
        with pytest.raises(ValueError, match="question"):
            retriever.retrieve("   ")

    def test_returns_list(self):
        chunks = [make_chunk("a"), make_chunk("b"), make_chunk("c")]
        embedder = MagicMock()
        embedder.embed.side_effect = [
            [VEC_A, VEC_B, VEC_C],
            [VEC_Q],
        ]
        retriever = Retriever(embedder=embedder, top_k=2)
        retriever.index(chunks)
        results = retriever.retrieve("question")
        assert isinstance(results, list)

    def test_returns_top_k_results(self):
        chunks = [make_chunk("a"), make_chunk("b"), make_chunk("c")]
        embedder = MagicMock()
        embedder.embed.side_effect = [
            [VEC_A, VEC_B, VEC_C],
            [VEC_Q],
        ]
        retriever = Retriever(embedder=embedder, top_k=2)
        retriever.index(chunks)
        results = retriever.retrieve("question")
        assert len(results) == 2

    def test_returns_fewer_than_top_k_if_not_enough_chunks(self):
        chunks = [make_chunk("a")]
        embedder = MagicMock()
        embedder.embed.side_effect = [[VEC_A], [VEC_Q]]
        retriever = Retriever(embedder=embedder, top_k=5)
        retriever.index(chunks)
        results = retriever.retrieve("question")
        assert len(results) == 1

    def test_results_sorted_by_score_descending(self):
        # VEC_Q is identical to VEC_A → similarity 1.0
        # VEC_Q vs VEC_B → similarity 0.0
        # VEC_Q vs VEC_C → similarity 0.0
        chunks = [make_chunk("b"), make_chunk("a"), make_chunk("c")]
        embedder = MagicMock()
        embedder.embed.side_effect = [
            [VEC_B, VEC_A, VEC_C],  # chunk vectors
            [VEC_Q],                 # question vector
        ]
        retriever = Retriever(embedder=embedder, top_k=3)
        retriever.index(chunks)
        results = retriever.retrieve("question")
        scores = [r["metadata"]["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_best_chunk_is_first(self):
        # chunk "a" has VEC_A which is identical to VEC_Q
        chunks = [make_chunk("b"), make_chunk("a"), make_chunk("c")]
        embedder = MagicMock()
        embedder.embed.side_effect = [
            [VEC_B, VEC_A, VEC_C],
            [VEC_Q],
        ]
        retriever = Retriever(embedder=embedder, top_k=3)
        retriever.index(chunks)
        results = retriever.retrieve("question")
        assert results[0]["text"] == "a"

    def test_score_added_to_metadata(self):
        chunks = [make_chunk("a")]
        embedder = MagicMock()
        embedder.embed.side_effect = [[VEC_A], [VEC_Q]]
        retriever = Retriever(embedder=embedder, top_k=1)
        retriever.index(chunks)
        results = retriever.retrieve("question")
        assert "score" in results[0]["metadata"]

    def test_score_is_between_zero_and_one(self):
        chunks = [make_chunk("a"), make_chunk("b")]
        embedder = MagicMock()
        embedder.embed.side_effect = [
            [VEC_A, VEC_B],
            [VEC_Q],
        ]
        retriever = Retriever(embedder=embedder, top_k=2)
        retriever.index(chunks)
        results = retriever.retrieve("question")
        for result in results:
            assert 0.0 <= result["metadata"]["score"] <= 1.0

    def test_identical_vectors_score_one(self):
        chunks = [make_chunk("a")]
        embedder = MagicMock()
        embedder.embed.side_effect = [[VEC_A], [VEC_Q]]  # VEC_A == VEC_Q
        retriever = Retriever(embedder=embedder, top_k=1)
        retriever.index(chunks)
        results = retriever.retrieve("question")
        assert results[0]["metadata"]["score"] == 1.0

    def test_original_chunk_metadata_preserved(self):
        chunks = [make_chunk("a", index=42)]
        embedder = MagicMock()
        embedder.embed.side_effect = [[VEC_A], [VEC_Q]]
        retriever = Retriever(embedder=embedder, top_k=1)
        retriever.index(chunks)
        results = retriever.retrieve("question")
        # Original chunk_index should still be there alongside new score
        assert results[0]["metadata"]["chunk_index"] == 42

    def test_returned_chunks_are_deep_copies(self):
        chunks = [make_chunk("a")]
        embedder = MagicMock()
        embedder.embed.side_effect = [[VEC_A], [VEC_Q]]
        retriever = Retriever(embedder=embedder, top_k=1)
        retriever.index(chunks)
        results = retriever.retrieve("question")
        # Mutating the result should not affect the indexed chunk
        results[0]["metadata"]["score"] = 999
        retriever._chunks[0]["metadata"]["chunk_index"] = 999
        # Re-retrieve — should get fresh copy
        embedder.embed.side_effect = [[VEC_Q]]
        results2 = retriever.retrieve("question")
        assert results2[0]["metadata"]["score"] != 999


# ── TestCosineSimlarity ───────────────────────────────────────────────────────

class TestCosimeSimilarity:
    """Tests the static _cosine_similarity helper directly."""

    def test_identical_vectors_return_one(self):
        assert Retriever._cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_orthogonal_vectors_return_zero(self):
        assert Retriever._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_zero_vector_a_returns_zero(self):
        assert Retriever._cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_zero_vector_b_returns_zero(self):
        assert Retriever._cosine_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0

    def test_result_clamped_to_zero(self):
        # Opposite vectors should return 0.0 (clamped from -1.0)
        result = Retriever._cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert result == 0.0

    def test_result_clamped_to_one(self):
        result = Retriever._cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert result <= 1.0

    def test_result_always_non_negative(self):
        import random
        random.seed(42)
        for _ in range(20):
            a = [random.uniform(-1, 1) for _ in range(5)]
            b = [random.uniform(-1, 1) for _ in range(5)]
            result = Retriever._cosine_similarity(a, b)
            assert result >= 0.0
