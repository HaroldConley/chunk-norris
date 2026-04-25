import pytest
from typing import Any

from chunk_norris.retrieval.base import BaseRetriever


# ── Minimal concrete implementation for testing ───────────────────────────────

class MinimalRetriever(BaseRetriever):
    """Minimal concrete retriever for testing the BaseRetriever contract."""

    def __init__(self) -> None:
        self._chunks: list[dict[str, Any]] = []
        self._indexed = False

    def index(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            raise ValueError("chunks must not be empty.")
        self._chunks = chunks
        self._indexed = True

    def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if not self._indexed:
            raise RuntimeError("Call index() before retrieve().")
        if not query or not query.strip():
            raise ValueError("query must not be empty.")
        return self._chunks[:top_k]


# ── TestBaseRetrieverContract ─────────────────────────────────────────────────

class TestBaseRetrieverContract:

    def test_cannot_instantiate_base_directly(self):
        with pytest.raises(TypeError):
            BaseRetriever()

    def test_subclass_without_index_raises(self):
        class IncompleteRetriever(BaseRetriever):
            def retrieve(self, query: str, top_k: int):
                pass
        with pytest.raises(TypeError):
            IncompleteRetriever()

    def test_subclass_without_retrieve_raises(self):
        class IncompleteRetriever(BaseRetriever):
            def index(self, chunks):
                pass
        with pytest.raises(TypeError):
            IncompleteRetriever()

    def test_concrete_subclass_can_be_instantiated(self):
        retriever = MinimalRetriever()
        assert retriever is not None

    def test_index_stores_chunks(self):
        retriever = MinimalRetriever()
        chunks = [{"text": "hello", "metadata": {}}]
        retriever.index(chunks)
        assert retriever._chunks == chunks

    def test_retrieve_before_index_raises(self):
        retriever = MinimalRetriever()
        with pytest.raises(RuntimeError):
            retriever.retrieve("query", top_k=3)

    def test_retrieve_returns_list(self):
        retriever = MinimalRetriever()
        chunks = [{"text": "hello", "metadata": {}}]
        retriever.index(chunks)
        result = retriever.retrieve("query", top_k=1)
        assert isinstance(result, list)

    def test_retrieve_respects_top_k(self):
        retriever = MinimalRetriever()
        chunks = [
            {"text": f"chunk {i}", "metadata": {}}
            for i in range(5)
        ]
        retriever.index(chunks)
        result = retriever.retrieve("query", top_k=3)
        assert len(result) == 3

    def test_empty_chunks_raises(self):
        retriever = MinimalRetriever()
        with pytest.raises(ValueError, match="chunks"):
            retriever.index([])

    def test_empty_query_raises(self):
        retriever = MinimalRetriever()
        chunks = [{"text": "hello", "metadata": {}}]
        retriever.index(chunks)
        with pytest.raises(ValueError, match="query"):
            retriever.retrieve("", top_k=1)
