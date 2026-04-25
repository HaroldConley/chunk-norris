from abc import ABC, abstractmethod
from typing import Any


class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval strategies.

    All retrievers follow the same two-step interface:
        1. index(chunks)    — build the search index from chunks
        2. retrieve(query)  — find the most relevant chunks for a query

    This interface allows chunk-norris to swap retrieval strategies
    transparently — the evaluation loop never needs to know whether
    it's using dense, BM25, or hybrid retrieval.

    Implementing a custom retriever:

        from chunk_norris.retrieval.base import BaseRetriever

        class MyRetriever(BaseRetriever):

            def index(self, chunks: list[dict[str, Any]]) -> None:
                # build your index here
                self._chunks = chunks

            def retrieve(
                self,
                query: str,
                top_k: int,
            ) -> list[dict[str, Any]]:
                # return top_k most relevant chunks
                return self._chunks[:top_k]
    """

    @abstractmethod
    def index(self, chunks: list[dict[str, Any]]) -> None:
        """
        Builds the search index from a list of chunks.

        Must be called before retrieve(). Calling retrieve() before
        index() should raise RuntimeError.

        Args:
            chunks (list[dict]): The chunks to index. Each chunk is a dict
                                 with at least a 'text' key.

        Raises:
            ValueError: If chunks is empty.
        """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """
        Retrieves the most relevant chunks for a query.

        Args:
            query (str): The search query.
            top_k (int): Number of chunks to return.

        Returns:
            list[dict]: The top_k most relevant chunks, sorted by relevance
                        descending. Each chunk includes a 'score' key in its
                        metadata indicating retrieval relevance.

        Raises:
            RuntimeError: If called before index().
            ValueError: If query is empty.
        """
