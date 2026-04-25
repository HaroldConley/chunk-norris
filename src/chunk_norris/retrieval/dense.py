import math
from copy import deepcopy
from typing import Any

from chunk_norris.embeddings.base import BaseEmbedder
from chunk_norris.retrieval.base import BaseRetriever


class DenseRetriever(BaseRetriever):
    """
    Retrieves chunks using dense semantic similarity.

    Embeds all chunks once at index time, then embeds the query at
    retrieval time and computes cosine similarity between the query
    vector and all chunk vectors.

    This is the semantic component of hybrid retrieval — it finds
    chunks that are semantically related to the query, even when
    they don't share exact keywords.

    Args:
        embedder (BaseEmbedder): Any embedder implementing BaseEmbedder.

    Example::

        from chunk_norris.embeddings.bert import BertEmbedder
        from chunk_norris.retrieval.dense import DenseRetriever

        retriever = DenseRetriever(embedder=BertEmbedder())
        retriever.index(chunks)
        results = retriever.retrieve("What is the refund policy?", top_k=3)
    """

    def __init__(self, embedder: BaseEmbedder) -> None:
        self.embedder = embedder
        self._chunks: list[dict[str, Any]] = []
        self._chunk_vectors: list[list[float]] = []

    def index(self, chunks: list[dict[str, Any]]) -> None:
        """
        Embeds and caches all chunks for semantic retrieval.

        Args:
            chunks (list[dict]): Chunks to index. Each must have a 'text' key.

        Raises:
            ValueError: If chunks is empty.
            EmbeddingError: If embedding fails.
        """
        if not chunks:
            raise ValueError("chunks must not be empty.")

        self._chunks = chunks
        self._chunk_vectors = self.embedder.embed(
            [chunk["text"] for chunk in chunks]
        )

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """
        Returns the top_k chunks most semantically similar to the query.

        Adds a 'semantic_score' key to each returned chunk's metadata,
        containing the cosine similarity score (0.0 to 1.0).

        Technical note: this is dense retrieval — BERT embeddings compared
        via cosine similarity. Also known as vector search or semantic search.
        It handles vocabulary mismatch well (synonyms, paraphrasing) but can
        miss exact keyword matches that BM25 would catch.

        Technical note: this is dense retrieval using BERT embeddings
        and cosine similarity — the semantic component of hybrid search.

        Args:
            query (str): The search query.
            top_k (int): Number of chunks to return.

        Returns:
            list[dict]: top_k chunks sorted by semantic_score descending.

        Raises:
            RuntimeError: If called before index().
            ValueError: If query is empty.
            EmbeddingError: If embedding the query fails.
        """
        if not self._chunks:
            raise RuntimeError(
                "No chunks indexed. Call index(chunks) before retrieve()."
            )
        if not query or not query.strip():
            raise ValueError("query must not be empty.")

        query_vector = self.embedder.embed([query])[0]

        scored = [
            (self._cosine_similarity(query_vector, vec), i)
            for i, vec in enumerate(self._chunk_vectors)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, i in scored[:top_k]:
            chunk = deepcopy(self._chunks[i])
            chunk["metadata"]["semantic_score"] = round(score, 4)
            results.append(chunk)

        return results

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Cosine similarity clamped to [0.0, 1.0]."""
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = math.sqrt(sum(a * a for a in vec_a))
        mag_b = math.sqrt(sum(b * b for b in vec_b))
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (mag_a * mag_b)))
