import math
from copy import deepcopy
from typing import Any

from chunk_norris.embeddings.base import BaseEmbedder


class Retriever:
    """
    Retrieves the most relevant chunks for a given question using
    semantic similarity between embeddings.

    The retriever embeds all chunks once and caches the vectors. This means
    if you ask multiple questions against the same set of chunks, the chunks
    are only embedded once — not once per question.

    Args:
        embedder (BaseEmbedder): Any embedder implementing BaseEmbedder.
                                 e.g. BertEmbedder().
        top_k (int): Number of most relevant chunks to return per question.
                     Default: 3.

    Example::

        from chunk_norris.embeddings.bert import BertEmbedder
        from chunk_norris.evaluator.retriever import Retriever
        from chunk_norris.chunkers.fixed import FixedChunker

        chunker = FixedChunker(chunk_size=256, overlap=0.1)
        chunks = chunker.chunk("Your document text here...")

        retriever = Retriever(embedder=BertEmbedder(), top_k=3)
        retriever.index(chunks)

        results = retriever.retrieve("What is the refund policy?")
        for chunk in results:
            print(chunk["metadata"]["score"])
            print(chunk["text"])
    """

    def __init__(self, embedder: BaseEmbedder, top_k: int = 3) -> None:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        self.embedder = embedder
        self.top_k = top_k

        # Set by index() — not available until chunks are indexed
        self._chunks: list[dict[str, Any]] = []
        self._chunk_vectors: list[list[float]] = []

    def index(self, chunks: list[dict[str, Any]]) -> None:
        """
        Embeds and caches all chunks so they are ready for retrieval.

        Call this once per set of chunks before calling retrieve().
        If you change the chunks (e.g. different chunker or config),
        call index() again to refresh the cache.

        Args:
            chunks (list[dict]): The output from any chunker's chunk() method.

        Raises:
            ValueError: If chunks is empty.
            EmbeddingError: If embedding the chunks fails.
        """
        if not chunks:
            raise ValueError("chunks must not be empty.")

        texts = [chunk["text"] for chunk in chunks]
        self._chunks = chunks
        self._chunk_vectors = self.embedder.embed(texts)

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        """
        Returns the top_k most relevant chunks for the given question.

        Embeds the question, computes cosine similarity against all indexed
        chunk vectors, and returns the top_k highest scoring chunks.

        A "score" field is added to each returned chunk's metadata indicating
        how semantically similar the chunk is to the question (0.0 to 1.0).

        Args:
            question (str): The question to retrieve relevant chunks for.

        Returns:
            list[dict]: The top_k most relevant chunks, sorted by score
                        descending. Each chunk is a deep copy of the original
                        with a "score" key added to its metadata.

        Raises:
            RuntimeError: If index() has not been called yet.
            ValueError: If question is empty.
            EmbeddingError: If embedding the question fails.
        """
        if not self._chunks:
            raise RuntimeError(
                "No chunks have been indexed. Call index(chunks) before retrieve()."
            )
        if not question or not question.strip():
            raise ValueError("question must not be empty.")

        question_vector = self.embedder.embed([question])[0]

        scored = [
            (self._cosine_similarity(question_vector, chunk_vector), i)
            for i, chunk_vector in enumerate(self._chunk_vectors)
        ]

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self.top_k]

        results = []
        for score, i in top:
            chunk = deepcopy(self._chunks[i])
            chunk["metadata"]["score"] = round(score, 4)
            results.append(chunk)

        return results

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """
        Computes the cosine similarity between two vectors.

        Cosine similarity measures the angle between two vectors:
            1.0  = identical direction (highly similar)
            0.0  = perpendicular (unrelated)
           -1.0  = opposite direction (very dissimilar)

        For sentence embeddings the result is typically between 0.0 and 1.0.

        Args:
            vec_a (list[float]): First vector.
            vec_b (list[float]): Second vector.

        Returns:
            float: Cosine similarity score clamped to [0.0, 1.0].
        """
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        magnitude_a = math.sqrt(sum(a * a for a in vec_a))
        magnitude_b = math.sqrt(sum(b * b for b in vec_b))

        if magnitude_a == 0.0 or magnitude_b == 0.0:
            return 0.0

        return max(0.0, min(1.0, dot_product / (magnitude_a * magnitude_b)))
