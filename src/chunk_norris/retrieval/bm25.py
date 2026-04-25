from copy import deepcopy
from typing import Any

from rank_bm25 import BM25Okapi  # rank-bm25

from chunk_norris.retrieval.base import BaseRetriever


def _tokenize(text: str) -> list[str]:
    """
    Lowercases and splits text into word tokens.
    Simple whitespace-based tokenisation — consistent with token recall
    in the metrics module, which also uses word-level matching.
    """
    return text.lower().split()


class BM25Retriever(BaseRetriever):
    """
    Retrieves chunks using BM25 keyword scoring.

    BM25 (Best Match 25) is a probabilistic ranking function that scores
    chunks based on keyword overlap with the query, adjusted for chunk
    length and term frequency. It excels at finding exact keyword matches
    but cannot handle vocabulary mismatch (synonyms, paraphrasing).

    This is the keyword component of hybrid retrieval — it complements
    semantic search by catching cases where the user's query uses the
    same words as the document.

    Technical note: uses the BM25Okapi variant with default parameters
    (k1=1.5, b=0.75), which are the standard values used in production
    search systems.

    User-facing score: keyword_score (0.0 to 1.0, normalised).

    Args:
        None — BM25 requires no model or API key.

    Example::

        from chunk_norris.retrieval.bm25 import BM25Retriever

        retriever = BM25Retriever()
        retriever.index(chunks)
        results = retriever.retrieve("What is the refund policy?", top_k=3)
    """

    def __init__(self) -> None:
        self._chunks: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None

    def index(self, chunks: list[dict[str, Any]]) -> None:
        """
        Builds the BM25 index from a list of chunks.

        Tokenises each chunk's text and builds a BM25Okapi index.
        This is fast and requires no embeddings or model loading.

        Args:
            chunks (list[dict]): Chunks to index. Each must have a 'text' key.

        Raises:
            ValueError: If chunks is empty.
        """
        if not chunks:
            raise ValueError("chunks must not be empty.")

        self._chunks = chunks
        tokenised = [_tokenize(chunk["text"]) for chunk in chunks]
        self._bm25 = BM25Okapi(tokenised)

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """
        Returns the top_k chunks with the highest BM25 keyword score.

        Scores are normalised to [0.0, 1.0] by dividing by the maximum
        score in the result set. If all scores are zero (no keyword
        overlap between query and any chunk), all chunks score 0.0.

        Adds a 'keyword_score' key to each returned chunk's metadata.

        Technical note: BM25Okapi scores are not inherently bounded —
        normalisation makes them comparable to semantic_score and
        interpretable as a relevance signal (0.0 = no keyword match,
        1.0 = best keyword match in the index).

        Args:
            query (str): The search query.
            top_k (int): Number of chunks to return.

        Returns:
            list[dict]: top_k chunks sorted by keyword_score descending.

        Raises:
            RuntimeError: If called before index().
            ValueError: If query is empty.
        """
        if self._bm25 is None or not self._chunks:
            raise RuntimeError(
                "No chunks indexed. Call index(chunks) before retrieve()."
            )
        if not query or not query.strip():
            raise ValueError("query must not be empty.")

        query_tokens = _tokenize(query)
        raw_scores = self._bm25.get_scores(query_tokens)

        # Normalise to [0.0, 1.0]
        max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0
        normalised = [s / max_score for s in raw_scores]

        # Rank by score descending, take top_k
        ranked = sorted(
            enumerate(normalised),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results = []
        for i, score in ranked:
            chunk = deepcopy(self._chunks[i])
            chunk["metadata"]["keyword_score"] = round(score, 4)
            results.append(chunk)

        return results
