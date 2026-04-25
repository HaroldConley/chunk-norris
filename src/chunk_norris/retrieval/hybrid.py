from copy import deepcopy
from typing import Any

from chunk_norris.embeddings.base import BaseEmbedder
from chunk_norris.retrieval.base import BaseRetriever
from chunk_norris.retrieval.dense import DenseRetriever
from chunk_norris.retrieval.bm25 import BM25Retriever


# Standard RRF constant — used in production search systems.
# Higher values reduce the impact of high-ranking documents,
# making the fusion more robust to outliers in either ranked list.
_RRF_K = 60


class HybridRetriever(BaseRetriever):
    """
    Retrieves chunks using hybrid search — combining semantic search
    and BM25 keyword search via Reciprocal Rank Fusion (RRF).

    Hybrid search outperforms either method alone because:
        - Semantic search handles vocabulary mismatch (synonyms, paraphrasing)
        - BM25 handles exact keyword matches and technical terms
        - RRF combines both ranked lists without requiring score normalisation

    This is the default retriever in chunk-norris because it most accurately
    reflects how modern production RAG systems retrieve content.

    How RRF works:
        Each chunk receives a score from both retrievers based on its rank
        in each list, not its raw score:

            rrf_score = 1/(k + rank_semantic) + 1/(k + rank_keyword)

        where k=60 is the standard constant. A chunk that ranks well in
        both lists scores highest. A chunk that ranks well in only one list
        still scores reasonably — hybrid search never ignores a signal
        entirely.

    User-facing scores added to chunk metadata:
        - semantic_score  : cosine similarity from dense retrieval (0.0–1.0)
        - keyword_score   : normalised BM25 score (0.0–1.0)
        - score           : final RRF combined score — primary ranking signal

    Args:
        embedder (BaseEmbedder): Any embedder implementing BaseEmbedder.
                                 Used for the semantic search component.

    Example::

        from chunk_norris.embeddings.bert import BertEmbedder
        from chunk_norris.retrieval.hybrid import HybridRetriever

        retriever = HybridRetriever(embedder=BertEmbedder())
        retriever.index(chunks)
        results = retriever.retrieve("What is the refund policy?", top_k=3)

        for chunk in results:
            print(chunk["text"])
            print(chunk["metadata"]["score"])          # RRF score
            print(chunk["metadata"]["semantic_score"]) # semantic component
            print(chunk["metadata"]["keyword_score"])  # keyword component
    """

    def __init__(self, embedder: BaseEmbedder) -> None:
        self._embedder = embedder
        self._dense = DenseRetriever(embedder=embedder)
        self._bm25  = BM25Retriever()
        self._chunks: list[dict[str, Any]] = []

    def index(self, chunks: list[dict[str, Any]]) -> None:
        """
        Builds both the dense and BM25 indexes from a list of chunks.

        Args:
            chunks (list[dict]): Chunks to index. Each must have a 'text' key.

        Raises:
            ValueError: If chunks is empty.
            EmbeddingError: If embedding the chunks fails.
        """
        if not chunks:
            raise ValueError("chunks must not be empty.")

        self._chunks = chunks
        self._dense.index(chunks)
        self._bm25.index(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """
        Returns the top_k chunks ranked by RRF hybrid score.

        Retrieves ALL chunks from both dense and BM25 retrievers,
        fuses their rankings using RRF, then returns the top_k results.
        Using all chunks for fusion (not just top_k from each) ensures
        no relevant chunk is excluded before fusion.

        Args:
            query (str): The search query.
            top_k (int): Number of chunks to return.

        Returns:
            list[dict]: top_k chunks sorted by RRF score descending,
                        each with semantic_score, keyword_score, and
                        score (RRF) in metadata.

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

        n = len(self._chunks)

        # Retrieve all chunks from both retrievers for complete fusion
        dense_results  = self._dense.retrieve(query, top_k=n)
        keyword_results = self._bm25.retrieve(query, top_k=n)

        # Build lookup: chunk text → scores from each retriever
        # Using chunk text as key — chunk identity within one document
        dense_scores   = {
            r["text"]: r["metadata"]["semantic_score"]
            for r in dense_results
        }
        keyword_scores = {
            r["text"]: r["metadata"]["keyword_score"]
            for r in keyword_results
        }

        # Build rank lookup: chunk text → rank (0-based) in each list
        dense_ranks   = {r["text"]: i for i, r in enumerate(dense_results)}
        keyword_ranks = {r["text"]: i for i, r in enumerate(keyword_results)}

        # Compute RRF score for every chunk
        rrf_scores: dict[str, float] = {}
        for chunk in self._chunks:
            text = chunk["text"]
            rank_d = dense_ranks.get(text, n)    # unseen chunks ranked last
            rank_k = keyword_ranks.get(text, n)
            rrf_scores[text] = (
                1.0 / (_RRF_K + rank_d) +
                1.0 / (_RRF_K + rank_k)
            )

        # Sort all chunks by RRF score descending, take top_k
        ranked_chunks = sorted(
            self._chunks,
            key=lambda c: rrf_scores[c["text"]],
            reverse=True,
        )[:top_k]

        # Assemble results with all three scores in metadata
        results = []
        for chunk in ranked_chunks:
            text = chunk["text"]
            result = deepcopy(chunk)
            result["metadata"]["semantic_score"] = dense_scores.get(text, 0.0)
            result["metadata"]["keyword_score"]  = keyword_scores.get(text, 0.0)
            result["metadata"]["score"]          = round(rrf_scores[text], 6)
            results.append(result)

        return results
