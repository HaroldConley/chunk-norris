import math
import re
from typing import Any

from chunk_norris.embeddings.base import BaseEmbedder


# ── Recall threshold guidance ─────────────────────────────────────────────────
#
# The recall_threshold controls which retrieved chunks are considered relevant.
# A chunk is relevant if at least this fraction of the expected answer's tokens
# appear in the chunk text.
#
# Unlike bert score thresholds, this value is directly interpretable:
#   0.75 means "the chunk must contain at least 75% of the answer's words"
#
# Recommended values:
#
#   0.50  Loose — useful when answers are long and paraphrased, or when
#                 the document uses different wording than the expected answer.
#                 Use when you expect partial matches to be meaningful.
#
#   0.75  Default — a chunk must contain three quarters of the answer's key
#                   tokens. Balances strictness with tolerance for minor
#                   wording differences. Good starting point for most docs.
#
#   0.90  Strict — chunk must contain almost all answer tokens verbatim.
#                  Use when expected answers are copied directly from the
#                  document and wording is consistent.
#
#   1.00  Exact — every answer token must appear in the chunk. Too strict
#                 for most use cases — any minor wording difference in the
#                 expected answer will cause the chunk to fail.
#
# Practical advice:
#   - Start with 0.75 and run your experiment.
#   - If n_relevant is always 0, lower the threshold — your expected answers
#     may use different wording than the document.
#   - If n_relevant always equals top_k, raise the threshold — your chunks
#     are too large and everything passes trivially.
#
DEFAULT_RECALL_THRESHOLD = 0.75


def _tokenize(text: str) -> set[str]:
    """
    Lowercases and splits text into word tokens, removing punctuation.
    Returns a set for O(1) membership checks.
    """
    return set(re.findall(r"\b\w+\b", text.lower()))


class Metrics:
    """
    Scores RAG retrieval quality using two complementary deterministic metrics:

        1. Token recall — measures COMPLETENESS
           "Does the chunk contain the answer's key tokens?"
           Catches cases where the chunk has the right information but
           bert score is diluted by chunk length (the needle-in-haystack problem).

        2. Bert score — measures FOCUS
           "Is the chunk semantically focused on the answer topic?"
           Catches cases where a large chunk accidentally contains the answer
           words but is mostly about something else (the noise problem).

        3. Combined score — the primary metric
           Average of token recall and bert score. Rewards chunks that are
           both complete AND focused. Penalises the two extremes:
             - Full document in one chunk: recall=1.0, bert=low → combined=medium
             - Wrong chunk: recall=low, bert=low → combined=low

    Relevance filtering uses token recall against recall_threshold.
    A chunk is considered relevant if it contains at least recall_threshold
    fraction of the expected answer's tokens. This is more interpretable than
    a bert-based threshold — 0.75 means "75% of answer words must be present".

    Args:
        embedder (BaseEmbedder): Any embedder implementing BaseEmbedder.
                                 The same instance used in the retriever
                                 can be reused — no extra model loading.
        recall_threshold (float): Minimum token recall for a chunk to be
                                  considered relevant. Between 0.0 and 1.0.
                                  See threshold guidance above.
                                  Default: 0.75.

    Example::

        from chunk_norris.embeddings.bert import BertEmbedder
        from chunk_norris.evaluator.metrics import Metrics

        metrics = Metrics(embedder=BertEmbedder(), recall_threshold=0.75)
        scored_results = metrics.score(pipeline_results)

        for result in scored_results:
            print(result["scores"]["best_combined"])
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        recall_threshold: float = DEFAULT_RECALL_THRESHOLD,
    ) -> None:
        if not (0.0 < recall_threshold <= 1.0):
            raise ValueError(
                f"recall_threshold must be between 0.0 (exclusive) and 1.0 (inclusive), "
                f"got {recall_threshold}"
            )
        self.embedder = embedder
        self.recall_threshold = recall_threshold

    def score(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Scores each result by comparing the expected answer against each
        retrieved chunk individually using both token recall and bert score.

        Args:
            results (list[dict]): The output from the retrieval step, each with:
                - "question" (str)
                - "expected_answer" (str)
                - "retrieved_chunks" (list[dict])

        Returns:
            list[dict]: Same results with a "scores" key added to each, containing:

                Per-chunk raw scores:
                - "bert_scores"     (list[float]): semantic similarity per chunk
                - "token_recalls"   (list[float]): answer token coverage per chunk
                - "combined_scores" (list[float]): average of bert + recall per chunk

                Summary scores:
                - "best_bert"       (float): highest bert score across chunks
                - "best_recall"     (float): highest token recall across chunks
                - "best_combined"   (float): highest combined score — primary metric

                Relevance filtering (based on token recall >= recall_threshold):
                - "relevant_combined" (list[float]): combined scores of relevant chunks
                - "avg_relevant"      (float): average combined score of relevant chunks
                - "n_relevant"        (int): number of relevant chunks
                - "n_retrieved"       (int): total chunks retrieved (= top_k)
                - "recall_threshold"  (float): threshold used, for traceability

        Raises:
            ValueError: If results is empty.
            EmbeddingError: If any embedding call fails.
        """
        if not results:
            raise ValueError("results must not be empty.")

        scored = []
        for result in results:
            expected = result["expected_answer"]
            chunks = result["retrieved_chunks"]

            bert_scores   = self._compute_bert_scores(expected, chunks)
            token_recalls = self._compute_token_recalls(expected, chunks)
            combined_scores = [
                round((b + r) / 2, 4)
                for b, r in zip(bert_scores, token_recalls)
            ]

            relevant_combined = [
                combined_scores[i]
                for i, r in enumerate(token_recalls)
                if r >= self.recall_threshold
            ]

            best_bert     = max(bert_scores)     if bert_scores     else 0.0
            best_recall   = max(token_recalls)   if token_recalls   else 0.0
            best_combined = max(combined_scores) if combined_scores else 0.0
            avg_relevant  = (
                round(sum(relevant_combined) / len(relevant_combined), 4)
                if relevant_combined else 0.0
            )

            scored_result = {**result}
            scored_result["scores"] = {
                # per-chunk raw scores
                "bert_scores":        [round(s, 4) for s in bert_scores],
                "token_recalls":      [round(s, 4) for s in token_recalls],
                "combined_scores":    combined_scores,
                # summary
                "best_bert":          round(best_bert, 4),
                "best_recall":        round(best_recall, 4),
                "best_combined":      round(best_combined, 4),
                # relevance filtering
                "relevant_combined":  [round(s, 4) for s in relevant_combined],
                "avg_relevant":       avg_relevant,
                "n_relevant":         len(relevant_combined),
                "n_retrieved":        len(chunks),
                "recall_threshold":   self.recall_threshold,
            }
            scored.append(scored_result)

        return scored

    def _compute_bert_scores(
        self,
        expected_answer: str,
        chunks: list[dict[str, Any]],
    ) -> list[float]:
        """
        Computes cosine similarity between the expected answer and each chunk.
        All texts are embedded in a single batched call for efficiency.
        """
        if not expected_answer.strip() or not chunks:
            return [0.0] * len(chunks)

        texts = [expected_answer] + [chunk["text"] for chunk in chunks]
        vectors = self.embedder.embed(texts)
        expected_vector = vectors[0]

        return [
            self._cosine_similarity(expected_vector, v)
            for v in vectors[1:]
        ]

    def _compute_token_recalls(
        self,
        expected_answer: str,
        chunks: list[dict[str, Any]],
    ) -> list[float]:
        """
        Computes token recall for each chunk — the fraction of the expected
        answer's tokens that appear in the chunk text.

        Token recall = |answer_tokens ∩ chunk_tokens| / |answer_tokens|

        Uses lowercased word tokens with punctuation removed. If the expected
        answer has no tokens, returns 0.0 for all chunks.
        """
        if not expected_answer.strip() or not chunks:
            return [0.0] * len(chunks)

        answer_tokens = _tokenize(expected_answer)
        if not answer_tokens:
            return [0.0] * len(chunks)

        recalls = []
        for chunk in chunks:
            chunk_tokens = _tokenize(chunk["text"])
            overlap = answer_tokens & chunk_tokens
            recalls.append(len(overlap) / len(answer_tokens))

        return recalls

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Computes cosine similarity clamped to [0.0, 1.0]."""
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = math.sqrt(sum(a * a for a in vec_a))
        mag_b = math.sqrt(sum(b * b for b in vec_b))
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (mag_a * mag_b)))
