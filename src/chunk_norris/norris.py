from typing import Any

import tiktoken

from chunk_norris.chunkers.base import BaseChunker
from chunk_norris.embeddings.base import BaseEmbedder
from chunk_norris.evaluator.metrics import Metrics
from chunk_norris.evaluator.report import Report
from chunk_norris.evaluator.retriever import Retriever

# Used only for the token count in the progress header.
_ENCODING = tiktoken.get_encoding("cl100k_base")


class Norris:
    """
    The main entry point for chunk-norris.

    Orchestrates the full evaluation loop: chunking, retrieval, scoring,
    and reporting. No LLM is required — scoring uses two complementary
    deterministic metrics:

        - Token recall: does the chunk contain the answer's key tokens?
          (measures completeness)
        - Bert score: is the chunk semantically focused on the answer?
          (measures focus / penalises noise)

    The combined score is the primary ranking metric.

    Note: LLM-based answer generation will be added in a future version
    as an optional step.

    Args:
        embedder (BaseEmbedder): Any embedder implementing BaseEmbedder.
                                 Used for both retrieval and bert scoring.
        top_k (int): Number of chunks to retrieve per question. Default: 3.
        recall_threshold (float): Minimum token recall for a chunk to be
            considered relevant. A chunk is relevant if at least this fraction
            of the expected answer's tokens appear in it.

            Recommended values:
              0.50  loose — useful when answers may be paraphrased
              0.75  default — chunk must contain 75% of answer tokens
              0.90  strict — chunk must contain almost all answer tokens
              1.00  exact — every answer token must be present

            If n_relevant is always 0, lower the threshold.
            If n_relevant always equals top_k, raise the threshold.
            Default: 0.75.

    Example::

        from chunk_norris import Norris
        from chunk_norris.chunkers.fixed import FixedChunker
        from chunk_norris.embeddings.bert import BertEmbedder

        norris = Norris(embedder=BertEmbedder())

        report = norris.run(
            text="Your document text here...",
            chunkers=[
                FixedChunker(chunk_size=128, overlap=0.1),
                FixedChunker(chunk_size=256, overlap=0.1),
                FixedChunker(chunk_size=256, overlap=0.25),
            ],
            questions=[
                {
                    "question": "What is the refund policy?",
                    "expected_answer": "Customers can request a refund within 30 days."
                },
            ]
        )

        report.compare()
        report.best()
        report.to_excel("results.xlsx")
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        top_k: int = 3,
        recall_threshold: float = 0.75,
    ) -> None:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if not (0.0 < recall_threshold <= 1.0):
            raise ValueError(
                f"recall_threshold must be between 0.0 (exclusive) and 1.0 (inclusive), "
                f"got {recall_threshold}"
            )

        self.embedder = embedder
        self.top_k = top_k
        self.recall_threshold = recall_threshold

    def run(
        self,
        text: str,
        chunkers: list[BaseChunker],
        questions: list[dict[str, Any]],
    ) -> Report:
        """
        Runs the full evaluation loop for each chunker configuration.

        For each chunker:
            1. Chunks the text.
            2. Indexes the chunks in the retriever.
            3. Retrieves top_k chunks per question.
            4. Scores each retrieved chunk against the expected answer
               using token recall and bert score.

        All configurations are then wrapped in a Report for comparison.

        Args:
            text (str): The document text to chunk and evaluate against.
            chunkers (list[BaseChunker]): One or more chunker configurations
                                          to compare.
            questions (list[dict]): A list of question dicts, each with:
                - "question" (str): The question to retrieve chunks for.
                - "expected_answer" (str): The ground truth answer for scoring.

        Returns:
            Report: A report containing scores for all configurations,
                    ready to compare, inspect, or export.

        Raises:
            ValueError: If text, chunkers, or questions are empty.
            EmbeddingError: If any embedding call fails.
        """
        if not text or not text.strip():
            raise ValueError("text must not be empty.")
        if not chunkers:
            raise ValueError("chunkers must not be empty.")
        if not questions:
            raise ValueError("questions must not be empty.")

        token_count = len(_ENCODING.encode(text))
        n_questions = len(questions)
        n_chunkers = len(chunkers)

        print("chunk-norris starting...")
        print(
            f"Document: {token_count:,} tokens | "
            f"{n_questions} question{'s' if n_questions != 1 else ''} | "
            f"{n_chunkers} chunker configuration{'s' if n_chunkers != 1 else ''}"
        )
        print()

        metrics = Metrics(
            embedder=self.embedder,
            recall_threshold=self.recall_threshold,
        )
        experiments = []

        for i, chunker in enumerate(chunkers, start=1):
            label = repr(chunker)
            print(f"[{i}/{n_chunkers}] {label}")

            print("      Chunking...   ", end="", flush=True)
            chunks = chunker.chunk(text)
            print(f"{len(chunks)} chunks")

            print("      Indexing...   ", end="", flush=True)
            retriever = Retriever(embedder=self.embedder, top_k=self.top_k)
            retriever.index(chunks)
            print("done")

            print(f"      Scoring...   ", end="", flush=True)
            results = []
            for j, question in enumerate(questions, start=1):
                retrieved_chunks = retriever.retrieve(question["question"])
                result = {
                    "question":        question["question"],
                    "expected_answer": question["expected_answer"],
                    "retrieved_chunks": retrieved_chunks,
                }
                scored = metrics.score([result])
                results.extend(scored)
                print(f"\r      Scoring...   {j}/{n_questions}", end="", flush=True)
            print()
            print()

            experiments.append({
                "chunker":        label,
                "chunker_object": chunker,
                "results":        results,
            })

        print("Done. Results ready.")
        print()

        return Report(experiments=experiments)
