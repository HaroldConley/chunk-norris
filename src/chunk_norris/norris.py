from typing import Any

import tiktoken

from chunk_norris.chunkers.base import BaseChunker
from chunk_norris.embeddings.base import BaseEmbedder
from chunk_norris.evaluator.metrics import Metrics
from chunk_norris.evaluator.report import Report
from chunk_norris.evaluator.retriever import Retriever
from chunk_norris.llm.base import BaseLLM
from chunk_norris.question_gen import QuestionGenerator

# Used only for the token count in the progress header.
_ENCODING = tiktoken.get_encoding("cl100k_base")


class Norris:
    """
    The main entry point for chunk-norris.

    Orchestrates the full evaluation loop: chunking, retrieval, scoring,
    and reporting. No LLM is required for evaluation — scoring uses two
    complementary deterministic metrics:

        - Answer span coverage (token recall): does the chunk contain
          the answer's key tokens? (measures completeness)
        - Semantic focus (bert score): is the chunk semantically focused
          on the answer? (measures focus / penalises noise)

    An LLM is optionally used for automatic question generation via
    generate_questions() — it is never used during evaluation.

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

        from chunk_norris import Norris, BertEmbedder
        from chunk_norris.chunkers.fixed import FixedChunker
        from chunk_norris.llm.openai_llm import OpenAILLM

        norris = Norris(embedder=BertEmbedder())

        # Option A — auto-generate questions
        questions = norris.generate_questions(
            text=TEXT,
            llm=OpenAILLM(model="gpt-4o-mini-2024-07-18"),
            n=20,
        )

        # Option B — provide your own questions
        questions = [
            {
                "question": "What is the refund policy?",
                "expected_answer": "Customers can request a refund within 30 days."
            },
        ]

        report = norris.run(
            text=TEXT,
            chunkers=[
                FixedChunker(chunk_size=128, overlap=0.1),
                FixedChunker(chunk_size=256, overlap=0.1),
            ],
            questions=questions,
        )

        report.compare()
        report.best_chunker().chunk(TEXT)
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

    def generate_questions(
        self,
        text: str,
        llm: BaseLLM,
        n: int = 20,
        seed: int | None = None,
    ) -> list[dict[str, str]]:
        """
        Generates question-answer pairs from the document using an LLM.

        Uses passage-based generation to guarantee location diversity
        and single-chunk answerability. Each question is grounded in
        exactly one passage, and answers are copied verbatim from the
        passage to maximise token recall accuracy during evaluation.

        Inspect the generated questions before running the evaluation —
        you can filter, edit, or add to them as needed.

        Args:
            text (str): The document text to generate questions from.
            llm (BaseLLM): Any LLM implementing BaseLLM. Recommended:
                           OpenAILLM(model="gpt-4o-mini-2024-07-18").
            n (int): Target number of question-answer pairs. Default: 20.
            seed (int | None): Random seed for reproducible passage sampling.
                               Default: None (random).

        Returns:
            list[dict]: A list of question dicts compatible with run():
                - "question" (str): The generated question.
                - "expected_answer" (str): Verbatim answer from the passage.

        Raises:
            ValueError: If text is empty or n is not positive.
            LLMError: If an LLM call fails.

        Example::

            from chunk_norris.llm.openai_llm import OpenAILLM

            norris = Norris(embedder=BertEmbedder())
            questions = norris.generate_questions(
                text=TEXT,
                llm=OpenAILLM(model="gpt-4o-mini-2024-07-18"),
                n=20,
            )

            # Inspect before running
            for q in questions:
                print(q["question"])
                print(q["expected_answer"])
                print()

            report = norris.run(text=TEXT, chunkers=[...], questions=questions)
        """
        token_count = len(_ENCODING.encode(text))
        print("chunk-norris: generating questions...")
        print(f"Document: {token_count:,} tokens | target: {n} questions")
        print()

        generator = QuestionGenerator(llm=llm)
        questions = generator.generate(text=text, n=n, seed=seed)

        print(f"Generated {len(questions)} question-answer pairs.")
        print()
        return questions

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
               using answer span coverage (token recall) and semantic
               focus (bert score).

        All configurations are then wrapped in a Report for comparison.

        Args:
            text (str): The document text to chunk and evaluate against.
            chunkers (list[BaseChunker]): One or more chunker configurations
                                          to compare.
            questions (list[dict]): A list of question dicts, each with:
                - "question" (str): The question to retrieve chunks for.
                - "expected_answer" (str): The ground truth answer for scoring.
                  Use norris.generate_questions() to generate these automatically.

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
                    "question":         question["question"],
                    "expected_answer":  question["expected_answer"],
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
