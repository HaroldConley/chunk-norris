from typing import Any

from chunk_norris.evaluator.retriever import Retriever
from chunk_norris.llm.base import BaseLLM


# Temperature used for answer generation — slightly creative but grounded.
# Lower than creative writing (0.7-1.0) but higher than scoring (0.0)
# to produce natural-sounding answers from the retrieved context.
_GENERATION_TEMPERATURE = 0.3
_GENERATION_MAX_TOKENS = 512

_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question below
using ONLY the context provided. If the context does not contain enough
information to answer the question, say "I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:"""


class Pipeline:
    """
    Runs the full RAG loop for a list of questions against a set of chunks.

    For each question the pipeline:
        1. Retrieves the most relevant chunks via the Retriever.
        2. Builds a prompt combining the question and retrieved chunks.
        3. Sends the prompt to the LLM and collects the generated answer.

    The pipeline does not score answers — that is the responsibility of
    metrics.py. It only produces the raw results that metrics.py consumes.

    Args:
        llm (BaseLLM): Any LLM implementing BaseLLM. e.g. OpenAILLM().
        retriever (Retriever): A Retriever instance with chunks already indexed.

    Example::

        from chunk_norris.llm.openai_llm import OpenAILLM
        from chunk_norris.embeddings.bert import BertEmbedder
        from chunk_norris.evaluator.retriever import Retriever
        from chunk_norris.evaluator.pipeline import Pipeline
        from chunk_norris.chunkers.fixed import FixedChunker

        chunker = FixedChunker(chunk_size=256, overlap=0.1)
        chunks = chunker.chunk("Your document text here...")

        retriever = Retriever(embedder=BertEmbedder(), top_k=3)
        retriever.index(chunks)

        pipeline = Pipeline(llm=OpenAILLM(), retriever=retriever)

        results = pipeline.run(questions=[
            {
                "question": "What is the refund policy?",
                "expected_answer": "Customers can request a refund within 30 days."
            }
        ])
    """

    def __init__(self, llm: BaseLLM, retriever: Retriever) -> None:
        self.llm = llm
        self.retriever = retriever

    def run(self, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Runs the full RAG loop for each question and returns the results.

        Args:
            questions (list[dict]): A list of question dicts, each containing:
                - "question" (str): The question to answer.
                - "expected_answer" (str): The expected answer for later scoring.

        Returns:
            list[dict]: One result dict per question, each containing:
                - "question" (str): The original question.
                - "expected_answer" (str): The expected answer (passed through).
                - "retrieved_chunks" (list[dict]): The top K chunks retrieved.
                - "generated_answer" (str): The LLM's answer.

        Raises:
            ValueError: If questions is empty or a question dict is missing
                        the "question" key.
            LLMError: If the LLM call fails.
            EmbeddingError: If the retrieval embedding call fails.
        """
        if not questions:
            raise ValueError("questions must not be empty.")

        results = []
        for item in questions:
            if "question" not in item:
                raise ValueError(
                    f"Each question dict must have a 'question' key. Got: {item}"
                )

            question = item["question"]
            expected_answer = item.get("expected_answer", "")

            retrieved_chunks = self.retriever.retrieve(question)
            prompt = self._build_prompt(question, retrieved_chunks)
            generated_answer = self.llm.generate(
                prompt=prompt,
                temperature=_GENERATION_TEMPERATURE,
                max_tokens=_GENERATION_MAX_TOKENS,
            )

            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "retrieved_chunks": retrieved_chunks,
                "generated_answer": generated_answer,
            })

        return results

    @staticmethod
    def _build_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
        """
        Builds the prompt sent to the LLM.

        Chunks are joined with a separator so the LLM can distinguish
        where one chunk ends and the next begins.

        Args:
            question (str): The question to answer.
            chunks (list[dict]): Retrieved chunks, each with a "text" key.

        Returns:
            str: The fully formatted prompt.
        """
        context = "\n---\n".join(chunk["text"] for chunk in chunks)
        return _PROMPT_TEMPLATE.format(context=context, question=question)