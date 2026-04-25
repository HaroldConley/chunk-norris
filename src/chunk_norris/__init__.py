"""
chunk-norris
~~~~~~~~~~~~
Evaluate and compare chunking strategies for RAG pipelines.

Basic usage::

    from chunk_norris import Norris, BertEmbedder
    from chunk_norris.chunkers import FixedChunker, RecursiveChunker
    from chunk_norris.llm.openai_llm import OpenAILLM

    norris = Norris(embedder=BertEmbedder())

    # Option A — auto-generate questions (requires OPENAI_API_KEY in .env)
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
            RecursiveChunker(chunk_size=256, overlap=0.1),
        ],
        questions=questions,
    )

    report.compare()
    report.best()

    # Use the best chunker directly in your pipeline
    best_chunker = report.best_chunker()
    chunks = best_chunker.chunk(TEXT)

"""

__version__ = "0.1.0"

from chunk_norris.norris import Norris

# --- chunkers ---
from chunk_norris.chunkers.fixed     import FixedChunker
from chunk_norris.chunkers.paragraph import ParagraphChunker
from chunk_norris.chunkers.sentence  import SentenceChunker
from chunk_norris.chunkers.recursive import RecursiveChunker

# --- embeddings ---
from chunk_norris.embeddings.bert import BertEmbedder
from chunk_norris.embeddings.base import BaseEmbedder, EmbeddingError

# --- llm ---
from chunk_norris.llm.openai_llm import OpenAILLM
from chunk_norris.llm.base       import BaseLLM, LLMError

# --- retrieval ---
from chunk_norris.retrieval.hybrid import HybridRetriever
from chunk_norris.retrieval.dense  import DenseRetriever
from chunk_norris.retrieval.bm25   import BM25Retriever
from chunk_norris.retrieval.base   import BaseRetriever

__all__ = [
    "__version__",
    # --- main entry point ---
    "Norris",
    # --- chunkers ---
    "FixedChunker",
    "ParagraphChunker",
    "SentenceChunker",
    "RecursiveChunker",
    # --- embeddings ---
    "BertEmbedder",
    "BaseEmbedder",
    "EmbeddingError",
    # --- llm ---
    "OpenAILLM",
    "BaseLLM",
    "LLMError",
    # --- retrieval ---
    "HybridRetriever",
    "DenseRetriever",
    "BM25Retriever",
    "BaseRetriever",
]
