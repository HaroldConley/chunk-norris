"""
chunk-norris
~~~~~~~~~~~~
Evaluate and compare chunking strategies for RAG pipelines.

Basic usage::

    from chunk_norris import Norris, BertEmbedder
    from chunk_norris.chunkers import FixedChunker, ParagraphChunker
    from chunk_norris.chunkers import SentenceChunker, RecursiveChunker

    norris = Norris(embedder=BertEmbedder())

    report = norris.run(
        text="Your document text here...",
        chunkers=[
            FixedChunker(chunk_size=128, overlap=0.1),
            FixedChunker(chunk_size=256, overlap=0.1),
            ParagraphChunker(max_tokens=256),
            SentenceChunker(sentences_per_chunk=5, overlap=1),
            RecursiveChunker(chunk_size=256, overlap=0.1),
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

    # Use the best chunker directly in your pipeline
    best_chunker = report.best_chunker()
    chunks = best_chunker.chunk("Your document text here...")

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

# LLM support will return in a future version for optional answer generation:
# from chunk_norris.llm.openai_llm import OpenAILLM
# from chunk_norris.llm.base import BaseLLM, LLMError

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
]
