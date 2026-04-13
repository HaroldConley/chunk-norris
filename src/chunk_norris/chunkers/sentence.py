from typing import Any

import nltk  # nltk==3.9.4
import tiktoken  # tiktoken==0.12.0

from chunk_norris.chunkers.base import BaseChunker


def _ensure_nltk_data() -> None:
    """
    Downloads the punkt_tab tokeniser data if not already present.
    Called once at import time — silent if data is already downloaded.
    """
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


_ensure_nltk_data()


class SentenceChunker(BaseChunker):
    """
    Splits text into chunks containing a fixed number of sentences.

    Uses NLTK's sentence tokeniser which correctly handles abbreviations
    ("Dr.", "U.S.A."), ellipsis, and other edge cases that break naive
    regex-based sentence splitting.

    Overlap is measured in sentences — overlap=1 means the last sentence
    of the previous chunk is repeated at the start of the next. This
    preserves context at chunk boundaries without token counting.

    Args:
        sentences_per_chunk (int): Number of sentences per chunk. Default: 5.
        overlap (int): Number of sentences to repeat from the end of the
                       previous chunk at the start of the next.
                       Must be less than sentences_per_chunk.
                       Default: 0 (no overlap).
        encoding_name (str): tiktoken encoding for token counting in metadata.
                             Default: "cl100k_base".

    Example::

        chunker = SentenceChunker(sentences_per_chunk=3, overlap=1)
        chunks = chunker.chunk("First sentence. Second sentence. Third sentence.")

        for chunk in chunks:
            print(chunk["text"])
            print(chunk["metadata"])
            # metadata includes:
            # chunk_index, token_count, sentence_count,
            # start_sentence, end_sentence, overlap
    """

    def __init__(
        self,
        sentences_per_chunk: int = 5,
        overlap: int = 0,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if sentences_per_chunk <= 0:
            raise ValueError(
                f"sentences_per_chunk must be positive, got {sentences_per_chunk}"
            )
        if overlap < 0:
            raise ValueError(
                f"overlap must be non-negative, got {overlap}"
            )
        if overlap >= sentences_per_chunk:
            raise ValueError(
                f"overlap ({overlap}) must be less than "
                f"sentences_per_chunk ({sentences_per_chunk})"
            )

        self.sentences_per_chunk = sentences_per_chunk
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk(self, text: str) -> list[dict[str, Any]]:
        """
        Splits the input text into sentence-based chunks.

        Args:
            text (str): The input text to split.

        Returns:
            List[Dict[str, Any]]: A list of chunk dicts, each containing:
                - 'text': the chunk string (sentences joined with a space)
                - 'metadata': dict with the following keys:
                    - 'chunk_index'        : position of this chunk (0-based)
                    - 'token_count'        : number of tokens in this chunk
                    - 'sentence_count'     : number of sentences in this chunk
                    - 'start_sentence'     : index of the first sentence (0-based)
                    - 'end_sentence'       : index of the last sentence (exclusive)
                    - 'overlap'            : configured overlap in sentences
        """
        if not text or not text.strip():
            return []

        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return []

        chunks = []
        step = self.sentences_per_chunk - self.overlap
        start = 0
        chunk_index = 0

        while start < len(sentences):
            end = min(start + self.sentences_per_chunk, len(sentences))
            chunk_sentences = sentences[start:end]
            chunk_text = " ".join(chunk_sentences)

            chunks.append(
                self._create_chunk(
                    text=chunk_text,
                    metadata={
                        "chunk_index":    chunk_index,
                        "token_count":    len(self.encoding.encode(chunk_text)),
                        "sentence_count": len(chunk_sentences),
                        "start_sentence": start,
                        "end_sentence":   end,
                        "overlap":        self.overlap,
                    },
                )
            )

            start += step
            chunk_index += 1

        return chunks

    def __repr__(self) -> str:
        return (
            f"SentenceChunker("
            f"sentences_per_chunk={self.sentences_per_chunk}, "
            f"overlap={self.overlap})"
        )
