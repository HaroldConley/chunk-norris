import re
from typing import Any

import tiktoken  # tiktoken==0.12.0

from chunk_norris.chunkers.base import BaseChunker


class ParagraphChunker(BaseChunker):
    """
    Splits text into chunks at paragraph boundaries (blank lines).

    Respects the document's natural structure — paragraphs are how authors
    group related ideas. Unlike FixedChunker, this never cuts mid-sentence
    or mid-paragraph, preserving semantic coherence.

    If a paragraph exceeds max_tokens, it is split further at sentence
    boundaries (full stops, exclamation marks, question marks) to keep
    chunks within the token limit. This ensures the size cap is respected
    without cutting words mid-sentence.

    No overlap is applied — paragraphs are treated as self-contained units.

    Args:
        max_tokens (int | None): Maximum number of tokens per chunk.
                                 If a paragraph exceeds this limit it is
                                 split at sentence boundaries.
                                 If None, no size cap is applied and each
                                 paragraph becomes exactly one chunk.
                                 Default: 512.
        encoding_name (str): tiktoken encoding for token counting.
                             Default: "cl100k_base".

    Example::

        chunker = ParagraphChunker(max_tokens=256)
        chunks = chunker.chunk("First paragraph.\\n\\nSecond paragraph.")

        for chunk in chunks:
            print(chunk["text"])
            print(chunk["metadata"])
            # metadata includes:
            # chunk_index, token_count, paragraph_index, was_split
    """

    def __init__(
        self,
        max_tokens: int | None = 512,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")

        self.max_tokens = max_tokens
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk(self, text: str) -> list[dict[str, Any]]:
        """
        Splits the input text into paragraph-based chunks.

        Args:
            text (str): The input text to split.

        Returns:
            List[Dict[str, Any]]: A list of chunk dicts, each containing:
                - 'text': the chunk string
                - 'metadata': dict with the following keys:
                    - 'chunk_index'     : position of this chunk (0-based)
                    - 'token_count'     : number of tokens in this chunk
                    - 'paragraph_index' : index of the source paragraph (0-based)
                    - 'was_split'       : True if paragraph exceeded max_tokens
                                          and was split into smaller pieces
        """
        if not text or not text.strip():
            return []

        paragraphs = self._split_paragraphs(text)
        chunks = []
        chunk_index = 0

        for para_index, paragraph in enumerate(paragraphs):
            token_count = len(self.encoding.encode(paragraph))

            if self.max_tokens is None or token_count <= self.max_tokens:
                chunks.append(
                    self._create_chunk(
                        text=paragraph,
                        metadata={
                            "chunk_index":     chunk_index,
                            "token_count":     token_count,
                            "paragraph_index": para_index,
                            "was_split":       False,
                        },
                    )
                )
                chunk_index += 1
            else:
                # Paragraph too long — split at sentence boundaries
                sentences = self._split_sentences(paragraph)
                for piece in self._group_sentences(sentences):
                    piece_tokens = len(self.encoding.encode(piece))
                    chunks.append(
                        self._create_chunk(
                            text=piece,
                            metadata={
                                "chunk_index":     chunk_index,
                                "token_count":     piece_tokens,
                                "paragraph_index": para_index,
                                "was_split":       True,
                            },
                        )
                    )
                    chunk_index += 1

        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        """
        Splits text on one or more consecutive blank lines.
        Strips whitespace from each paragraph and discards empty ones.
        """
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_sentences(self, text: str) -> list[str]:
        """
        Splits text into sentences at '.', '!', '?' boundaries.
        Preserves the punctuation at the end of each sentence.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _group_sentences(self, sentences: list[str]) -> list[str]:
        """
        Groups sentences into pieces that each fit within max_tokens.
        If a single sentence exceeds max_tokens, it becomes its own piece.
        """
        pieces = []
        current_sentences: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(self.encoding.encode(sentence))

            if current_sentences and current_tokens + sentence_tokens > self.max_tokens:
                pieces.append(" ".join(current_sentences))
                current_sentences = [sentence]
                current_tokens = sentence_tokens
            else:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens

        if current_sentences:
            pieces.append(" ".join(current_sentences))

        return pieces

    def __repr__(self) -> str:
        return f"ParagraphChunker(max_tokens={self.max_tokens})"
