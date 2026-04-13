from typing import Any

import tiktoken  # tiktoken==0.12.0

from chunk_norris.chunkers.base import BaseChunker


# Default separator priority order — from coarsest to finest granularity.
# The chunker tries each separator in order, falling to the next only when
# a piece still exceeds chunk_size after splitting at the current level.
_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveChunker(BaseChunker):
    """
    Splits text by trying natural boundaries in priority order.

    Attempts to split on paragraph breaks first, then line breaks, then
    sentence endings, then words, then individual characters as a last
    resort. Falls to the next separator only when a piece still exceeds
    chunk_size after splitting at the current level.

    This produces chunks that respect semantic boundaries as much as
    possible while guaranteeing a maximum token size — making it the
    most generally useful chunker for documents without a known structure.

    This is the same approach used by LangChain's RecursiveCharacterTextSplitter,
    adapted to use token counts instead of character counts.

    Args:
        chunk_size (int): Maximum number of tokens per chunk. Default: 256.
        overlap (float): Fraction of chunk_size to repeat at the start of
                         the next chunk. Between 0.0 and 1.0 (exclusive).
                         Default: 0.0 (no overlap).
        separators (list[str] | None): Priority-ordered list of separators.
                                       If None, uses the default list:
                                       ["\\n\\n", "\\n", ". ", " ", ""]
                                       The empty string "" is the character-level
                                       fallback and should always be last.
        encoding_name (str): tiktoken encoding. Default: "cl100k_base".

    Example::

        chunker = RecursiveChunker(chunk_size=256, overlap=0.1)
        chunks = chunker.chunk("Your document text here...")

        for chunk in chunks:
            print(chunk["text"])
            print(chunk["metadata"])
            # metadata includes:
            # chunk_index, token_count, chunk_size,
            # overlap_fraction, overlap_tokens, separator_used
    """

    def __init__(
        self,
        chunk_size: int = 256,
        overlap: float = 0.0,
        separators: list[str] | None = None,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if not (0.0 <= overlap < 1.0):
            raise ValueError(
                f"overlap must be between 0.0 and 1.0 (exclusive), got {overlap}"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.overlap_tokens = int(chunk_size * overlap)
        self.separators = separators if separators is not None else _DEFAULT_SEPARATORS
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk(self, text: str) -> list[dict[str, Any]]:
        """
        Splits the input text into chunks using recursive separator fallback.

        Args:
            text (str): The input text to split.

        Returns:
            List[Dict[str, Any]]: A list of chunk dicts, each containing:
                - 'text': the chunk string
                - 'metadata': dict with the following keys:
                    - 'chunk_index'     : position of this chunk (0-based)
                    - 'token_count'     : number of tokens in this chunk
                    - 'chunk_size'      : the configured chunk_size
                    - 'overlap_fraction': the configured overlap as a fraction
                    - 'overlap_tokens'  : the configured overlap in actual tokens
                    - 'separator_used'  : the separator that produced this chunk
        """
        if not text or not text.strip():
            return []

        raw_chunks = self._split(text, self.separators)
        merged = self._merge(raw_chunks)
        chunks = []

        for i, (chunk_text, separator) in enumerate(merged):
            token_count = len(self.encoding.encode(chunk_text))
            chunks.append(
                self._create_chunk(
                    text=chunk_text,
                    metadata={
                        "chunk_index":      i,
                        "token_count":      token_count,
                        "chunk_size":       self.chunk_size,
                        "overlap_fraction": self.overlap,
                        "overlap_tokens":   self.overlap_tokens,
                        "separator_used":   repr(separator),
                    },
                )
            )

        return chunks

    def _split(self, text: str, separators: list[str]) -> list[tuple[str, str]]:
        """
        Recursively splits text using separators in priority order.

        Returns a list of (piece_text, separator_used) tuples — the separator
        that was used to produce each piece is tracked for metadata.

        If the current separator produces a piece that still exceeds chunk_size,
        that piece is recursively split using the remaining separators.
        """
        if not separators:
            return [(text, "")]

        separator = separators[0]
        remaining = separators[1:]

        if separator == "":
            # Character-level fallback — split into individual tokens
            tokens = self.encoding.encode(text)
            pieces = []
            for i in range(0, len(tokens), self.chunk_size):
                piece_tokens = tokens[i:i + self.chunk_size]
                pieces.append((self.encoding.decode(piece_tokens), ""))
            return pieces

        parts = text.split(separator)
        result: list[tuple[str, str]] = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            token_count = len(self.encoding.encode(part))
            if token_count <= self.chunk_size:
                result.append((part, separator))
            else:
                # Part still too large — recurse with next separator
                result.extend(self._split(part, remaining))

        return result

    def _merge(
        self, pieces: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        """
        Merges small pieces into chunks up to chunk_size, with overlap.

        Pieces that individually fit within chunk_size are merged greedily.
        When adding the next piece would exceed chunk_size, the current
        buffer is saved as a chunk and a new buffer starts — beginning with
        overlap_tokens worth of content from the end of the previous chunk.
        """
        if not pieces:
            return []

        chunks: list[tuple[str, str]] = []
        current_texts: list[str] = []
        current_tokens = 0
        last_separator = ""

        for piece_text, separator in pieces:
            piece_tokens = len(self.encoding.encode(piece_text))

            if current_texts and current_tokens + piece_tokens > self.chunk_size:
                # Save current buffer as a chunk
                chunk_text = " ".join(current_texts)
                chunks.append((chunk_text, last_separator))

                # Start new buffer with overlap
                if self.overlap_tokens > 0:
                    current_texts, current_tokens = self._apply_overlap(
                        current_texts
                    )
                else:
                    current_texts = []
                    current_tokens = 0

            current_texts.append(piece_text)
            current_tokens += piece_tokens
            last_separator = separator

        if current_texts:
            chunks.append((" ".join(current_texts), last_separator))

        return chunks

    def _apply_overlap(
        self, texts: list[str]
    ) -> tuple[list[str], int]:
        """
        Returns the tail of a text list that fits within overlap_tokens.
        Works backwards through the list, adding texts until the token
        budget is exhausted.
        """
        overlap_texts: list[str] = []
        token_count = 0

        for text in reversed(texts):
            text_tokens = len(self.encoding.encode(text))
            if token_count + text_tokens > self.overlap_tokens:
                break
            overlap_texts.insert(0, text)
            token_count += text_tokens

        return overlap_texts, token_count

    def __repr__(self) -> str:
        return (
            f"RecursiveChunker("
            f"chunk_size={self.chunk_size}, "
            f"overlap={self.overlap})"
        )
