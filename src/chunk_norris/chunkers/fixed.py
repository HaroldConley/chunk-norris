from typing import Any

import tiktoken  # tiktoken==0.12.0

from chunk_norris.chunkers.base import BaseChunker


class FixedChunker(BaseChunker):
    """
    Splits text into fixed-size chunks measured in tokens, with optional overlap.

    Tokens are counted using tiktoken, which matches the tokenisation used by
    most LLM APIs (OpenAI, Anthropic, etc.). This makes chunk_size a reliable
    proxy for the context window cost of each chunk.

    Args:
        chunk_size (int): Number of tokens per chunk. Default: 256.
        overlap (float): Fraction of chunk_size to repeat at the start of the
                         next chunk. Must be between 0.0 and 1.0 (exclusive).
                         Example: overlap=0.1 means 10% of chunk_size tokens
                         are repeated. Default: 0.0 (no overlap).
        encoding_name (str): tiktoken encoding to use. Default: "cl100k_base",
                             which is used by GPT-4, Claude, and most modern LLMs.

    Example::

        chunker = FixedChunker(chunk_size=256, overlap=0.1)
        chunks = chunker.chunk("Your document text here...")

        for chunk in chunks:
            print(chunk["text"])
            print(chunk["metadata"])
            # metadata includes:
            # chunk_index, token_count, start_token, end_token,
            # chunk_size, overlap_fraction, overlap_tokens
    """

    def __init__(
        self,
        chunk_size: int = 256,
        overlap: float = 0.0,
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
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk(self, text: str) -> list[dict[str, Any]]:
        """
        Splits the input text into fixed-size token chunks with optional overlap.

        Args:
            text (str): The input text to split.

        Returns:
            List[Dict[str, Any]]: A list of chunk dicts, each containing:
                - 'text': the decoded chunk string
                - 'metadata': dict with the following keys:
                    - 'chunk_index'     : position of this chunk (0-based)
                    - 'token_count'     : number of tokens in this chunk
                    - 'start_token'     : index of the first token in the original sequence
                    - 'end_token'       : index of the last token (exclusive)
                    - 'chunk_size'      : the configured chunk_size
                    - 'overlap_fraction': the configured overlap as a fraction
                    - 'overlap_tokens'  : the configured overlap in actual tokens
        """
        if not text or not text.strip():
            return []

        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)
        step = self.chunk_size - self.overlap_tokens
        chunks = []
        start = 0
        index = 0

        while start < total_tokens:
            end = min(start + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)

            chunks.append(
                self._create_chunk(
                    text=chunk_text,
                    metadata={
                        "chunk_index": index,
                        "token_count": len(chunk_tokens),
                        "start_token": start,
                        "end_token": end,
                        "chunk_size": self.chunk_size,
                        "overlap_fraction": self.overlap,
                        "overlap_tokens": self.overlap_tokens,
                    },
                )
            )

            start += step
            index += 1

        return chunks

    def __repr__(self) -> str:
        return (
            f"FixedChunker("
            f"chunk_size={self.chunk_size}, "
            f"overlap={self.overlap})"
        )
