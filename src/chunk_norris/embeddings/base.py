from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """
    Abstract base class for text embedding providers.

    All embedding integrations (BERT, OpenAI, Cohere, etc.) must inherit
    from this class and implement the embed() method. This keeps the
    retriever completely decoupled from any specific embedding library.

    The embed() method accepts a list of texts and returns a list of vectors
    (one vector per text). Batching is handled at the implementation level —
    the caller does not need to worry about it.

    Example of implementing a custom embedder::

        from chunk_norris.embeddings.base import BaseEmbedder

        class MyEmbedder(BaseEmbedder):
            def embed(self, texts: list[str]) -> list[list[float]]:
                return my_model.encode(texts).tolist()

    Then pass it to Norris::

        from chunk_norris import Norris
        norris = Norris(llm=..., embedder=MyEmbedder())
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Converts a list of texts into a list of embedding vectors.

        Args:
            texts (list[str]): The texts to embed. Can be questions or chunk texts.

        Returns:
            list[list[float]]: A list of vectors, one per input text.
                               All vectors from the same embedder have the
                               same dimensionality.

        Raises:
            EmbeddingError: If the embedding call fails for any reason.
        """
        pass


class EmbeddingError(Exception):
    """
    Raised when an embedding call fails.

    Wraps embedder-specific exceptions into a single exception type
    so the retriever does not need to handle BERT, OpenAI, or Cohere
    errors separately.

    Attributes:
        message (str): Human-readable description of the failure.
        provider (str): Name of the embedding provider that raised the error.
        original_error (Exception): The original exception from the provider.
    """

    def __init__(self, message: str, provider: str, original_error: Exception) -> None:
        super().__init__(message)
        self.provider = provider
        self.original_error = original_error

    def __str__(self) -> str:
        return (
            f"[{self.provider}] {super().__str__()} "
            f"(original: {type(self.original_error).__name__}: {self.original_error})"
        )