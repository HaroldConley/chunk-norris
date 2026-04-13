from sentence_transformers import SentenceTransformer  # sentence-transformers==5.3.0

from chunk_norris.embeddings.base import BaseEmbedder, EmbeddingError


class BertEmbedder(BaseEmbedder):
    """
    BERT-based embedder using the sentence-transformers library.

    Downloads and caches the model locally on first use. Subsequent uses
    load from cache — no internet connection required after the first run.

    Uses all-MiniLM-L6-v2 by default, which is a strong general-purpose
    model for semantic search: fast, lightweight (~80MB), and works well
    for RAG retrieval across a wide range of document types.

    Args:
        model_name (str): Any model from https://huggingface.co/sentence-transformers.
                          Default: "all-MiniLM-L6-v2".
        device (str | None): Device to run the model on. Options: "cpu", "cuda"
                             (for NVIDIA GPU), "mps" (for Apple Silicon).
                             If None, automatically selects the best available
                             device. Default: None.
        batch_size (int): Number of texts to embed in a single forward pass.
                          Larger batches are faster but use more memory.
                          Default: 32.

    Example::

        from chunk_norris.embeddings.bert import BertEmbedder

        embedder = BertEmbedder()

        vectors = embedder.embed([
            "What is the refund policy?",
            "Customers can request a refund within 30 days.",
        ])

        # vectors is a list of two float vectors, one per input text
        print(len(vectors))     # 2
        print(len(vectors[0]))  # 384 (dimensionality of all-MiniLM-L6-v2)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

        try:
            self._model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            raise EmbeddingError(
                message=(
                    f"Failed to load embedding model '{model_name}'. "
                    "Check that the model name is valid and you have an internet "
                    "connection for the first download."
                ),
                provider="BERT (sentence-transformers)",
                original_error=e,
            ) from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Converts a list of texts into a list of embedding vectors.

        Texts are processed in batches of batch_size for efficiency.
        The model is run in inference mode (no gradient computation).

        Args:
            texts (list[str]): The texts to embed.

        Returns:
            list[list[float]]: A list of embedding vectors, one per input text.
                               Each vector has 384 dimensions for the default
                               all-MiniLM-L6-v2 model.

        Raises:
            ValueError: If texts is empty.
            EmbeddingError: If the embedding call fails.
        """
        if not texts:
            raise ValueError("texts must not be empty.")

        try:
            vectors = self._model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return [v.tolist() for v in vectors]

        except Exception as e:
            raise EmbeddingError(
                message="Failed to generate embeddings.",
                provider="BERT (sentence-transformers)",
                original_error=e,
            ) from e