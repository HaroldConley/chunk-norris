from abc import ABC, abstractmethod
from typing import List, Dict, Any
from copy import deepcopy

class BaseChunker(ABC):
    """
    Abstract base class for text chunkers with flexible metadata support.
    """

    @abstractmethod
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """
        Splits the input text into chunks with metadata.

        Args:
            text (str): The input text to split.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each with:
                - 'text': the chunk string
                - 'metadata': a dictionary with chunk-specific metadata
        """
        pass

    def _create_chunk(self, text: str, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Helper method to create a chunk dictionary.

        Args:
            text (str): The chunk text.
            metadata (Dict[str, Any], optional): Arbitrary chunk-specific metadata.
                Each chunker is responsible for defining its own metadata fields.
                Common conventions:
                    - 'chunk_index'  : int  — position of the chunk in the document
                    - 'doc_name'     : str  — source document name
                    - 'page_number'  : int  — page number (if available)
                    - 'chunk_type'   : str  — e.g. 'fixed', 'sentence', 'semantic'

        Returns:
            Dict[str, Any]: A dictionary containing the chunk text and metadata.
        """
        return {
            "text": text,
            "metadata": deepcopy(metadata) if metadata else {}
        }
