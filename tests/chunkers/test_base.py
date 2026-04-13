import pytest
from typing import Any

from chunk_norris.chunkers.base import BaseChunker


# ── Minimal concrete implementation for testing ───────────────────────────────
#
# BaseChunker is abstract — we can't instantiate it directly.
# This minimal subclass lets us test the base class behaviour
# without any of the complexity of a real chunker.

class MinimalChunker(BaseChunker):
    """Minimal concrete chunker for testing BaseChunker behaviour."""

    def chunk(self, text: str) -> list[dict[str, Any]]:
        """Splits text into single-word chunks."""
        return [
            self._create_chunk(text=word, metadata={"word_index": i})
            for i, word in enumerate(text.split())
        ]

    def __repr__(self) -> str:
        return "MinimalChunker()"


# ── TestBaseChunkerContract ───────────────────────────────────────────────────

class TestBaseChunkerContract:
    """
    Verifies that the BaseChunker abstract contract is enforced correctly.
    Any class that inherits from BaseChunker must implement chunk().
    """

    def test_cannot_instantiate_base_directly(self):
        with pytest.raises(TypeError):
            BaseChunker()

    def test_subclass_without_chunk_raises(self):
        class IncompleteChunker(BaseChunker):
            pass  # missing chunk() implementation

        with pytest.raises(TypeError):
            IncompleteChunker()

    def test_concrete_subclass_can_be_instantiated(self):
        chunker = MinimalChunker()
        assert chunker is not None

    def test_chunk_returns_list(self):
        chunker = MinimalChunker()
        result = chunker.chunk("hello world")
        assert isinstance(result, list)

    def test_chunk_returns_list_of_dicts(self):
        chunker = MinimalChunker()
        result = chunker.chunk("hello world foo")
        assert all(isinstance(c, dict) for c in result)

    def test_each_chunk_has_text_key(self):
        chunker = MinimalChunker()
        result = chunker.chunk("hello world")
        for chunk in result:
            assert "text" in chunk

    def test_each_chunk_has_metadata_key(self):
        chunker = MinimalChunker()
        result = chunker.chunk("hello world")
        for chunk in result:
            assert "metadata" in chunk

    def test_chunk_text_is_string(self):
        chunker = MinimalChunker()
        result = chunker.chunk("hello world")
        for chunk in result:
            assert isinstance(chunk["text"], str)

    def test_chunk_metadata_is_dict(self):
        chunker = MinimalChunker()
        result = chunker.chunk("hello world")
        for chunk in result:
            assert isinstance(chunk["metadata"], dict)


# ── TestCreateChunk ───────────────────────────────────────────────────────────

class TestCreateChunk:
    """
    Tests for the _create_chunk() helper method.
    This covers the one line that was missing from coverage.
    """

    def test_with_metadata_returns_correct_text(self):
        chunker = MinimalChunker()
        chunk = chunker._create_chunk(text="hello", metadata={"key": "value"})
        assert chunk["text"] == "hello"

    def test_with_metadata_returns_correct_metadata(self):
        chunker = MinimalChunker()
        chunk = chunker._create_chunk(text="hello", metadata={"key": "value"})
        assert chunk["metadata"] == {"key": "value"}

    def test_without_metadata_returns_empty_dict(self):
        # This is the previously uncovered line — the else {} branch
        chunker = MinimalChunker()
        chunk = chunker._create_chunk(text="hello")
        assert chunk["metadata"] == {}

    def test_none_metadata_returns_empty_dict(self):
        chunker = MinimalChunker()
        chunk = chunker._create_chunk(text="hello", metadata=None)
        assert chunk["metadata"] == {}

    def test_empty_metadata_returns_empty_dict(self):
        chunker = MinimalChunker()
        chunk = chunker._create_chunk(text="hello", metadata={})
        assert chunk["metadata"] == {}

    def test_nested_metadata_is_deep_copied(self):
        # Mutating the original metadata should not affect the chunk
        original_metadata = {"scores": [0.9, 0.8], "source": {"file": "doc.txt"}}
        chunker = MinimalChunker()
        chunk = chunker._create_chunk(text="hello", metadata=original_metadata)

        # Mutate the original
        original_metadata["scores"].append(0.5)
        original_metadata["source"]["file"] = "other.txt"

        # Chunk metadata should be unchanged
        assert chunk["metadata"]["scores"] == [0.9, 0.8]
        assert chunk["metadata"]["source"]["file"] == "doc.txt"

    def test_chunk_metadata_independent_between_calls(self):
        # Two chunks with the same metadata dict should be independent
        chunker = MinimalChunker()
        shared_metadata = {"tag": "original"}

        chunk_a = chunker._create_chunk(text="a", metadata=shared_metadata)
        chunk_b = chunker._create_chunk(text="b", metadata=shared_metadata)

        chunk_a["metadata"]["tag"] = "modified"
        assert chunk_b["metadata"]["tag"] == "original"

    def test_arbitrary_metadata_types_supported(self):
        chunker = MinimalChunker()
        metadata = {
            "int_field":   42,
            "float_field": 0.95,
            "bool_field":  True,
            "list_field":  [1, 2, 3],
            "dict_field":  {"nested": "value"},
            "none_field":  None,
        }
        chunk = chunker._create_chunk(text="hello", metadata=metadata)
        assert chunk["metadata"]["int_field"]   == 42
        assert chunk["metadata"]["float_field"] == 0.95
        assert chunk["metadata"]["bool_field"]  is True
        assert chunk["metadata"]["list_field"]  == [1, 2, 3]
        assert chunk["metadata"]["dict_field"]  == {"nested": "value"}
        assert chunk["metadata"]["none_field"]  is None


# ── TestMinimalChunkerBehaviour ───────────────────────────────────────────────

class TestMinimalChunkerBehaviour:
    """
    Verifies that a correctly implemented subclass works as expected.
    These tests serve as a template — every chunker should pass
    equivalents of these tests.
    """

    def test_produces_one_chunk_per_word(self):
        chunker = MinimalChunker()
        result = chunker.chunk("one two three")
        assert len(result) == 3

    def test_chunk_texts_match_words(self):
        chunker = MinimalChunker()
        result = chunker.chunk("hello world")
        assert result[0]["text"] == "hello"
        assert result[1]["text"] == "world"

    def test_empty_string_returns_empty_list(self):
        chunker = MinimalChunker()
        result = chunker.chunk("")
        assert result == []

    def test_metadata_contains_word_index(self):
        chunker = MinimalChunker()
        result = chunker.chunk("a b c")
        assert result[0]["metadata"]["word_index"] == 0
        assert result[1]["metadata"]["word_index"] == 1
        assert result[2]["metadata"]["word_index"] == 2

    def test_repr_returns_string(self):
        chunker = MinimalChunker()
        assert isinstance(repr(chunker), str)

    def test_repr_is_not_empty(self):
        chunker = MinimalChunker()
        assert len(repr(chunker)) > 0
