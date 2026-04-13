import pytest

from chunk_norris.chunkers.recursive import RecursiveChunker, _DEFAULT_SEPARATORS


# ── Helpers ───────────────────────────────────────────────────────────────────

SHORT_TEXT  = "The quick brown fox jumps over the lazy dog."
PARA_TEXT   = "First paragraph content here.\n\nSecond paragraph content here.\n\nThird paragraph content here."
LONG_TEXT   = "The quick brown fox jumps over the lazy dog. " * 100


# ── TestRecursiveChunkerInit ──────────────────────────────────────────────────

class TestRecursiveChunkerInit:

    def test_default_values(self):
        chunker = RecursiveChunker()
        assert chunker.chunk_size == 256
        assert chunker.overlap == 0.0
        assert chunker.overlap_tokens == 0
        assert chunker.separators == _DEFAULT_SEPARATORS

    def test_custom_values_stored(self):
        chunker = RecursiveChunker(chunk_size=128, overlap=0.1)
        assert chunker.chunk_size == 128
        assert chunker.overlap == 0.1
        assert chunker.overlap_tokens == 12  # 10% of 128

    def test_custom_separators_stored(self):
        separators = ["\n\n", "\n", " "]
        chunker = RecursiveChunker(separators=separators)
        assert chunker.separators == separators

    def test_none_separators_uses_default(self):
        chunker = RecursiveChunker(separators=None)
        assert chunker.separators == _DEFAULT_SEPARATORS

    def test_overlap_tokens_rounds_down(self):
        # 10% of 256 = 25.6 → rounds down to 25
        chunker = RecursiveChunker(chunk_size=256, overlap=0.1)
        assert chunker.overlap_tokens == 25

    # --- chunk_size errors ---

    def test_zero_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size"):
            RecursiveChunker(chunk_size=0)

    def test_negative_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size"):
            RecursiveChunker(chunk_size=-10)

    # --- overlap errors ---

    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            RecursiveChunker(overlap=-0.1)

    def test_overlap_one_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            RecursiveChunker(overlap=1.0)

    def test_overlap_above_one_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            RecursiveChunker(overlap=1.5)

    # --- valid boundary values ---

    def test_overlap_zero_is_valid(self):
        chunker = RecursiveChunker(overlap=0.0)
        assert chunker.overlap == 0.0

    def test_overlap_just_below_one_is_valid(self):
        chunker = RecursiveChunker(overlap=0.99)
        assert chunker.overlap == 0.99

    # --- repr ---

    def test_repr(self):
        chunker = RecursiveChunker(chunk_size=128, overlap=0.1)
        assert repr(chunker) == "RecursiveChunker(chunk_size=128, overlap=0.1)"

    def test_repr_default_values(self):
        chunker = RecursiveChunker()
        assert repr(chunker) == "RecursiveChunker(chunk_size=256, overlap=0.0)"


# ── TestEdgeCases ─────────────────────────────────────────────────────────────

class TestRecursiveChunkerEdgeCases:

    def test_empty_string_returns_empty_list(self):
        chunker = RecursiveChunker()
        assert chunker.chunk("") == []

    def test_whitespace_only_returns_empty_list(self):
        chunker = RecursiveChunker()
        assert chunker.chunk("   \n\n   ") == []

    def test_short_text_returns_one_chunk(self):
        chunker = RecursiveChunker(chunk_size=256)
        chunks = chunker.chunk(SHORT_TEXT)
        assert len(chunks) == 1

    def test_short_text_content_preserved(self):
        chunker = RecursiveChunker(chunk_size=256)
        chunks = chunker.chunk(SHORT_TEXT)
        assert SHORT_TEXT in chunks[0]["text"]


# ── TestNormalBehaviour ───────────────────────────────────────────────────────

class TestRecursiveChunkerNormalBehaviour:

    def test_returns_list_of_dicts(self):
        chunker = RecursiveChunker(chunk_size=20)
        chunks = chunker.chunk(LONG_TEXT)
        assert isinstance(chunks, list)
        assert all(isinstance(c, dict) for c in chunks)

    def test_each_chunk_has_text_and_metadata(self):
        chunker = RecursiveChunker(chunk_size=20)
        chunks = chunker.chunk(LONG_TEXT)
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk

    def test_chunk_indices_are_sequential(self):
        chunker = RecursiveChunker(chunk_size=20)
        chunks = chunker.chunk(LONG_TEXT)
        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["chunk_index"] == i

    def test_each_chunk_respects_max_token_size(self):
        chunker = RecursiveChunker(chunk_size=20)
        chunks = chunker.chunk(LONG_TEXT)
        for chunk in chunks:
            assert chunk["metadata"]["token_count"] <= 20

    def test_multiple_chunks_for_long_text(self):
        chunker = RecursiveChunker(chunk_size=20)
        chunks = chunker.chunk(LONG_TEXT)
        assert len(chunks) > 1

    def test_metadata_contains_all_keys(self):
        chunker = RecursiveChunker(chunk_size=20)
        chunks = chunker.chunk(LONG_TEXT)
        expected_keys = {
            "chunk_index",
            "token_count",
            "chunk_size",
            "overlap_fraction",
            "overlap_tokens",
            "separator_used",
        }
        for chunk in chunks:
            assert expected_keys == set(chunk["metadata"].keys())

    def test_chunk_size_stored_in_metadata(self):
        chunker = RecursiveChunker(chunk_size=30)
        chunks = chunker.chunk(LONG_TEXT)
        for chunk in chunks:
            assert chunk["metadata"]["chunk_size"] == 30

    def test_overlap_fraction_stored_in_metadata(self):
        chunker = RecursiveChunker(chunk_size=30, overlap=0.2)
        chunks = chunker.chunk(LONG_TEXT)
        for chunk in chunks:
            assert chunk["metadata"]["overlap_fraction"] == 0.2

    def test_overlap_tokens_stored_in_metadata(self):
        chunker = RecursiveChunker(chunk_size=30, overlap=0.2)
        chunks = chunker.chunk(LONG_TEXT)
        for chunk in chunks:
            assert chunk["metadata"]["overlap_tokens"] == 6  # 20% of 30

    def test_separator_used_is_string(self):
        chunker = RecursiveChunker(chunk_size=20)
        chunks = chunker.chunk(LONG_TEXT)
        for chunk in chunks:
            assert isinstance(chunk["metadata"]["separator_used"], str)

    def test_token_count_positive(self):
        chunker = RecursiveChunker(chunk_size=20)
        chunks = chunker.chunk(LONG_TEXT)
        for chunk in chunks:
            assert chunk["metadata"]["token_count"] > 0


# ── TestSeparatorPriority ─────────────────────────────────────────────────────

class TestRecursiveChunkerSeparatorPriority:

    def test_paragraph_separator_used_for_paragraph_text(self):
        # chunk_size=5 ensures no two paragraphs can be merged together.
        # Each paragraph is ~5-6 tokens so each stays as its own chunk.
        chunker = RecursiveChunker(chunk_size=5)
        chunks = chunker.chunk(PARA_TEXT)
        # Should produce at least 3 chunks — one per paragraph minimum
        assert len(chunks) >= 3

    def test_falls_back_to_word_level_for_no_separators(self):
        # Text with no paragraph or sentence breaks forces word-level splitting
        text = "word " * 100   # 100 words, no punctuation
        chunker = RecursiveChunker(chunk_size=10)
        chunks = chunker.chunk(text)
        # Should produce multiple chunks, each within token limit
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk["metadata"]["token_count"] <= 10

    def test_custom_separators_respected(self):
        # chunk_size=3 ensures each part stays separate after splitting on "|"
        # "part one", "part two", "part three" are each ~2 tokens
        # but combined they exceed 3 tokens so they cannot be merged
        text = "part one|part two|part three"
        chunker = RecursiveChunker(chunk_size=3, separators=["|", ""])
        chunks = chunker.chunk(text)
        assert len(chunks) == 3

    def test_default_separators_order(self):
        assert _DEFAULT_SEPARATORS[0] == "\n\n"   # paragraph first
        assert _DEFAULT_SEPARATORS[-1] == ""       # character last


# ── TestOverlapBehaviour ──────────────────────────────────────────────────────

class TestRecursiveChunkerOverlapBehaviour:

    def test_no_overlap_chunks_independent(self):
        chunker = RecursiveChunker(chunk_size=20, overlap=0.0)
        chunks = chunker.chunk(LONG_TEXT)
        # With no overlap, metadata overlap_tokens should be 0
        for chunk in chunks:
            assert chunk["metadata"]["overlap_tokens"] == 0

    def test_overlap_metadata_reflects_config(self):
        chunker = RecursiveChunker(chunk_size=20, overlap=0.25)
        chunks = chunker.chunk(LONG_TEXT)
        for chunk in chunks:
            assert chunk["metadata"]["overlap_fraction"] == 0.25
            assert chunk["metadata"]["overlap_tokens"] == 5  # 25% of 20

    def test_overlap_produces_more_chunks_than_no_overlap(self):
        # Overlap means the window advances less each step → more chunks
        chunker_no_overlap   = RecursiveChunker(chunk_size=20, overlap=0.0)
        chunker_with_overlap = RecursiveChunker(chunk_size=20, overlap=0.25)
        chunks_no  = chunker_no_overlap.chunk(LONG_TEXT)
        chunks_yes = chunker_with_overlap.chunk(LONG_TEXT)
        assert len(chunks_yes) >= len(chunks_no)


# ── TestApplyOverlap ──────────────────────────────────────────────────────────

class TestApplyOverlap:
    """Tests the internal _apply_overlap helper directly."""

    def test_returns_empty_for_zero_overlap(self):
        chunker = RecursiveChunker(chunk_size=20, overlap=0.0)
        texts, tokens = chunker._apply_overlap(["hello", "world"])
        assert texts == []
        assert tokens == 0

    def test_returns_last_text_fitting_overlap(self):
        # overlap_tokens = 25% of 20 = 5
        chunker = RecursiveChunker(chunk_size=20, overlap=0.25)
        # "world" is ~1 token, fits within 5
        texts, tokens = chunker._apply_overlap(["hello", "world"])
        assert "world" in texts
        assert tokens > 0

    def test_does_not_exceed_overlap_tokens(self):
        chunker = RecursiveChunker(chunk_size=20, overlap=0.25)
        texts, tokens = chunker._apply_overlap(["hello", "world", "foo"])
        assert tokens <= chunker.overlap_tokens
