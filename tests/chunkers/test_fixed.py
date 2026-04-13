import pytest
from chunk_norris.chunkers.fixed import FixedChunker


# ── Helpers ───────────────────────────────────────────────────────────────────

SHORT_TEXT = "The quick brown fox jumps over the lazy dog."
LONG_TEXT = "The quick brown fox jumps over the lazy dog. " * 100


# ── Initialisation ────────────────────────────────────────────────────────────

class TestFixedChunkerInit:

    def test_default_values(self):
        chunker = FixedChunker()
        assert chunker.chunk_size == 256
        assert chunker.overlap == 0.0
        assert chunker.overlap_tokens == 0

    def test_custom_values_stored_correctly(self):
        chunker = FixedChunker(chunk_size=128, overlap=0.25)
        assert chunker.chunk_size == 128
        assert chunker.overlap == 0.25
        assert chunker.overlap_tokens == 32  # 25% of 128

    def test_overlap_tokens_rounds_down(self):
        # 10% of 256 = 25.6 → should round down to 25
        chunker = FixedChunker(chunk_size=256, overlap=0.1)
        assert chunker.overlap_tokens == 25

    # --- chunk_size errors ---

    def test_chunk_size_zero_raises(self):
        with pytest.raises(ValueError, match="chunk_size"):
            FixedChunker(chunk_size=0)

    def test_chunk_size_negative_raises(self):
        with pytest.raises(ValueError, match="chunk_size"):
            FixedChunker(chunk_size=-10)

    # --- overlap errors ---

    def test_overlap_negative_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            FixedChunker(overlap=-0.1)

    def test_overlap_exactly_one_raises(self):
        # 1.0 would mean the window never advances
        with pytest.raises(ValueError, match="overlap"):
            FixedChunker(overlap=1.0)

    def test_overlap_greater_than_one_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            FixedChunker(overlap=1.5)

    # --- valid boundary values ---

    def test_overlap_zero_is_valid(self):
        chunker = FixedChunker(overlap=0.0)
        assert chunker.overlap == 0.0

    def test_overlap_just_below_one_is_valid(self):
        chunker = FixedChunker(overlap=0.99)
        assert chunker.overlap == 0.99

    def test_repr(self):
        chunker = FixedChunker(chunk_size=128, overlap=0.25)
        assert repr(chunker) == "FixedChunker(chunk_size=128, overlap=0.25)"


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestFixedChunkerEdgeCases:

    def test_empty_string_returns_empty_list(self):
        chunker = FixedChunker()
        assert chunker.chunk("") == []

    def test_whitespace_only_returns_empty_list(self):
        chunker = FixedChunker()
        assert chunker.chunk("     ") == []
        assert chunker.chunk("\n\n\n") == []
        assert chunker.chunk("\t\t") == []

    def test_text_shorter_than_chunk_size_returns_one_chunk(self):
        chunker = FixedChunker(chunk_size=256)
        chunks = chunker.chunk(SHORT_TEXT)
        assert len(chunks) == 1

    def test_text_exactly_chunk_size_returns_one_chunk(self):
        # Build a text that is exactly chunk_size tokens long
        chunker = FixedChunker(chunk_size=10)
        tokens = chunker.encoding.encode(SHORT_TEXT)[:10]
        exact_text = chunker.encoding.decode(tokens)
        chunks = chunker.chunk(exact_text)
        assert len(chunks) == 1


# ── Normal chunking behaviour ─────────────────────────────────────────────────

class TestFixedChunkerChunk:

    def test_returns_list_of_dicts(self):
        chunker = FixedChunker(chunk_size=20)
        chunks = chunker.chunk(LONG_TEXT)
        assert isinstance(chunks, list)
        assert all(isinstance(c, dict) for c in chunks)

    def test_each_chunk_has_text_and_metadata_keys(self):
        chunker = FixedChunker(chunk_size=20)
        chunks = chunker.chunk(LONG_TEXT)
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk

    def test_chunk_text_is_non_empty_string(self):
        chunker = FixedChunker(chunk_size=20)
        chunks = chunker.chunk(LONG_TEXT)
        for chunk in chunks:
            assert isinstance(chunk["text"], str)
            assert len(chunk["text"]) > 0

    def test_each_chunk_respects_max_token_size(self):
        chunker = FixedChunker(chunk_size=20)
        chunks = chunker.chunk(LONG_TEXT)
        for chunk in chunks:
            assert chunk["metadata"]["token_count"] <= 20

    def test_multiple_chunks_produced_for_long_text(self):
        chunker = FixedChunker(chunk_size=20)
        chunks = chunker.chunk(LONG_TEXT)
        assert len(chunks) > 1

    def test_chunk_indices_are_sequential(self):
        chunker = FixedChunker(chunk_size=20)
        chunks = chunker.chunk(LONG_TEXT)
        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["chunk_index"] == i

    def test_start_and_end_tokens_are_consistent(self):
        chunker = FixedChunker(chunk_size=20)
        chunks = chunker.chunk(LONG_TEXT)
        for chunk in chunks:
            meta = chunk["metadata"]
            assert meta["end_token"] - meta["start_token"] == meta["token_count"]

    def test_first_chunk_starts_at_token_zero(self):
        chunker = FixedChunker(chunk_size=20)
        chunks = chunker.chunk(LONG_TEXT)
        assert chunks[0]["metadata"]["start_token"] == 0

    def test_metadata_contains_all_expected_keys(self):
        chunker = FixedChunker(chunk_size=20, overlap=0.1)
        chunks = chunker.chunk(LONG_TEXT)
        expected_keys = {
            "chunk_index",
            "token_count",
            "start_token",
            "end_token",
            "chunk_size",
            "overlap_fraction",
            "overlap_tokens",
        }
        for chunk in chunks:
            assert expected_keys == set(chunk["metadata"].keys())

    def test_metadata_reflects_chunker_config(self):
        chunker = FixedChunker(chunk_size=30, overlap=0.2)
        chunks = chunker.chunk(LONG_TEXT)
        for chunk in chunks:
            assert chunk["metadata"]["chunk_size"] == 30
            assert chunk["metadata"]["overlap_fraction"] == 0.2
            assert chunk["metadata"]["overlap_tokens"] == 6  # 20% of 30


# ── Overlap behaviour ─────────────────────────────────────────────────────────

class TestFixedChunkerOverlap:

    def test_no_overlap_chunks_do_not_repeat_tokens(self):
        chunker = FixedChunker(chunk_size=20, overlap=0.0)
        chunks = chunker.chunk(LONG_TEXT)
        # With no overlap, consecutive start tokens should advance by chunk_size
        for i in range(len(chunks) - 1):
            current_end = chunks[i]["metadata"]["end_token"]
            next_start = chunks[i + 1]["metadata"]["start_token"]
            assert next_start == current_end

    def test_overlap_advances_window_correctly(self):
        chunker = FixedChunker(chunk_size=20, overlap=0.25)
        chunks = chunker.chunk(LONG_TEXT)
        step = 20 - 5  # chunk_size - overlap_tokens (25% of 20 = 5)
        for i in range(len(chunks) - 1):
            current_start = chunks[i]["metadata"]["start_token"]
            next_start = chunks[i + 1]["metadata"]["start_token"]
            assert next_start - current_start == step

    def test_overlapping_chunks_share_tokens(self):
        chunker = FixedChunker(chunk_size=20, overlap=0.25)
        chunks = chunker.chunk(LONG_TEXT)
        overlap_tokens = chunker.overlap_tokens
        # The last N tokens of chunk[i] should equal the first N tokens of chunk[i+1]
        for i in range(len(chunks) - 1):
            end_of_current = chunks[i]["metadata"]["end_token"]
            start_of_next = chunks[i + 1]["metadata"]["start_token"]
            shared = end_of_current - start_of_next
            assert shared == overlap_tokens

    def test_metadata_is_independent_between_chunks(self):
        # Verifies deepcopy — mutating one chunk's metadata should not affect others
        chunker = FixedChunker(chunk_size=20)
        chunks = chunker.chunk(LONG_TEXT)
        original_index = chunks[1]["metadata"]["chunk_index"]
        chunks[0]["metadata"]["chunk_index"] = 999
        assert chunks[1]["metadata"]["chunk_index"] == original_index