import pytest

from chunk_norris.chunkers.paragraph import ParagraphChunker


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_text(*paragraphs: str) -> str:
    """Joins paragraphs with double newlines."""
    return "\n\n".join(paragraphs)


SHORT = "The quick brown fox jumps over the lazy dog."
MEDIUM = " ".join(["This is a medium length sentence."] * 10)
LONG   = " ".join(["This is a fairly long sentence that will use up tokens."] * 30)


# ── TestParagraphChunkerInit ──────────────────────────────────────────────────

class TestParagraphChunkerInit:

    def test_default_max_tokens(self):
        chunker = ParagraphChunker()
        assert chunker.max_tokens == 512

    def test_custom_max_tokens_stored(self):
        chunker = ParagraphChunker(max_tokens=256)
        assert chunker.max_tokens == 256

    def test_none_max_tokens_stored(self):
        chunker = ParagraphChunker(max_tokens=None)
        assert chunker.max_tokens is None

    def test_zero_max_tokens_raises(self):
        with pytest.raises(ValueError, match="max_tokens"):
            ParagraphChunker(max_tokens=0)

    def test_negative_max_tokens_raises(self):
        with pytest.raises(ValueError, match="max_tokens"):
            ParagraphChunker(max_tokens=-10)

    def test_repr_with_max_tokens(self):
        chunker = ParagraphChunker(max_tokens=256)
        assert repr(chunker) == "ParagraphChunker(max_tokens=256)"

    def test_repr_with_none(self):
        chunker = ParagraphChunker(max_tokens=None)
        assert repr(chunker) == "ParagraphChunker(max_tokens=None)"


# ── TestEdgeCases ─────────────────────────────────────────────────────────────

class TestParagraphChunkerEdgeCases:

    def test_empty_string_returns_empty_list(self):
        chunker = ParagraphChunker()
        assert chunker.chunk("") == []

    def test_whitespace_only_returns_empty_list(self):
        chunker = ParagraphChunker()
        assert chunker.chunk("   \n\n   ") == []

    def test_single_paragraph_returns_one_chunk(self):
        chunker = ParagraphChunker()
        chunks = chunker.chunk(SHORT)
        assert len(chunks) == 1

    def test_single_paragraph_text_preserved(self):
        chunker = ParagraphChunker()
        chunks = chunker.chunk(SHORT)
        assert chunks[0]["text"] == SHORT

    def test_multiple_blank_lines_treated_as_one_separator(self):
        # Three blank lines between paragraphs should still produce two chunks
        text = "First paragraph.\n\n\n\nSecond paragraph."
        chunker = ParagraphChunker()
        chunks = chunker.chunk(text)
        assert len(chunks) == 2

    def test_leading_and_trailing_whitespace_stripped(self):
        text = "\n\n  First paragraph.  \n\n  Second paragraph.  \n\n"
        chunker = ParagraphChunker()
        chunks = chunker.chunk(text)
        assert chunks[0]["text"] == "First paragraph."
        assert chunks[1]["text"] == "Second paragraph."

    def test_empty_paragraphs_ignored(self):
        # A paragraph with only whitespace should be discarded
        text = "First.\n\n   \n\nSecond."
        chunker = ParagraphChunker()
        chunks = chunker.chunk(text)
        assert len(chunks) == 2


# ── TestNormalBehaviour ───────────────────────────────────────────────────────

class TestParagraphChunkerNormalBehaviour:

    def test_two_paragraphs_produce_two_chunks(self):
        chunker = ParagraphChunker()
        text = make_text("First paragraph.", "Second paragraph.")
        chunks = chunker.chunk(text)
        assert len(chunks) == 2

    def test_three_paragraphs_produce_three_chunks(self):
        chunker = ParagraphChunker()
        text = make_text("First.", "Second.", "Third.")
        chunks = chunker.chunk(text)
        assert len(chunks) == 3

    def test_chunk_texts_match_paragraphs(self):
        chunker = ParagraphChunker()
        text = make_text("Alpha paragraph.", "Beta paragraph.")
        chunks = chunker.chunk(text)
        assert chunks[0]["text"] == "Alpha paragraph."
        assert chunks[1]["text"] == "Beta paragraph."

    def test_chunk_indices_are_sequential(self):
        chunker = ParagraphChunker()
        text = make_text("One.", "Two.", "Three.", "Four.")
        chunks = chunker.chunk(text)
        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["chunk_index"] == i

    def test_paragraph_indices_are_sequential(self):
        chunker = ParagraphChunker()
        text = make_text("One.", "Two.", "Three.")
        chunks = chunker.chunk(text)
        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["paragraph_index"] == i

    def test_token_count_is_positive(self):
        chunker = ParagraphChunker()
        chunks = chunker.chunk(make_text("Hello world.", "Foo bar."))
        for chunk in chunks:
            assert chunk["metadata"]["token_count"] > 0

    def test_was_split_false_for_short_paragraphs(self):
        chunker = ParagraphChunker(max_tokens=512)
        chunks = chunker.chunk(make_text(SHORT, SHORT))
        for chunk in chunks:
            assert chunk["metadata"]["was_split"] is False

    def test_metadata_contains_all_keys(self):
        chunker = ParagraphChunker()
        chunks = chunker.chunk(SHORT)
        expected_keys = {"chunk_index", "token_count", "paragraph_index", "was_split"}
        assert expected_keys == set(chunks[0]["metadata"].keys())


# ── TestMaxTokens ─────────────────────────────────────────────────────────────

class TestParagraphChunkerMaxTokens:

    def test_short_paragraph_not_split(self):
        chunker = ParagraphChunker(max_tokens=512)
        chunks = chunker.chunk(SHORT)
        assert len(chunks) == 1
        assert chunks[0]["metadata"]["was_split"] is False

    def test_long_paragraph_is_split(self):
        # LONG has ~300+ tokens, so max_tokens=50 will force splitting
        chunker = ParagraphChunker(max_tokens=50)
        chunks = chunker.chunk(LONG)
        assert len(chunks) > 1

    def test_split_chunks_marked_was_split_true(self):
        chunker = ParagraphChunker(max_tokens=50)
        chunks = chunker.chunk(LONG)
        for chunk in chunks:
            assert chunk["metadata"]["was_split"] is True

    def test_split_chunks_respect_max_tokens(self):
        chunker = ParagraphChunker(max_tokens=50)
        chunks = chunker.chunk(LONG)
        for chunk in chunks:
            assert chunk["metadata"]["token_count"] <= 50 + 20  # 20 token tolerance for sentence boundaries

    def test_none_max_tokens_no_splitting(self):
        # With no size cap, even a very long paragraph stays as one chunk
        chunker = ParagraphChunker(max_tokens=None)
        chunks = chunker.chunk(LONG)
        assert len(chunks) == 1
        assert chunks[0]["metadata"]["was_split"] is False

    def test_split_chunks_share_paragraph_index(self):
        # All pieces from the same paragraph should have the same paragraph_index
        chunker = ParagraphChunker(max_tokens=50)
        chunks = chunker.chunk(LONG)
        para_indices = {chunk["metadata"]["paragraph_index"] for chunk in chunks}
        assert para_indices == {0}  # all from paragraph 0

    def test_split_chunks_have_sequential_chunk_indices(self):
        chunker = ParagraphChunker(max_tokens=50)
        chunks = chunker.chunk(LONG)
        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["chunk_index"] == i

    def test_mixed_short_and_long_paragraphs(self):
        # Short paragraph stays whole, long gets split
        text = make_text(SHORT, LONG, SHORT)
        chunker = ParagraphChunker(max_tokens=50)
        chunks = chunker.chunk(text)

        # First chunk: SHORT paragraph, not split
        assert chunks[0]["metadata"]["was_split"] is False
        assert chunks[0]["metadata"]["paragraph_index"] == 0

        # Middle chunks: LONG paragraph, split
        middle_chunks = [c for c in chunks if c["metadata"]["paragraph_index"] == 1]
        assert len(middle_chunks) > 1
        for c in middle_chunks:
            assert c["metadata"]["was_split"] is True

        # Last chunk: SHORT paragraph, not split
        assert chunks[-1]["metadata"]["was_split"] is False
        assert chunks[-1]["metadata"]["paragraph_index"] == 2

    def test_chunk_indices_sequential_across_mixed_paragraphs(self):
        text = make_text(SHORT, LONG, SHORT)
        chunker = ParagraphChunker(max_tokens=50)
        chunks = chunker.chunk(text)
        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["chunk_index"] == i


# ── TestSplitParagraphs ───────────────────────────────────────────────────────

class TestSplitParagraphs:
    """Tests the internal _split_paragraphs helper directly."""

    def test_splits_on_double_newline(self):
        chunker = ParagraphChunker()
        result = chunker._split_paragraphs("First.\n\nSecond.")
        assert result == ["First.", "Second."]

    def test_splits_on_multiple_blank_lines(self):
        chunker = ParagraphChunker()
        result = chunker._split_paragraphs("First.\n\n\n\nSecond.")
        assert result == ["First.", "Second."]

    def test_strips_whitespace_from_paragraphs(self):
        chunker = ParagraphChunker()
        result = chunker._split_paragraphs("  First.  \n\n  Second.  ")
        assert result == ["First.", "Second."]

    def test_discards_empty_paragraphs(self):
        chunker = ParagraphChunker()
        result = chunker._split_paragraphs("First.\n\n   \n\nSecond.")
        assert result == ["First.", "Second."]

    def test_single_paragraph_returns_list_of_one(self):
        chunker = ParagraphChunker()
        result = chunker._split_paragraphs("Just one paragraph.")
        assert result == ["Just one paragraph."]
