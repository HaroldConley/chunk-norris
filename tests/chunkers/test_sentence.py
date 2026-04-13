import pytest

from chunk_norris.chunkers.sentence import SentenceChunker


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_text(*sentences: str) -> str:
    """Joins sentences with a space."""
    return " ".join(sentences)


S1 = "The quick brown fox jumps over the lazy dog."
S2 = "She sells seashells by the seashore."
S3 = "How much wood would a woodchuck chuck?"
S4 = "Peter Piper picked a peck of pickled peppers."
S5 = "All that glitters is not gold."
S6 = "To be or not to be, that is the question."
S7 = "The early bird catches the worm."

FIVE_SENTENCES  = make_text(S1, S2, S3, S4, S5)
SEVEN_SENTENCES = make_text(S1, S2, S3, S4, S5, S6, S7)


# ── TestSentenceChunkerInit ───────────────────────────────────────────────────

class TestSentenceChunkerInit:

    def test_default_values(self):
        chunker = SentenceChunker()
        assert chunker.sentences_per_chunk == 5
        assert chunker.overlap == 0

    def test_custom_values_stored(self):
        chunker = SentenceChunker(sentences_per_chunk=3, overlap=1)
        assert chunker.sentences_per_chunk == 3
        assert chunker.overlap == 1

    # --- sentences_per_chunk errors ---

    def test_zero_sentences_per_chunk_raises(self):
        with pytest.raises(ValueError, match="sentences_per_chunk"):
            SentenceChunker(sentences_per_chunk=0)

    def test_negative_sentences_per_chunk_raises(self):
        with pytest.raises(ValueError, match="sentences_per_chunk"):
            SentenceChunker(sentences_per_chunk=-1)

    # --- overlap errors ---

    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            SentenceChunker(overlap=-1)

    def test_overlap_equal_to_sentences_per_chunk_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            SentenceChunker(sentences_per_chunk=3, overlap=3)

    def test_overlap_greater_than_sentences_per_chunk_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            SentenceChunker(sentences_per_chunk=3, overlap=5)

    # --- valid boundary values ---

    def test_overlap_zero_is_valid(self):
        chunker = SentenceChunker(overlap=0)
        assert chunker.overlap == 0

    def test_overlap_one_less_than_sentences_per_chunk_is_valid(self):
        chunker = SentenceChunker(sentences_per_chunk=3, overlap=2)
        assert chunker.overlap == 2

    def test_sentences_per_chunk_one_is_valid(self):
        chunker = SentenceChunker(sentences_per_chunk=1)
        assert chunker.sentences_per_chunk == 1

    # --- repr ---

    def test_repr(self):
        chunker = SentenceChunker(sentences_per_chunk=3, overlap=1)
        assert repr(chunker) == "SentenceChunker(sentences_per_chunk=3, overlap=1)"

    def test_repr_default_values(self):
        chunker = SentenceChunker()
        assert repr(chunker) == "SentenceChunker(sentences_per_chunk=5, overlap=0)"


# ── TestEdgeCases ─────────────────────────────────────────────────────────────

class TestSentenceChunkerEdgeCases:

    def test_empty_string_returns_empty_list(self):
        chunker = SentenceChunker()
        assert chunker.chunk("") == []

    def test_whitespace_only_returns_empty_list(self):
        chunker = SentenceChunker()
        assert chunker.chunk("   \n\n   ") == []

    def test_single_sentence_returns_one_chunk(self):
        chunker = SentenceChunker(sentences_per_chunk=5)
        chunks = chunker.chunk(S1)
        assert len(chunks) == 1

    def test_fewer_sentences_than_chunk_size_returns_one_chunk(self):
        chunker = SentenceChunker(sentences_per_chunk=10)
        chunks = chunker.chunk(FIVE_SENTENCES)
        assert len(chunks) == 1

    def test_exactly_chunk_size_returns_one_chunk(self):
        chunker = SentenceChunker(sentences_per_chunk=5)
        chunks = chunker.chunk(FIVE_SENTENCES)
        assert len(chunks) == 1


# ── TestNormalBehaviour ───────────────────────────────────────────────────────

class TestSentenceChunkerNormalBehaviour:

    def test_returns_list_of_dicts(self):
        chunker = SentenceChunker(sentences_per_chunk=2)
        chunks = chunker.chunk(FIVE_SENTENCES)
        assert isinstance(chunks, list)
        assert all(isinstance(c, dict) for c in chunks)

    def test_each_chunk_has_text_and_metadata(self):
        chunker = SentenceChunker(sentences_per_chunk=2)
        chunks = chunker.chunk(FIVE_SENTENCES)
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk

    def test_chunk_indices_are_sequential(self):
        chunker = SentenceChunker(sentences_per_chunk=2)
        chunks = chunker.chunk(FIVE_SENTENCES)
        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["chunk_index"] == i

    def test_metadata_contains_all_keys(self):
        chunker = SentenceChunker(sentences_per_chunk=2)
        chunks = chunker.chunk(FIVE_SENTENCES)
        expected_keys = {
            "chunk_index",
            "token_count",
            "sentence_count",
            "start_sentence",
            "end_sentence",
            "overlap",
        }
        for chunk in chunks:
            assert expected_keys == set(chunk["metadata"].keys())

    def test_token_count_is_positive(self):
        chunker = SentenceChunker(sentences_per_chunk=2)
        chunks = chunker.chunk(FIVE_SENTENCES)
        for chunk in chunks:
            assert chunk["metadata"]["token_count"] > 0

    def test_sentence_count_per_chunk(self):
        chunker = SentenceChunker(sentences_per_chunk=2)
        chunks = chunker.chunk(FIVE_SENTENCES)
        # All chunks except possibly the last should have 2 sentences
        for chunk in chunks[:-1]:
            assert chunk["metadata"]["sentence_count"] == 2

    def test_last_chunk_may_have_fewer_sentences(self):
        # 5 sentences with chunks of 2 → [2, 2, 1]
        chunker = SentenceChunker(sentences_per_chunk=2)
        chunks = chunker.chunk(FIVE_SENTENCES)
        assert chunks[-1]["metadata"]["sentence_count"] <= 2

    def test_start_and_end_sentence_consistent(self):
        chunker = SentenceChunker(sentences_per_chunk=2)
        chunks = chunker.chunk(FIVE_SENTENCES)
        for chunk in chunks:
            meta = chunk["metadata"]
            expected_count = meta["end_sentence"] - meta["start_sentence"]
            assert meta["sentence_count"] == expected_count

    def test_first_chunk_starts_at_sentence_zero(self):
        chunker = SentenceChunker(sentences_per_chunk=3)
        chunks = chunker.chunk(FIVE_SENTENCES)
        assert chunks[0]["metadata"]["start_sentence"] == 0

    def test_overlap_value_stored_in_metadata(self):
        chunker = SentenceChunker(sentences_per_chunk=3, overlap=1)
        chunks = chunker.chunk(SEVEN_SENTENCES)
        for chunk in chunks:
            assert chunk["metadata"]["overlap"] == 1

    def test_chunk_text_is_non_empty_string(self):
        chunker = SentenceChunker(sentences_per_chunk=2)
        chunks = chunker.chunk(FIVE_SENTENCES)
        for chunk in chunks:
            assert isinstance(chunk["text"], str)
            assert len(chunk["text"]) > 0

    def test_correct_number_of_chunks_no_overlap(self):
        # 7 sentences, chunk_size=2, no overlap → ceil(7/2) = 4 chunks
        chunker = SentenceChunker(sentences_per_chunk=2, overlap=0)
        chunks = chunker.chunk(SEVEN_SENTENCES)
        assert len(chunks) == 4

    def test_correct_number_of_chunks_with_overlap(self):
        # 7 sentences, chunk_size=3, overlap=1 → step=2
        # starts: 0, 2, 4, 6 → 4 chunks
        chunker = SentenceChunker(sentences_per_chunk=3, overlap=1)
        chunks = chunker.chunk(SEVEN_SENTENCES)
        assert len(chunks) == 4


# ── TestOverlapBehaviour ──────────────────────────────────────────────────────

class TestSentenceChunkerOverlapBehaviour:

    def test_no_overlap_chunks_do_not_share_sentences(self):
        chunker = SentenceChunker(sentences_per_chunk=2, overlap=0)
        chunks = chunker.chunk(FIVE_SENTENCES)
        # With no overlap, start of chunk[i+1] = end of chunk[i]
        for i in range(len(chunks) - 1):
            assert chunks[i + 1]["metadata"]["start_sentence"] == \
                   chunks[i]["metadata"]["end_sentence"]

    def test_overlap_advances_window_correctly(self):
        # sentences_per_chunk=3, overlap=1 → step=2
        chunker = SentenceChunker(sentences_per_chunk=3, overlap=1)
        chunks = chunker.chunk(SEVEN_SENTENCES)
        for i in range(len(chunks) - 1):
            step = chunks[i + 1]["metadata"]["start_sentence"] - \
                   chunks[i]["metadata"]["start_sentence"]
            assert step == 2  # sentences_per_chunk - overlap = 3 - 1

    def test_overlapping_chunks_share_sentences(self):
        # With overlap=1, last sentence of chunk[i] = first of chunk[i+1]
        chunker = SentenceChunker(sentences_per_chunk=3, overlap=1)
        chunks = chunker.chunk(SEVEN_SENTENCES)
        for i in range(len(chunks) - 1):
            end_of_current   = chunks[i]["metadata"]["end_sentence"]
            start_of_next    = chunks[i + 1]["metadata"]["start_sentence"]
            shared_sentences = end_of_current - start_of_next
            assert shared_sentences == 1

    def test_overlap_two_sentences(self):
        # sentences_per_chunk=4, overlap=2 → step=2
        # Only check non-final chunks — the last chunk may have fewer
        # sentences than sentences_per_chunk, making shared < 2
        chunker = SentenceChunker(sentences_per_chunk=4, overlap=2)
        chunks = chunker.chunk(SEVEN_SENTENCES)
        for i in range(len(chunks) - 1):
            # Skip if the next chunk is the last and has fewer sentences
            if chunks[i + 1]["metadata"]["sentence_count"] < 2:
                continue
            end_of_current = chunks[i]["metadata"]["end_sentence"]
            start_of_next  = chunks[i + 1]["metadata"]["start_sentence"]
            shared         = end_of_current - start_of_next
            assert shared == 2


# ── TestNLTKSentenceHandling ──────────────────────────────────────────────────

class TestNLTKSentenceHandling:
    """
    Verifies that NLTK handles real-world sentence splitting edge cases
    correctly — abbreviations, multiple punctuation, etc.
    """

    def test_abbreviations_not_split(self):
        # "Dr." should not be treated as a sentence boundary
        text = "Dr. Smith went to Washington. He was a good man."
        chunker = SentenceChunker(sentences_per_chunk=5)
        chunks = chunker.chunk(text)
        # Should produce one chunk containing both sentences
        assert len(chunks) == 1
        assert chunks[0]["metadata"]["sentence_count"] == 2

    def test_multiple_sentences_correctly_counted(self):
        chunker = SentenceChunker(sentences_per_chunk=10)
        chunks = chunker.chunk(SEVEN_SENTENCES)
        assert chunks[0]["metadata"]["sentence_count"] == 7

    def test_sentences_per_chunk_one_produces_one_chunk_per_sentence(self):
        chunker = SentenceChunker(sentences_per_chunk=1, overlap=0)
        chunks = chunker.chunk(make_text(S1, S2, S3))
        assert len(chunks) == 3
        for chunk in chunks:
            assert chunk["metadata"]["sentence_count"] == 1
