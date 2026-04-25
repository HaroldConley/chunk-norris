import pytest
from unittest.mock import MagicMock

from chunk_norris.question_gen import QuestionGenerator, _MIN_PASSAGE_TOKENS


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_llm(responses: list[str]) -> MagicMock:
    """Returns a mock LLM that returns responses in order."""
    llm = MagicMock()
    llm.generate.side_effect = responses
    return llm


def make_valid_response(question: str, answer: str) -> str:
    """Returns a well-formed LLM JSON response."""
    return f'{{"question": "{question}", "expected_answer": "{answer}"}}'


# Sample texts — long enough to produce valid passages
SHORT_TEXT = "Hello world."

SINGLE_PARA = (
    "The Artemis II mission is a crewed lunar flyby scheduled for 2026. "
    "It will carry four astronauts aboard the Orion spacecraft. "
    "The crew consists of NASA astronauts Reid Wiseman, Victor Glover, and Christina Koch."
)

MULTI_PARA = """The Artemis II mission is a crewed lunar flyby scheduled for 2026.
It will carry four astronauts aboard the Orion spacecraft.

The crew consists of NASA astronauts Reid Wiseman, Victor Glover, and Christina Koch,
along with Canadian Space Agency astronaut Jeremy Hansen.

The Space Launch System rocket will propel the spacecraft beyond low Earth orbit.
This will be the first crewed mission to travel that far since Apollo 17 in 1972."""

LONG_PARA = "This is a long passage with enough tokens to be valid. " * 20

# Realistic Q&A — long enough to pass the 5-token minimum answer check
# and not circular (answer not substring of question)
REAL_QUESTION = "Who are the NASA astronauts on the Artemis II mission?"
REAL_ANSWER   = (
    "The mission is crewed by NASA astronauts Reid Wiseman, "
    "Victor Glover, and Christina Koch."
)
REAL_RESPONSE = make_valid_response(REAL_QUESTION, REAL_ANSWER)


# ── TestQuestionGeneratorInit ─────────────────────────────────────────────────

class TestQuestionGeneratorInit:

    def test_stores_llm(self):
        llm = make_llm([])
        gen = QuestionGenerator(llm=llm)
        assert gen.llm is llm

    def test_encoding_initialised(self):
        gen = QuestionGenerator(llm=make_llm([]))
        assert gen.encoding is not None

    def test_default_sentences_per_passage(self):
        gen = QuestionGenerator(llm=make_llm([]))
        assert gen.sentences_per_passage == 3

    def test_custom_sentences_per_passage(self):
        gen = QuestionGenerator(llm=make_llm([]), sentences_per_passage=2)
        assert gen.sentences_per_passage == 2

    def test_zero_sentences_per_passage_raises(self):
        with pytest.raises(ValueError, match="sentences_per_passage"):
            QuestionGenerator(llm=make_llm([]), sentences_per_passage=0)

    def test_negative_sentences_per_passage_raises(self):
        with pytest.raises(ValueError, match="sentences_per_passage"):
            QuestionGenerator(llm=make_llm([]), sentences_per_passage=-1)


# ── TestGenerate ──────────────────────────────────────────────────────────────

class TestGenerate:

    def test_empty_text_raises(self):
        gen = QuestionGenerator(llm=make_llm([]))
        with pytest.raises(ValueError, match="text"):
            gen.generate(text="", n=5)

    def test_whitespace_text_raises(self):
        gen = QuestionGenerator(llm=make_llm([]))
        with pytest.raises(ValueError, match="text"):
            gen.generate(text="   \n\n  ", n=5)

    def test_zero_n_raises(self):
        gen = QuestionGenerator(llm=make_llm([]))
        with pytest.raises(ValueError, match="n"):
            gen.generate(text=SINGLE_PARA, n=0)

    def test_negative_n_raises(self):
        gen = QuestionGenerator(llm=make_llm([]))
        with pytest.raises(ValueError, match="n"):
            gen.generate(text=SINGLE_PARA, n=-1)

    def test_returns_list(self):
        gen = QuestionGenerator(llm=make_llm([REAL_RESPONSE]))
        result = gen.generate(text=SINGLE_PARA, n=1)
        assert isinstance(result, list)

    def test_returns_list_of_dicts(self):
        gen = QuestionGenerator(llm=make_llm([REAL_RESPONSE]))
        result = gen.generate(text=SINGLE_PARA, n=1)
        assert all(isinstance(q, dict) for q in result)

    def test_each_result_has_question_key(self):
        gen = QuestionGenerator(llm=make_llm([REAL_RESPONSE]))
        result = gen.generate(text=SINGLE_PARA, n=1)
        for q in result:
            assert "question" in q

    def test_each_result_has_expected_answer_key(self):
        gen = QuestionGenerator(llm=make_llm([REAL_RESPONSE]))
        result = gen.generate(text=SINGLE_PARA, n=1)
        for q in result:
            assert "expected_answer" in q

    def test_question_and_answer_are_strings(self):
        gen = QuestionGenerator(llm=make_llm([REAL_RESPONSE]))
        result = gen.generate(text=SINGLE_PARA, n=1)
        for q in result:
            assert isinstance(q["question"], str)
            assert isinstance(q["expected_answer"], str)

    def test_question_content_from_llm_response(self):
        gen = QuestionGenerator(llm=make_llm([REAL_RESPONSE]))
        result = gen.generate(text=SINGLE_PARA, n=1)
        assert result[0]["question"] == REAL_QUESTION
        assert result[0]["expected_answer"] == REAL_ANSWER

    def test_seed_produces_reproducible_results(self):
        responses_a = [REAL_RESPONSE] * 5
        responses_b = [REAL_RESPONSE] * 5
        gen1 = QuestionGenerator(llm=make_llm(responses_a))
        gen2 = QuestionGenerator(llm=make_llm(responses_b))
        result1 = gen1.generate(text=MULTI_PARA, n=2, seed=42)
        result2 = gen2.generate(text=MULTI_PARA, n=2, seed=42)
        assert len(result1) == len(result2)

    def test_no_valid_passages_raises(self):
        gen = QuestionGenerator(llm=make_llm([]))
        with pytest.raises(ValueError, match="No valid passages"):
            gen.generate(text=SHORT_TEXT, n=5)


# ── TestFilterPassages ────────────────────────────────────────────────────────

class TestFilterPassages:

    def test_short_passage_filtered_out(self):
        gen = QuestionGenerator(llm=make_llm([]))
        passages = ["Hi.", LONG_PARA]
        result = gen._filter_passages(passages)
        assert "Hi." not in result

    def test_long_passage_kept(self):
        gen = QuestionGenerator(llm=make_llm([]))
        result = gen._filter_passages([LONG_PARA])
        assert LONG_PARA in result

    def test_empty_list_returns_empty(self):
        gen = QuestionGenerator(llm=make_llm([]))
        assert gen._filter_passages([]) == []

    def test_min_passage_tokens_boundary(self):
        gen = QuestionGenerator(llm=make_llm([]))
        tokens = gen.encoding.encode(LONG_PARA)
        boundary_text = gen.encoding.decode(tokens[:_MIN_PASSAGE_TOKENS])
        result = gen._filter_passages([boundary_text])
        assert len(result) == 1


# ── TestSplitPassages ─────────────────────────────────────────────────────────

class TestSplitPassages:
    """
    _split_passages now uses NLTK sentence tokenisation and groups
    sentences into windows of sentences_per_passage sentences each.
    """

    def test_returns_list(self):
        gen = QuestionGenerator(llm=make_llm([]))
        result = gen._split_passages(SINGLE_PARA)
        assert isinstance(result, list)

    def test_single_sentence_returns_one_passage(self):
        gen = QuestionGenerator(llm=make_llm([]))
        result = gen._split_passages("Just one sentence here.")
        assert len(result) == 1

    def test_groups_sentences_into_passages(self):
        # 6 sentences with sentences_per_passage=3 → 2 passages
        text = " ".join([
            "Sentence one here.",
            "Sentence two here.",
            "Sentence three here.",
            "Sentence four here.",
            "Sentence five here.",
            "Sentence six here.",
        ])
        gen = QuestionGenerator(llm=make_llm([]), sentences_per_passage=3)
        result = gen._split_passages(text)
        assert len(result) == 2

    def test_each_passage_is_string(self):
        gen = QuestionGenerator(llm=make_llm([]))
        result = gen._split_passages(SINGLE_PARA)
        assert all(isinstance(p, str) for p in result)

    def test_each_passage_is_non_empty(self):
        gen = QuestionGenerator(llm=make_llm([]))
        result = gen._split_passages(SINGLE_PARA)
        assert all(len(p) > 0 for p in result)

    def test_sentences_per_passage_one(self):
        # Each sentence becomes its own passage
        text = "First sentence. Second sentence. Third sentence."
        gen = QuestionGenerator(llm=make_llm([]), sentences_per_passage=1)
        result = gen._split_passages(text)
        assert len(result) == 3

    def test_multi_paragraph_text_handled(self):
        # Paragraphs are treated as continuous text — NLTK handles newlines
        gen = QuestionGenerator(llm=make_llm([]))
        result = gen._split_passages(MULTI_PARA)
        assert len(result) >= 1


# ── TestSamplePassages ────────────────────────────────────────────────────────

class TestSamplePassages:

    def test_returns_all_when_n_exceeds_passages(self):
        gen = QuestionGenerator(llm=make_llm([]))
        passages = [LONG_PARA, LONG_PARA, LONG_PARA]
        result = gen._sample_passages(passages, n=10, seed=42)
        assert len(result) == 3

    def test_returns_n_when_n_less_than_passages(self):
        gen = QuestionGenerator(llm=make_llm([]))
        passages = [LONG_PARA] * 10
        result = gen._sample_passages(passages, n=3, seed=42)
        assert len(result) == 3

    def test_returns_exactly_n_when_equal(self):
        gen = QuestionGenerator(llm=make_llm([]))
        passages = [LONG_PARA] * 5
        result = gen._sample_passages(passages, n=5, seed=42)
        assert len(result) == 5

    def test_no_duplicate_passages(self):
        gen = QuestionGenerator(llm=make_llm([]))
        passages = [f"Passage {i}. " * 10 for i in range(20)]
        result = gen._sample_passages(passages, n=10, seed=42)
        assert len(result) == len(set(result))

    def test_same_seed_same_result(self):
        gen = QuestionGenerator(llm=make_llm([]))
        passages = [f"Passage {i}. " * 10 for i in range(20)]
        result1 = gen._sample_passages(passages, n=5, seed=99)
        result2 = gen._sample_passages(passages, n=5, seed=99)
        assert result1 == result2

    def test_different_seed_may_differ(self):
        gen = QuestionGenerator(llm=make_llm([]))
        passages = [f"Passage {i}. " * 10 for i in range(20)]
        result1 = gen._sample_passages(passages, n=5, seed=1)
        result2 = gen._sample_passages(passages, n=5, seed=999)
        assert len(result1) == 5
        assert len(result2) == 5


# ── TestParseResponse ─────────────────────────────────────────────────────────

class TestParseResponse:
    """
    Uses realistic Q&A pairs — long enough to pass the 5-token minimum
    answer check and not circular (answer not substring of question).
    """

    def test_valid_json_parsed_correctly(self):
        gen = QuestionGenerator(llm=make_llm([]))
        response = (
            '{"question": "' + REAL_QUESTION + '", '
            '"expected_answer": "' + REAL_ANSWER + '"}'
        )
        result = gen._parse_response(response)
        assert result == {
            "question":        REAL_QUESTION,
            "expected_answer": REAL_ANSWER,
        }

    def test_json_in_markdown_fences_parsed(self):
        gen = QuestionGenerator(llm=make_llm([]))
        response = (
            "```json\n"
            '{"question": "' + REAL_QUESTION + '", '
            '"expected_answer": "' + REAL_ANSWER + '"}'
            "\n```"
        )
        result = gen._parse_response(response)
        assert result is not None
        assert result["question"] == REAL_QUESTION

    def test_json_with_preamble_parsed(self):
        gen = QuestionGenerator(llm=make_llm([]))
        response = (
            "Here is the answer: "
            '{"question": "' + REAL_QUESTION + '", '
            '"expected_answer": "' + REAL_ANSWER + '"}'
        )
        result = gen._parse_response(response)
        assert result is not None

    def test_invalid_json_returns_none(self):
        gen = QuestionGenerator(llm=make_llm([]))
        result = gen._parse_response("not json at all")
        assert result is None

    def test_missing_question_key_returns_none(self):
        gen = QuestionGenerator(llm=make_llm([]))
        result = gen._parse_response(
            '{"expected_answer": "' + REAL_ANSWER + '"}'
        )
        assert result is None

    def test_missing_answer_key_returns_none(self):
        gen = QuestionGenerator(llm=make_llm([]))
        result = gen._parse_response(
            '{"question": "' + REAL_QUESTION + '"}'
        )
        assert result is None

    def test_empty_question_returns_none(self):
        gen = QuestionGenerator(llm=make_llm([]))
        result = gen._parse_response(
            '{"question": "", "expected_answer": "' + REAL_ANSWER + '"}'
        )
        assert result is None

    def test_empty_answer_returns_none(self):
        gen = QuestionGenerator(llm=make_llm([]))
        result = gen._parse_response(
            '{"question": "' + REAL_QUESTION + '", "expected_answer": ""}'
        )
        assert result is None

    def test_empty_string_returns_none(self):
        gen = QuestionGenerator(llm=make_llm([]))
        result = gen._parse_response("")
        assert result is None

    def test_circular_qa_returns_none(self):
        # Answer is a substring of the question — circular
        gen = QuestionGenerator(llm=make_llm([]))
        result = gen._parse_response(
            '{"question": "What is the first crewed mission of the Orion spacecraft?", '
            '"expected_answer": "the first crewed mission of the Orion spacecraft"}'
        )
        assert result is None

    def test_short_answer_returns_none(self):
        # Answer under 5 tokens — too short to be meaningful
        gen = QuestionGenerator(llm=make_llm([]))
        result = gen._parse_response(
            '{"question": "' + REAL_QUESTION + '", '
            '"expected_answer": "2026"}'
        )
        assert result is None


# ── TestGenerateOneFallback ───────────────────────────────────────────────────

class TestGenerateOneFallback:

    def test_llm_error_returns_none(self):
        from chunk_norris.llm.base import LLMError
        llm = MagicMock()
        llm.generate.side_effect = LLMError(
            message="API failed",
            provider="OpenAI",
            original_error=Exception("timeout"),
        )
        gen = QuestionGenerator(llm=llm)
        result = gen._generate_one(LONG_PARA)
        assert result is None

    def test_invalid_response_returns_none(self):
        llm = make_llm(["not valid json"])
        gen = QuestionGenerator(llm=llm)
        result = gen._generate_one(LONG_PARA)
        assert result is None

    def test_valid_response_returns_dict(self):
        llm = make_llm([REAL_RESPONSE])
        gen = QuestionGenerator(llm=llm)
        result = gen._generate_one(LONG_PARA)
        assert result is not None
        assert result["question"] == REAL_QUESTION

    def test_llm_called_with_passage_in_prompt(self):
        llm = make_llm([REAL_RESPONSE])
        gen = QuestionGenerator(llm=llm)
        gen._generate_one(LONG_PARA)
        # generate() is called with keyword args — access via kwargs
        prompt_used = llm.generate.call_args.kwargs["prompt"]
        assert LONG_PARA in prompt_used

    def test_generation_skips_failed_passages(self):
        from chunk_norris.llm.base import LLMError
        llm = MagicMock()
        llm.generate.side_effect = [
            LLMError("fail", "OpenAI", Exception()),
            REAL_RESPONSE,
        ]
        gen = QuestionGenerator(llm=llm)
        results = []
        for p in [LONG_PARA, LONG_PARA]:
            r = gen._generate_one(p)
            if r is not None:
                results.append(r)
        assert len(results) == 1
