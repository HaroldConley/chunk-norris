import pytest
from unittest.mock import MagicMock, patch

from chunk_norris.norris import Norris
from chunk_norris.chunkers.fixed import FixedChunker
from chunk_norris.evaluator.report import Report


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_embedder() -> MagicMock:
    """
    Returns a mock embedder that produces stable unit vectors.
    Returns a new list on each call so tests don't share state.
    """
    embedder = MagicMock()
    embedder.embed.side_effect = lambda texts: [
        [1.0, 0.0, 0.0] for _ in texts
    ]
    return embedder


SAMPLE_TEXT = """
The refund policy allows customers to return items within 30 days.
Contact support at support@example.com for assistance.
The company was founded in 2010 and operates in 15 countries.
""".strip()

SAMPLE_QUESTIONS = [
    {
        "question": "What is the refund policy?",
        "expected_answer": "Customers can return items within 30 days.",
    },
    {
        "question": "How do I contact support?",
        "expected_answer": "Contact support at support@example.com.",
    },
]


# ── TestNorrisInit ────────────────────────────────────────────────────────────

class TestNorrisInit:

    def test_default_values(self):
        norris = Norris(embedder=make_embedder())
        assert norris.top_k == 3
        assert norris.recall_threshold == 0.75

    def test_custom_values_stored(self):
        norris = Norris(
            embedder=make_embedder(),
            top_k=5,
            recall_threshold=0.9,
        )
        assert norris.top_k == 5
        assert norris.recall_threshold == 0.9

    def test_embedder_stored(self):
        embedder = make_embedder()
        norris = Norris(embedder=embedder)
        assert norris.embedder is embedder

    # --- top_k errors ---

    def test_zero_top_k_raises(self):
        with pytest.raises(ValueError, match="top_k"):
            Norris(embedder=make_embedder(), top_k=0)

    def test_negative_top_k_raises(self):
        with pytest.raises(ValueError, match="top_k"):
            Norris(embedder=make_embedder(), top_k=-1)

    # --- recall_threshold errors ---

    def test_zero_recall_threshold_raises(self):
        with pytest.raises(ValueError, match="recall_threshold"):
            Norris(embedder=make_embedder(), recall_threshold=0.0)

    def test_above_one_recall_threshold_raises(self):
        with pytest.raises(ValueError, match="recall_threshold"):
            Norris(embedder=make_embedder(), recall_threshold=1.1)

    # --- valid boundary values ---

    def test_recall_threshold_one_is_valid(self):
        norris = Norris(embedder=make_embedder(), recall_threshold=1.0)
        assert norris.recall_threshold == 1.0

    def test_recall_threshold_just_above_zero_is_valid(self):
        norris = Norris(embedder=make_embedder(), recall_threshold=0.01)
        assert norris.recall_threshold == 0.01

    def test_top_k_one_is_valid(self):
        norris = Norris(embedder=make_embedder(), top_k=1)
        assert norris.top_k == 1


# ── TestNorrisRunValidation ───────────────────────────────────────────────────

class TestNorrisRunValidation:

    def test_empty_text_raises(self):
        norris = Norris(embedder=make_embedder())
        with pytest.raises(ValueError, match="text"):
            norris.run(
                text="",
                chunkers=[FixedChunker()],
                questions=SAMPLE_QUESTIONS,
            )

    def test_whitespace_text_raises(self):
        norris = Norris(embedder=make_embedder())
        with pytest.raises(ValueError, match="text"):
            norris.run(
                text="   \n\n  ",
                chunkers=[FixedChunker()],
                questions=SAMPLE_QUESTIONS,
            )

    def test_empty_chunkers_raises(self):
        norris = Norris(embedder=make_embedder())
        with pytest.raises(ValueError, match="chunkers"):
            norris.run(
                text=SAMPLE_TEXT,
                chunkers=[],
                questions=SAMPLE_QUESTIONS,
            )

    def test_empty_questions_raises(self):
        norris = Norris(embedder=make_embedder())
        with pytest.raises(ValueError, match="questions"):
            norris.run(
                text=SAMPLE_TEXT,
                chunkers=[FixedChunker()],
                questions=[],
            )


# ── TestNorrisRunOutput ───────────────────────────────────────────────────────

class TestNorrisRunOutput:

    def test_returns_report(self):
        norris = Norris(embedder=make_embedder())
        report = norris.run(
            text=SAMPLE_TEXT,
            chunkers=[FixedChunker(chunk_size=64)],
            questions=SAMPLE_QUESTIONS,
        )
        assert isinstance(report, Report)

    def test_report_has_one_experiment_per_chunker(self):
        norris = Norris(embedder=make_embedder())
        chunkers = [
            FixedChunker(chunk_size=64),
            FixedChunker(chunk_size=128),
        ]
        report = norris.run(
            text=SAMPLE_TEXT,
            chunkers=chunkers,
            questions=SAMPLE_QUESTIONS,
        )
        assert len(report.experiments) == 2

    def test_each_experiment_has_one_result_per_question(self):
        norris = Norris(embedder=make_embedder())
        report = norris.run(
            text=SAMPLE_TEXT,
            chunkers=[FixedChunker(chunk_size=64)],
            questions=SAMPLE_QUESTIONS,
        )
        for experiment in report.experiments:
            assert len(experiment["results"]) == len(SAMPLE_QUESTIONS)

    def test_experiment_labels_match_chunker_repr(self):
        chunker = FixedChunker(chunk_size=64, overlap=0.1)
        norris = Norris(embedder=make_embedder())
        report = norris.run(
            text=SAMPLE_TEXT,
            chunkers=[chunker],
            questions=SAMPLE_QUESTIONS,
        )
        assert report.experiments[0]["chunker"] == repr(chunker)

    def test_chunker_object_stored_in_experiment(self):
        chunker = FixedChunker(chunk_size=64)
        norris = Norris(embedder=make_embedder())
        report = norris.run(
            text=SAMPLE_TEXT,
            chunkers=[chunker],
            questions=SAMPLE_QUESTIONS,
        )
        assert report.experiments[0]["chunker_object"] is chunker

    def test_each_result_has_question_field(self):
        norris = Norris(embedder=make_embedder())
        report = norris.run(
            text=SAMPLE_TEXT,
            chunkers=[FixedChunker(chunk_size=64)],
            questions=SAMPLE_QUESTIONS,
        )
        for experiment in report.experiments:
            for result in experiment["results"]:
                assert "question" in result

    def test_each_result_has_expected_answer_field(self):
        norris = Norris(embedder=make_embedder())
        report = norris.run(
            text=SAMPLE_TEXT,
            chunkers=[FixedChunker(chunk_size=64)],
            questions=SAMPLE_QUESTIONS,
        )
        for experiment in report.experiments:
            for result in experiment["results"]:
                assert "expected_answer" in result

    def test_each_result_has_scores_field(self):
        norris = Norris(embedder=make_embedder())
        report = norris.run(
            text=SAMPLE_TEXT,
            chunkers=[FixedChunker(chunk_size=64)],
            questions=SAMPLE_QUESTIONS,
        )
        for experiment in report.experiments:
            for result in experiment["results"]:
                assert "scores" in result

    def test_each_result_has_retrieved_chunks_field(self):
        norris = Norris(embedder=make_embedder())
        report = norris.run(
            text=SAMPLE_TEXT,
            chunkers=[FixedChunker(chunk_size=64)],
            questions=SAMPLE_QUESTIONS,
        )
        for experiment in report.experiments:
            for result in experiment["results"]:
                assert "retrieved_chunks" in result

    def test_question_text_preserved_in_results(self):
        norris = Norris(embedder=make_embedder())
        report = norris.run(
            text=SAMPLE_TEXT,
            chunkers=[FixedChunker(chunk_size=64)],
            questions=SAMPLE_QUESTIONS,
        )
        result_questions = [
            r["question"]
            for r in report.experiments[0]["results"]
        ]
        expected_questions = [q["question"] for q in SAMPLE_QUESTIONS]
        assert result_questions == expected_questions

    def test_recall_threshold_passed_to_metrics(self):
        norris = Norris(embedder=make_embedder(), recall_threshold=0.9)
        report = norris.run(
            text=SAMPLE_TEXT,
            chunkers=[FixedChunker(chunk_size=64)],
            questions=SAMPLE_QUESTIONS,
        )
        for experiment in report.experiments:
            for result in experiment["results"]:
                assert result["scores"]["recall_threshold"] == 0.9


# ── TestNorrisPipelineIntegration ─────────────────────────────────────────────

class TestNorrisPipelineIntegration:
    """
    Tests the pipeline handoff — best_chunker() returns a usable object.
    These are integration-level tests: they exercise the full Norris.run()
    flow including Report.best_chunker().
    """

    def test_best_chunker_returns_chunker_instance(self):
        norris = Norris(embedder=make_embedder())
        report = norris.run(
            text=SAMPLE_TEXT,
            chunkers=[FixedChunker(chunk_size=64)],
            questions=SAMPLE_QUESTIONS,
        )
        best = report.best_chunker()
        assert isinstance(best, FixedChunker)

    def test_best_chunker_can_chunk_text(self):
        norris = Norris(embedder=make_embedder())
        report = norris.run(
            text=SAMPLE_TEXT,
            chunkers=[FixedChunker(chunk_size=64)],
            questions=SAMPLE_QUESTIONS,
        )
        best = report.best_chunker()
        chunks = best.chunk(SAMPLE_TEXT)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_best_chunker_is_one_of_input_chunkers(self):
        chunker_a = FixedChunker(chunk_size=64)
        chunker_b = FixedChunker(chunk_size=128)
        norris = Norris(embedder=make_embedder())
        report = norris.run(
            text=SAMPLE_TEXT,
            chunkers=[chunker_a, chunker_b],
            questions=SAMPLE_QUESTIONS,
        )
        best = report.best_chunker()
        assert best is chunker_a or best is chunker_b

    def test_multiple_chunkers_all_present_in_report(self):
        chunkers = [
            FixedChunker(chunk_size=64),
            FixedChunker(chunk_size=128),
            FixedChunker(chunk_size=256),
        ]
        norris = Norris(embedder=make_embedder())
        report = norris.run(
            text=SAMPLE_TEXT,
            chunkers=chunkers,
            questions=SAMPLE_QUESTIONS,
        )
        labels = [e["chunker"] for e in report.experiments]
        for chunker in chunkers:
            assert repr(chunker) in labels
