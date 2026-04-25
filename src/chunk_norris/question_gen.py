import json
import random
import re

import nltk
import tiktoken  # tiktoken==0.12.0

from chunk_norris.llm.base import BaseLLM, LLMError


# ── NLTK data ─────────────────────────────────────────────────────────────────

def _ensure_nltk_data() -> None:
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


_ensure_nltk_data()


# ── Prompt ────────────────────────────────────────────────────────────────────

_GENERATION_PROMPT = """You are a precise question-answer generator for RAG evaluation.
You will be given a passage.
You must follow this internal process:

STEP 1 — FACT EXTRACTION
Extract all atomic facts from the passage.
An atomic fact is:
- One self-contained piece of information
- One event, entity relation, value, or action
- Minimal and non-composite
Rules:
- Do NOT merge multiple facts
- Do NOT infer or add external knowledge
- Stay close to the passage wording

STEP 2 — FACT SELECTION
Select ONE atomic fact that is:
- Specific
- Clearly supported by the passage
- Self-contained (not dependent on other facts)
- Prefer facts about people, processes, technical specifications,
  causal relationships, or outcomes over dates and times
- Only select a date or time fact if it is the single most significant
  piece of information in the passage and no other fact type is available

STEP 3 — QUESTION GENERATION
From the selected fact:
- Generate exactly ONE question
- The question must be answerable using ONLY that fact

STEP 4 — ANSWER EXTRACTION
The answer MUST be:
- An exact continuous substring from the ORIGINAL passage
- Copied verbatim (no paraphrasing, no modification, no added words)
- At least one complete sentence where possible
- Long enough to be self-contained and meaningful out of context

IMPORTANT RULES:
- Only ONE question-answer pair must be produced
- Do NOT output the full fact list
- Do NOT output intermediate steps
- Do NOT include explanations
- Do NOT combine multiple facts

PASSAGE:
{passage}

OUTPUT FORMAT (strict JSON only):
{{
    "question": "...",
    "expected_answer": "..."
}}"""

# Minimum token count for a passage to be worth generating a question from.
# Passages shorter than this are likely headers, captions, or single lines
# that don't contain enough information for a meaningful question.
_MIN_PASSAGE_TOKENS = 20

# Number of sentences to group per passage.
# Smaller units → more atomic facts per passage → better question diversity.
# 2-3 sentences is the sweet spot: enough context for a meaningful question,
# small enough to contain roughly one fact.
_SENTENCES_PER_PASSAGE = 3

# Temperature — low enough to stay grounded, high enough for varied questions.
_GENERATION_TEMPERATURE = 0.4
_GENERATION_MAX_TOKENS  = 256


class QuestionGenerator:
    """
    Generates question-answer pairs from a document using an LLM.

    Splits the document into small sentence-based passages (2-3 sentences)
    rather than full paragraphs. This atomicity alignment ensures:

        - Each passage contains roughly one fact
        - The LLM selects from atomic facts, not composite paragraphs
        - Questions are more diverse and less date/entity biased

    Uses a 4-step internal prompt process (fact extraction → fact selection
    → question generation → answer extraction) to decouple fact identification
    from question generation, reducing repetition bias.

    Answers are extracted as exact verbatim substrings from the passage,
    maximising token recall accuracy during evaluation.

    Args:
        llm (BaseLLM): Any LLM implementing BaseLLM. Used to generate
                       questions from each passage.
        sentences_per_passage (int): Number of sentences to group per passage.
                                     Default: 3.
        encoding_name (str): tiktoken encoding for passage token counting.
                             Default: "cl100k_base".

    Example::

        from chunk_norris.llm.openai_llm import OpenAILLM
        from chunk_norris.question_gen import QuestionGenerator

        generator = QuestionGenerator(llm=OpenAILLM())
        questions = generator.generate(text=TEXT, n=20)

        for q in questions:
            print(q["question"])
            print(q["expected_answer"])
    """

    def __init__(
        self,
        llm: BaseLLM,
        sentences_per_passage: int = _SENTENCES_PER_PASSAGE,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if sentences_per_passage <= 0:
            raise ValueError(
                f"sentences_per_passage must be positive, got {sentences_per_passage}"
            )
        self.llm = llm
        self.sentences_per_passage = sentences_per_passage
        self.encoding = tiktoken.get_encoding(encoding_name)

    def generate(
        self,
        text: str,
        n: int = 20,
        seed: int | None = None,
    ) -> list[dict[str, str]]:
        """
        Generates up to n question-answer pairs from the document.

        Splits the document into sentence-based passages, filters out
        passages that are too short, then samples evenly across the
        remaining passages to ensure location diversity.

        If the document has fewer valid passages than n, all passages are
        used and the returned list will have fewer than n items.

        Args:
            text (str): The full document text to generate questions from.
            n (int): Target number of question-answer pairs. Default: 20.
            seed (int | None): Random seed for reproducible passage sampling.
                               Default: None (random).

        Returns:
            list[dict]: A list of question dicts, each containing:
                - "question" (str): The generated question.
                - "expected_answer" (str): Verbatim answer from the passage.

        Raises:
            ValueError: If text is empty or n is not positive.
            LLMError: If an LLM call fails.
        """
        if not text or not text.strip():
            raise ValueError("text must not be empty.")
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

        passages = self._split_passages(text)
        valid_passages = self._filter_passages(passages)

        if not valid_passages:
            raise ValueError(
                "No valid passages found. The document may be too short "
                "or consist entirely of very short sentences."
            )

        sampled = self._sample_passages(valid_passages, n=n, seed=seed)

        questions = []
        for i, passage in enumerate(sampled, start=1):
            print(
                f"\r      Generating question {i}/{len(sampled)}...",
                end="",
                flush=True,
            )
            result = self._generate_one(passage)
            if result is not None:
                questions.append(result)

        print()
        return questions

    def _split_passages(self, text: str) -> list[str]:
        """
        Splits text into small sentence-based passages.

        Uses NLTK sentence tokenisation to split the document into
        individual sentences, then groups them into passages of
        sentences_per_passage sentences each.

        Smaller passage units → passages contain fewer facts → LLM
        generates more atomic, diverse questions.
        """
        sentences = nltk.sent_tokenize(text)
        passages = []
        for i in range(0, len(sentences), self.sentences_per_passage):
            group = sentences[i:i + self.sentences_per_passage]
            passage = " ".join(group).strip()
            if passage:
                passages.append(passage)
        return passages

    def _filter_passages(self, passages: list[str]) -> list[str]:
        """
        Filters out passages that are too short for meaningful Q&A generation.
        """
        return [
            p for p in passages
            if len(self.encoding.encode(p)) >= _MIN_PASSAGE_TOKENS
        ]

    def _sample_passages(
        self,
        passages: list[str],
        n: int,
        seed: int | None,
    ) -> list[str]:
        """
        Samples up to n passages, distributed evenly across the document.

        Even sampling ensures question diversity across the full document
        rather than clustering around the first or most prominent sections.
        If n >= len(passages), all passages are returned in order.
        """
        if n >= len(passages):
            return passages

        rng = random.Random(seed)
        step = len(passages) / n
        indices = [int(i * step) for i in range(n)]
        jittered = [
            min(int(idx + rng.uniform(-step / 4, step / 4)), len(passages) - 1)
            for idx in indices
        ]
        seen: set[int] = set()
        unique = []
        for idx in jittered:
            if idx not in seen:
                seen.add(idx)
                unique.append(idx)

        return [passages[i] for i in sorted(unique)]

    def _generate_one(
        self, passage: str
    ) -> dict[str, str] | None:
        """
        Asks the LLM to generate one question-answer pair for a passage.

        Returns None if the LLM response cannot be parsed as valid JSON
        with the expected keys — skips the passage rather than crashing.
        """
        prompt = _GENERATION_PROMPT.format(passage=passage)
        try:
            response = self.llm.generate(
                prompt=prompt,
                temperature=_GENERATION_TEMPERATURE,
                max_tokens=_GENERATION_MAX_TOKENS,
            )
            return self._parse_response(response)
        except LLMError:
            return None

    def _parse_response(self, response: str) -> dict[str, str] | None:
        """
        Parses the LLM response as JSON and validates the required keys.

        Handles cases where the LLM wraps the JSON in markdown code fences
        or adds surrounding text. Returns None if parsing fails.

        Also rejects Q&A pairs that are:
        - Circular (answer is a substring of the question or vice versa)
        - Too short (answer is fewer than 5 tokens — likely a fragment)
        """
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", response).strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return None

        if "question" not in data or "expected_answer" not in data:
            return None

        question = str(data["question"]).strip()
        answer   = str(data["expected_answer"]).strip()

        if not question or not answer:
            return None

        # Reject circular Q&A — answer restates the question or vice versa
        if answer.lower() in question.lower() or question.lower() in answer.lower():
            return None

        # Reject fragments — answers shorter than 5 tokens are rarely meaningful
        if len(self.encoding.encode(answer)) < 5:
            return None

        return {
            "question":        question,
            "expected_answer": answer,
        }
