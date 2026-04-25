import json
from pathlib import Path

from dotenv import load_dotenv  # python-dotenv==1.2.2

from chunk_norris import (
    Norris,
    BertEmbedder,
    FixedChunker,
    ParagraphChunker,
    SentenceChunker,
    RecursiveChunker,
)

# Load environment variables from .env file.
# Copy .env.example to .env and fill in your API keys.
load_dotenv()


# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Use sample_data/ for the built-in example (committed to the repo).
# Switch to data/ for your own document and questions (gitignored, stays local).
DATA_DIR = BASE_DIR / "sample_data"
# DATA_DIR = BASE_DIR / "data"


# ── Load document ─────────────────────────────────────────────────────────────

with open(DATA_DIR / "document.txt", encoding="utf-8") as f:
    TEXT = f.read()


# ── Questions ─────────────────────────────────────────────────────────────────

# Option A — auto-generate using an LLM (requires OPENAI_API_KEY in .env)
from chunk_norris.llm.openai_llm import OpenAILLM
norris_gen = Norris(embedder=BertEmbedder())
llm = OpenAILLM(model="gpt-4o-mini-2024-07-18")
QUESTIONS = norris_gen.generate_questions(text=TEXT, llm=llm, n=20)
# Optionally save for reuse:
with open(DATA_DIR / "questions.json", "w", encoding="utf-8") as f:
    json.dump(QUESTIONS, f, indent=2)

# Option B — load from file (no LLM needed)
# with open(DATA_DIR / "questions.json", encoding="utf-8") as f:
#     QUESTIONS = json.load(f)


# ── Run ───────────────────────────────────────────────────────────────────────

norris = Norris(
    embedder=BertEmbedder(),
    top_k=3,
    # Recall threshold — fraction of expected answer tokens that must appear
    # in a chunk for it to be considered relevant:
    #   0.50  loose — useful when answers may be paraphrased
    #   0.75  default — chunk must contain 75% of answer tokens
    #   0.90  strict — chunk must contain almost all answer tokens
    recall_threshold=0.75,
)

report = norris.run(
    text=TEXT,
    chunkers=[
        # Fixed size — baseline, split every N tokens
        FixedChunker(chunk_size=128, overlap=0.1),
        FixedChunker(chunk_size=256, overlap=0.1),
        FixedChunker(chunk_size=512, overlap=0.1),

        # Paragraph — respects natural document structure
        ParagraphChunker(max_tokens=256),
        ParagraphChunker(max_tokens=512),

        # Sentence — preserves sentence boundaries
        SentenceChunker(sentences_per_chunk=3, overlap=0),
        SentenceChunker(sentences_per_chunk=5, overlap=1),

        # Recursive — tries paragraph → sentence → word in priority order
        RecursiveChunker(chunk_size=256, overlap=0.1),
        RecursiveChunker(chunk_size=512, overlap=0.1),
    ],
    questions=QUESTIONS,
)


# ── Results ───────────────────────────────────────────────────────────────────

report.compare()

print("\nBest configuration:")
print(report.best())

report.to_json(str(RESULTS_DIR / "results.json"))
report.to_excel(str(RESULTS_DIR / "results.xlsx"))


# ── Pipeline handoff ──────────────────────────────────────────────────────────

# Get the winning chunker instance — ready to use in your RAG pipeline
best_chunker = report.best_chunker()
print(f"\nBest chunker: {best_chunker}")

# Use it to produce the final chunks for your system
final_chunks = best_chunker.chunk(TEXT)
print(f"Final chunks produced: {len(final_chunks)}")

# Next step: pass final_chunks to your vector store or RAG pipeline
# vector_store.add(final_chunks)
