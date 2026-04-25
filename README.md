# chunk-norris 🥋

**Find the best chunking strategy for your RAG pipeline — empirically, not by gut feel.**

chunk-norris evaluates and compares chunking strategies against your actual document and questions, then hands the winning chunker directly to your pipeline. No guesswork, no boilerplate, no LLM required for evaluation.

```python
from chunk_norris import Norris, BertEmbedder, FixedChunker, RecursiveChunker
from chunk_norris.llm.openai_llm import OpenAILLM

norris = Norris(embedder=BertEmbedder())

# Option A — auto-generate questions (recommended)
questions = norris.generate_questions(
    text=my_document,
    llm=OpenAILLM(model="gpt-4o-mini-2024-07-18"),
    n=20,
)

# Option B — provide your own questions
# questions = [{"question": "...", "expected_answer": "..."}, ...]

report = norris.run(
    text=my_document,
    chunkers=[
        FixedChunker(chunk_size=128, overlap=0.1),
        FixedChunker(chunk_size=256, overlap=0.1),
        RecursiveChunker(chunk_size=256, overlap=0.1),
    ],
    questions=questions,
)

report.compare()

# Hand the winner directly to your pipeline
chunks = report.best_chunker().chunk(my_document)
```

---

## The problem

Most RAG systems pick a chunking strategy once and never revisit it. Fixed 512 tokens with 10% overlap — because that's what the tutorial used. But chunking strategy has a bigger impact on retrieval quality than almost any other decision in a RAG pipeline, and the optimal strategy depends on your specific document.

chunk-norris turns chunking strategy selection from a guess into a measurement.

---

## How it works

1. **You provide** a document and either write questions or let an LLM generate them automatically
2. **chunk-norris chunks** the document with each strategy you want to compare
3. **For each question**, it retrieves the top K chunks using hybrid search — combining semantic search (BERT embeddings) and keyword search (BM25) via Reciprocal Rank Fusion (RRF)
4. **Each retrieved chunk is scored** against the expected answer using two complementary metrics:
   - **Answer span coverage** (token recall) — what fraction of the answer's tokens appear in the chunk? Proxy for BM25 retrievability.
   - **Semantic focus** (bert score) — is the chunk semantically focused on the answer topic? Proxy for semantic search retrievability.
5. **The combined score** approximates hybrid search retrievability — rewarding chunks that are both complete AND focused
6. **Results are ranked** and the best chunker is returned as a usable object

```
Document → [FixedChunker]     → chunks → hybrid retrieve → score → 0.74
Document → [SentenceChunker]  → chunks → hybrid retrieve → score → 0.84  ← best
Document → [RecursiveChunker] → chunks → hybrid retrieve → score → 0.79
                                                                    ↓
                                                best_chunker().chunk(document)
```

**Technical note:** hybrid retrieval combines dense retrieval (BERT embeddings + cosine similarity) and sparse retrieval (BM25) via Reciprocal Rank Fusion (RRF, k=60). This mirrors how modern production RAG systems retrieve content and ensures the evaluation reflects real-world retrieval behaviour.

---

## Design decisions and assumptions

### Design decisions

**One document per run — by design.**
chunk-norris evaluates chunking strategy for a single document at a time. This is intentional: different documents have different optimal strategies. A legal contract, a technical manual, and a customer FAQ should not necessarily use the same chunk size. Running per-document gives you the right strategy for each document, not a compromise that works poorly for all of them.

**Designed as a pipeline component.**
chunk-norris is not a standalone tool — it is a step in your RAG pipeline. After evaluating strategies, `report.best_chunker()` returns the winning chunker instance ready to use. The expected workflow is:

```
Evaluate (chunk-norris) → Chunk → Embed → Store → Query → Answer
```

**No LLM required for evaluation.**
Scoring uses deterministic metrics — token recall and cosine similarity — rather than LLM-as-judge. This makes results reproducible, cost-free, and model-independent. An LLM is optionally used for question generation, never for evaluation.

**Hybrid retrieval by default.**
chunk-norris uses hybrid search (semantic + BM25 via RRF) for retrieval, matching current production RAG best practices. The two evaluation metrics mirror the two retrieval signals: answer span coverage approximates BM25 retrievability, semantic focus approximates semantic search retrievability.

### Assumptions

**Expected answers come from the document vocabulary.**
Token recall measures whether the expected answer's words appear in a retrieved chunk. It works best when your expected answers use the same vocabulary as the document — either copied directly or lightly paraphrased. Heavy paraphrasing or translation may produce lower scores than the actual retrieval quality deserves.

**No adversarial chunks.**
Token recall measures word *presence*, not word *context*. A chunk containing the right words in a misleading context ("Reid Wiseman was not selected for the mission") would score high on recall despite being unhelpful. In practice this is rare in well-structured documents, but be aware that recall is a necessary condition for a good chunk, not a sufficient one.

**Questions are answerable from a single passage.**
Each question should be answerable from a single chunk of the document. Questions that require combining information from multiple sections — comparisons, aggregations, summaries — will score lower than their actual retrieval quality. Multi-chunk question support is planned for a future version.

**LLM-generated questions are optimistic.**
Auto-generated questions are derived from the document's own vocabulary and structure. Absolute scores will be higher than you'd see with real user queries. chunk-norris is designed for relative comparison between chunking strategies — the ranking between strategies is reliable even if absolute scores are optimistic.

**English-language documents.**
BERT embeddings and NLTK sentence tokenisation are optimised for English. Other languages may produce degraded results, particularly for `SentenceChunker`.

---

## Installation

```bash
pip install git+https://github.com/HaroldConley/chunk-norris.git
```

**Requirements:** Python 3.10+

> PyPI release (`pip install chunk-norris`) coming soon.

On first use, `SentenceChunker` will automatically download a small NLTK model (~1MB). This requires an internet connection once — subsequent uses load from cache.

---

## Quick start

### 1. Set up your environment

Copy `.env.example` to `.env` and add your OpenAI key if you want to auto-generate questions:

```bash
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=your-key-here
```

The OpenAI key is only needed for question generation — not for evaluation.

### 2. Prepare your questions

**Option A — auto-generate (recommended):**

```python
from chunk_norris import Norris, BertEmbedder
from chunk_norris.llm.openai_llm import OpenAILLM
from dotenv import load_dotenv

load_dotenv()

norris = Norris(embedder=BertEmbedder())
questions = norris.generate_questions(
    text=text,
    llm=OpenAILLM(model="gpt-4o-mini-2024-07-18"),
    n=20,
)

# Inspect before running — edit or filter as needed
for q in questions:
    print(q["question"])
    print(q["expected_answer"])
```

**Option B — write your own:**

```json
[
    {
        "question": "What is the refund policy?",
        "expected_answer": "Customers can request a refund within 30 days of purchase."
    },
    {
        "question": "How do I contact support?",
        "expected_answer": "Support is available via email at support@company.com."
    }
]
```

**Tip:** Write expected answers using the document's own vocabulary — this gives the most accurate scores. Avoid heavy paraphrasing.

### 3. Run the evaluation

```python
import json
from chunk_norris import (
    Norris, BertEmbedder,
    FixedChunker, ParagraphChunker,
    SentenceChunker, RecursiveChunker,
)

with open("document.txt") as f:
    text = f.read()

norris = Norris(
    embedder=BertEmbedder(),
    top_k=3,
    recall_threshold=0.75,
)

report = norris.run(
    text=text,
    chunkers=[
        FixedChunker(chunk_size=128, overlap=0.1),
        FixedChunker(chunk_size=256, overlap=0.1),
        ParagraphChunker(max_tokens=256),
        SentenceChunker(sentences_per_chunk=3, overlap=0),
        RecursiveChunker(chunk_size=256, overlap=0.1),
    ],
    questions=questions,
)

report.compare()
```

### 4. Use the results

```python
# Comparison table
report.compare()
# Chunker                                    Combined  Recall  Bert   NRel
# ---                                        --------  ------  ----   ----
# SentenceChunker(sentences_per_chunk=3)       0.8417  1.0000  0.689   1.1  <-- best
# ParagraphChunker(max_tokens=256)             0.8072  0.9444  0.679   0.9
# RecursiveChunker(chunk_size=256, ...)        0.7856  0.9833  0.593   1.1

# Export
report.to_excel("results.xlsx")   # Config 1, Config 2... tabs with full detail
report.to_json("results.json")

# Pipeline handoff
best_chunker = report.best_chunker()
chunks = best_chunker.chunk(text)
# vector_store.add(chunks)
```

---

## Core concepts

### Retrieval

chunk-norris uses **hybrid search** by default — the same approach used by modern production RAG systems:

| Component | Technical name | What it does |
|---|---|---|
| Semantic search | Dense retrieval | BERT embeddings + cosine similarity — handles synonyms and paraphrasing |
| Keyword search | Sparse retrieval (BM25) | Term frequency matching — handles exact keywords and technical terms |
| Fusion | Reciprocal Rank Fusion (RRF) | Combines both ranked lists without requiring score normalisation |

Chunks that rank well in both lists score highest. Neither signal is ignored entirely.

### Metrics

The two evaluation metrics mirror the two retrieval signals:

| chunk-norris name | Research equivalent | Retrieval proxy | Measures |
|---|---|---|---|
| **Answer span coverage** | Token recall | BM25 retrievability | Fraction of expected answer tokens present in chunk |
| **Semantic focus** | Semantic similarity | Dense retrievability | Semantic similarity between chunk and expected answer |
| **Combined** | Recall@K signal | Hybrid retrievability | Average of both — primary ranking metric |

**Why two metrics?** They catch different failure modes:

```
Full document as one chunk:
  answer span coverage = 1.00  (contains everything — trivially true)
  semantic focus       = 0.10  (diluted by all other content)
  combined             = 0.55  ← correctly penalised

Perfect focused chunk:
  answer span coverage = 1.00  (gold span fully covered)
  semantic focus       = 0.89  (focused on the answer topic)
  combined             = 0.95  ← correctly rewarded
```

### Recall threshold

The `recall_threshold` controls which chunks are considered relevant. A chunk is relevant if it contains at least this fraction of the expected answer's tokens.

| Value | Meaning | When to use |
|---|---|---|
| `0.50` | 50% of answer words must be present | Conversational docs, heavy paraphrasing |
| `0.75` | 75% of answer words must be present | General purpose — default |
| `0.90` | 90% of answer words must be present | Technical docs with precise vocabulary |
| `1.00` | Every answer word must be present | Exact match required |

If `n_relevant` is always 0 across questions, lower your threshold. If `n_relevant` always equals `top_k`, raise it.

### Questions and expected answers

The quality of your evaluation depends on the quality of your questions. A few guidelines:

- **Write questions that have a specific, locatable answer** in the document
- **Use the document's vocabulary** in expected answers where possible — answer span coverage depends on vocabulary match
- **Avoid questions that require combining multiple sections** (planned for a future version)
- **Aim for 15-30 questions** covering different sections of the document
- **Vary answer span length** — mix short factual questions ("Who?", "When?") with longer explanatory ones ("How does X work?") to stress-test different chunk sizes

---

## Supported chunkers

| Chunker | Strategy | Best for |
|---|---|---|
| `FixedChunker(chunk_size, overlap)` | Fixed token count | Baseline comparison, consistent chunk sizes |
| `ParagraphChunker(max_tokens)` | Natural paragraph boundaries | Well-structured documents with clear sections |
| `SentenceChunker(sentences_per_chunk, overlap)` | Sentence boundaries via NLTK | Documents where sentence integrity matters |
| `RecursiveChunker(chunk_size, overlap)` | Paragraph → sentence → word fallback | General purpose, most document types |

### FixedChunker

```python
FixedChunker(
    chunk_size=256,    # tokens per chunk
    overlap=0.1,       # 10% of chunk_size repeated in next chunk
)
```

### ParagraphChunker

```python
ParagraphChunker(
    max_tokens=512,    # max tokens per chunk (None = no limit)
)
```

Splits on blank lines (`\n\n`). If a paragraph exceeds `max_tokens`, it falls back to sentence-level splitting within that paragraph.

### SentenceChunker

```python
SentenceChunker(
    sentences_per_chunk=3,   # sentences per chunk
    overlap=0,               # sentences repeated from previous chunk
)
```

Uses NLTK for accurate sentence detection — handles abbreviations, ellipsis, and other edge cases that break regex-based splitting.

### RecursiveChunker

```python
RecursiveChunker(
    chunk_size=256,    # max tokens per chunk
    overlap=0.1,       # fraction of chunk_size repeated in next chunk
    separators=None,   # custom separator list (default: ["\n\n", "\n", ". ", " ", ""])
)
```

Tries separators in priority order — paragraphs first, then sentences, then words. Falls back to the next separator only when a piece still exceeds `chunk_size`.

---

## Pluggable embedder

By default chunk-norris uses `BertEmbedder` — a local, free embedding model based on `all-MiniLM-L6-v2`. No API key required.

You can plug in any embedding model by implementing `BaseEmbedder`:

```python
from chunk_norris.embeddings.base import BaseEmbedder

class MyEmbedder(BaseEmbedder):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return my_model.encode(texts).tolist()

norris = Norris(embedder=MyEmbedder())
```

---

## Project status

chunk-norris is in active development. Current version: **0.2.0**

### What works now
- Four chunking strategies: Fixed, Paragraph, Sentence, Recursive
- Hybrid retrieval: semantic search + BM25 via Reciprocal Rank Fusion
- Automatic question generation using an LLM (gpt-4o-mini recommended)
- Deterministic evaluation: answer span coverage + semantic focus
- Pipeline integration: `best_chunker()` returns the winner ready to use
- Excel export with Config tabs, full chunker labels, and per-question detail
- Pluggable embedder and LLM interfaces

### Planned
- Multi-chunk question support — for questions that require combining information from multiple sections
- More chunking strategies — Semantic chunker, Markdown-aware chunker

---

## Contributing

Contributions are welcome. The most valuable contributions right now:

- New chunking strategies (see `src/chunk_norris/chunkers/base.py` for the interface)
- New embedding providers (see `src/chunk_norris/embeddings/base.py`)
- New retrieval strategies (see `src/chunk_norris/retrieval/base.py`)
- Bug reports with minimal reproducible examples

Please write tests for any new code. Run the test suite before opening a PR:

```bash
pip install chunk-norris[dev]
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full setup instructions.

---

## License

MIT — use it, fork it, build on it.

---

*"In Chuck Norris's RAG pipeline, the chunks retrieve themselves."*
