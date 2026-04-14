# chunk-norris 🥋

**Find the best chunking strategy for your RAG pipeline — empirically, not by gut feel.**

chunk-norris evaluates and compares chunking strategies against your actual document and questions, then hands the winning chunker directly to your pipeline. No guesswork, no boilerplate, no LLM required for evaluation.

```python
from chunk_norris import Norris, BertEmbedder, FixedChunker, RecursiveChunker

norris = Norris(embedder=BertEmbedder())

report = norris.run(
    text=my_document,
    chunkers=[
        FixedChunker(chunk_size=128, overlap=0.1),
        FixedChunker(chunk_size=256, overlap=0.1),
        RecursiveChunker(chunk_size=256, overlap=0.1),
    ],
    questions=my_questions,
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

1. **You provide** a document, a set of questions, and their expected answers
2. **chunk-norris chunks** the document with each strategy you want to compare
3. **For each question**, it retrieves the top K chunks using semantic search
4. **Each retrieved chunk is scored** against the expected answer using two complementary metrics:
   - **Token recall** — does the chunk contain the answer's key words? *(measures completeness)*
   - **Bert score** — is the chunk semantically focused on the answer? *(penalises noise)*
5. **The combined score** is their average — rewarding chunks that are both complete and focused
6. **Results are ranked** and the best chunker is returned as a usable object

```
Document → [FixedChunker]     → chunks → retrieve → score → 0.74
Document → [SentenceChunker]  → chunks → retrieve → score → 0.81  ← best
Document → [RecursiveChunker] → chunks → retrieve → score → 0.79
                                                              ↓
                                              best_chunker().chunk(document)
```

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
Scoring uses deterministic metrics — token recall and cosine similarity — rather than LLM-as-judge. This makes results reproducible, cost-free, and model-independent. LLM answer generation is planned as an optional step in a future version.

### Assumptions

**Expected answers come from the document vocabulary.**
Token recall measures whether the expected answer's words appear in a retrieved chunk. It works best when your expected answers use the same vocabulary as the document — either copied directly or lightly paraphrased. Heavy paraphrasing or translation may produce lower scores than the actual retrieval quality deserves.

**No adversarial chunks.**
Token recall measures word *presence*, not word *context*. A chunk containing the right words in a misleading context ("Reid Wiseman was not selected for the mission") would score high on recall despite being unhelpful. In practice this is rare in well-structured documents, but be aware that recall is a necessary condition for a good chunk, not a sufficient one.

**Questions are answerable from a single passage.**
Each question should be answerable from a single chunk of the document. Questions that require combining information from multiple sections — comparisons, aggregations, summaries — will score lower than their actual retrieval quality. Multi-chunk question support is planned for v2.

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

### 1. Prepare your data

Create `questions.json` with your questions and expected answers:

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

**Tip:** Write expected answers using the document's own vocabulary where possible — this gives the most accurate token recall scores. Avoid heavy paraphrasing.

### 2. Run the evaluation

```python
import json
from chunk_norris import (
    Norris, BertEmbedder,
    FixedChunker, ParagraphChunker,
    SentenceChunker, RecursiveChunker,
)

with open("document.txt") as f:
    text = f.read()

with open("questions.json") as f:
    questions = json.load(f)

norris = Norris(
    embedder=BertEmbedder(),
    top_k=3,
    recall_threshold=0.75,  # chunk must contain 75% of answer tokens to be relevant
)

report = norris.run(
    text=text,
    chunkers=[
        FixedChunker(chunk_size=128, overlap=0.1),
        FixedChunker(chunk_size=256, overlap=0.1),
        ParagraphChunker(max_tokens=256),
        SentenceChunker(sentences_per_chunk=5, overlap=1),
        RecursiveChunker(chunk_size=256, overlap=0.1),
    ],
    questions=questions,
)

report.compare()
```

### 3. Use the results

```python
# Print comparison table
report.compare()
# Chunker                                 Combined  Recall  Bert   NRel
# ---                                     --------  ------  ----   ----
# RecursiveChunker(chunk_size=256, ...)     0.8200  0.9100  0.730   2.3  <-- best
# SentenceChunker(sentences_per_chunk=5)   0.7800  0.8800  0.680   2.1
# FixedChunker(chunk_size=256, ...)        0.7400  0.8500  0.630   1.8

# Export full results
report.to_excel("results.xlsx")   # multi-sheet workbook with all details
report.to_json("results.json")    # machine-readable full export

# Pipeline handoff — use the winner directly
best_chunker = report.best_chunker()
chunks = best_chunker.chunk(text)
print(f"Best strategy: {best_chunker}")
print(f"Chunks produced: {len(chunks)}")

# Pass to your vector store or RAG pipeline
# vector_store.add(chunks)
```

---

## Core concepts

### Metrics

chunk-norris uses two complementary metrics to score each retrieved chunk against the expected answer:

| Metric | Measures | Catches |
|---|---|---|
| **Token recall** | Fraction of expected answer tokens present in chunk | Missing content, wrong section retrieved |
| **Bert score** | Semantic similarity between chunk and expected answer | Noisy chunks, full-document chunks |
| **Combined** | Average of both | Primary ranking signal |

**Why two metrics?** They catch different failure modes:

```
Full document as one chunk:
  token_recall = 1.00  (contains everything — trivially true)
  bert_score   = 0.10  (diluted by all other content)
  combined     = 0.55  ← correctly penalised

Perfect focused chunk:
  token_recall = 1.00  (contains the answer)
  bert_score   = 0.89  (focused on the answer topic)
  combined     = 0.95  ← correctly rewarded
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
- **Use the document's vocabulary** in expected answers where possible
- **Avoid questions that require combining multiple sections** (planned for v2)
- **Aim for 15-30 questions** that cover different sections of the document
- Each expected answer should be **one to three sentences** — concise and factual

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
    sentences_per_chunk=5,   # sentences per chunk
    overlap=1,               # sentences repeated from previous chunk
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

Tries separators in priority order — paragraphs first, then sentences, then words. Falls back to the next separator only when a piece still exceeds `chunk_size`. The most generally useful chunker for documents without a known structure.

---

## Pluggable embedder

By default chunk-norris uses `BertEmbedder` — a local, free embedding model based on `all-MiniLM-L6-v2`. No API key required.

You can plug in any embedding model by implementing `BaseEmbedder`:

```python
from chunk_norris.embeddings.base import BaseEmbedder

class MyEmbedder(BaseEmbedder):
    def embed(self, texts: list[str]) -> list[list[float]]:
        # return one vector per text
        return my_model.encode(texts).tolist()

norris = Norris(embedder=MyEmbedder())
```

---

## Project status

chunk-norris is in active development. Current version: **0.1.0**

### What works now
- Four chunking strategies: Fixed, Paragraph, Sentence, Recursive
- Deterministic evaluation: token recall + bert score
- Pipeline integration: `best_chunker()` returns the winner ready to use
- Excel export with per-question, per-chunk detail
- Pluggable embedder interface

### Planned for v2
- **Automatic question generation** — LLM generates Q&A pairs from the document so you don't need to write them manually
- **Multi-chunk question support** — for questions that require combining information from multiple sections
- **Optional LLM answer generation** — generate and inspect actual answers alongside retrieval scores
- **More chunking strategies** — Semantic chunker, Markdown-aware chunker

---

## Contributing

Contributions are welcome. The most valuable contributions right now:

- New chunking strategies (see `src/chunk_norris/chunkers/base.py` for the interface)
- New embedding providers (see `src/chunk_norris/embeddings/base.py`)
- Bug reports with minimal reproducible examples

Please write tests for any new code. Run the test suite before opening a PR:

```bash
pip install chunk-norris[dev]
pytest
```

---

## License

MIT — use it, fork it, build on it.

---

*"In Chuck Norris's RAG pipeline, the chunks retrieve themselves."*
