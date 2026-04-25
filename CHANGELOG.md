# Changelog

All notable changes to chunk-norris will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned for v0.2.0
- Multi-chunk question support — evaluation for questions whose answers
  span multiple sections of the document
- SemanticChunker — embedding-based boundary detection
- MarkdownChunker — header-aware splitting for Markdown documents

---

## [0.2.0] — 2026-04-25

### Added

**Automatic question generation**
- `QuestionGenerator` — generates question-answer pairs from a document
  using a 4-step structured LLM prompt: fact extraction → fact selection
  → question generation → verbatim answer extraction
- Passage-based generation using sentence-level windows (2-3 sentences)
  for atomic fact alignment — avoids multi-fact collapse from paragraph-level splitting
- Prompt designed to reduce date/time bias in favour of people, processes,
  technical specifications, and causal relationships
- Circular Q&A detection — rejects pairs where the answer is a substring
  of the question
- Minimum answer length filter — rejects fragment answers under 5 tokens
- `Norris.generate_questions()` — public interface, accepts any `BaseLLM`
- `OpenAILLM` re-exposed in public API
- `.env` support via `python-dotenv` — API keys loaded from `.env` file
- `.env.example` — template for contributors

**Hybrid retrieval**
- `HybridRetriever` — combines semantic search and BM25 keyword search
  via Reciprocal Rank Fusion (RRF, k=60)
- `DenseRetriever` — semantic search using BERT embeddings and cosine
  similarity (formerly `evaluator/Retriever`)
- `BM25Retriever` — keyword search using BM25Okapi
- `BaseRetriever` — pluggable interface for custom retrieval strategies
- Retrieval scores in chunk metadata: `semantic_score`, `keyword_score`,
  `score` (RRF combined)
- Norris now uses `HybridRetriever` by default — mirrors modern production
  RAG systems that combine dense and sparse retrieval

**Excel improvements**
- Detail sheets now use `Config N` tab names (Excel 31-character limit)
- Full chunker label shown in title row of each detail sheet
- Config column added to Summary sheet for easy navigation between sheets

### Changed
- Updated metrics terminology to align with RAG research standards:
  token recall → answer span coverage (BM25 proxy),
  bert score → semantic focus (semantic search proxy),
  combined → Recall@K signal (hybrid search proxy)
- `Norris` docstring updated to document both question workflows

### Technical notes
- `answer span coverage` (token recall) approximates BM25 retrievability
- `semantic focus` (bert score) approximates semantic search retrievability
- `combined score` approximates hybrid search retrievability
- The combined metric mirrors a 50/50 hybrid retrieval system

---

## [0.1.0] — 2026-04-13

First public release.

### Added

**Core evaluation loop**
- `Norris` — main entry point, orchestrates the full evaluation pipeline
- `Retriever` — semantic chunk retrieval using cosine similarity
- `Metrics` — deterministic scoring using answer span coverage (token recall)
  and semantic focus (bert score), combined as a Recall@K signal
- `Report` — comparison table, Excel export, JSON export

**Chunking strategies**
- `FixedChunker` — fixed token size with percentage overlap
- `ParagraphChunker` — splits on blank lines, falls back to sentence splitting
  for oversized paragraphs
- `SentenceChunker` — groups sentences using NLTK, with sentence-level overlap
- `RecursiveChunker` — tries paragraph → sentence → word boundaries in
  priority order

**Embeddings**
- `BertEmbedder` — local, free embeddings using `sentence-transformers`
  (`all-MiniLM-L6-v2`)
- `BaseEmbedder` — pluggable interface for custom embedding providers

**Pipeline integration**
- `report.best_chunker()` — returns the winning chunker instance ready to
  use in a RAG pipeline

**Evaluation metrics**
- Answer span coverage (token recall) — fraction of expected answer tokens
  present in retrieved chunk
- Semantic focus (bert score) — semantic similarity between chunk and
  expected answer
- Combined score — average of both, primary Recall@K signal
- Relevance filtering — configurable `recall_threshold` to distinguish
  relevant from noise chunks

**Export**
- `report.to_excel()` — multi-sheet workbook: summary + per-chunker detail
  with chunk texts and individual scores
- `report.to_json()` — full machine-readable export

**Developer experience**
- Full test suite — 257 tests across all modules
- Type hints throughout
- Docstrings on all public methods and classes
