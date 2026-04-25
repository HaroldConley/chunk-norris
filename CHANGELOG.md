# Changelog

All notable changes to chunk-norris will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `QuestionGenerator` — generates question-answer pairs from a document
  using passage-based LLM generation, guaranteeing location diversity
  and single-chunk answerability
- `Norris.generate_questions()` — public interface for question generation,
  accepts any `BaseLLM` implementation
- `OpenAILLM` re-exposed in public API — required for question generation
- `.env` support via `python-dotenv` — API keys loaded from `.env` file,
  never hardcoded
- `.env.example` — template for contributors to set up their environment

### Changed
- Updated metrics terminology in README to align with RAG research standards:
  token recall → answer span coverage, bert score → semantic focus,
  best_combined → Recall@K signal
- `Norris` docstring updated to show both question workflows
  (auto-generate and manual)

### Planned for v0.2.0
- Automatic question generation from document structure using LLM
  *(in progress — see QuestionGenerator)*
- Multi-chunk question support — evaluation for questions whose answers
  span multiple sections of the document
- Optional LLM answer generation — generate and inspect actual RAG answers
  alongside retrieval scores
- SemanticChunker — embedding-based boundary detection
- MarkdownChunker — header-aware splitting for Markdown documents

### Planned for v0.3.0
- BM25 retrieval — keyword-based retrieval as an alternative to dense search
- Hybrid retrieval — combine dense embeddings and BM25 via Reciprocal Rank
  Fusion (RRF), matching current production RAG best practices
- Question diversity guidance — ensure generated question sets cover
  different answer span lengths and document sections

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
