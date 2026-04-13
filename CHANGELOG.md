# Changelog

All notable changes to chunk-norris will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

Planned for v0.2.0:
- Automatic question generation using an LLM
- Multi-chunk question support
- Optional LLM answer generation for result inspection
- SemanticChunker — embedding-based boundary detection
- MarkdownChunker — header-aware splitting

---

## [0.1.0] — 2026-04-13

First public release.

### Added

**Core evaluation loop**
- `Norris` — main entry point, orchestrates the full evaluation pipeline
- `Retriever` — semantic chunk retrieval using cosine similarity
- `Metrics` — deterministic scoring using token recall and bert score
- `Report` — comparison table, Excel export, JSON export

**Chunking strategies**
- `FixedChunker` — fixed token size with percentage overlap
- `ParagraphChunker` — splits on blank lines, falls back to sentence splitting for oversized paragraphs
- `SentenceChunker` — groups sentences using NLTK, with sentence-level overlap
- `RecursiveChunker` — tries paragraph → sentence → word boundaries in priority order

**Embeddings**
- `BertEmbedder` — local, free embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`)
- `BaseEmbedder` — pluggable interface for custom embedding providers

**Pipeline integration**
- `report.best_chunker()` — returns the winning chunker instance ready to use in a RAG pipeline

**Evaluation metrics**
- Token recall — fraction of expected answer tokens present in retrieved chunk
- Bert score — semantic similarity between chunk and expected answer
- Combined score — average of both, primary ranking metric
- Relevance filtering — configurable `recall_threshold` to distinguish relevant from noise chunks

**Export**
- `report.to_excel()` — multi-sheet workbook: summary + per-chunker detail with chunk texts and scores
- `report.to_json()` — full machine-readable export

**Developer experience**
- Full test suite — 257 tests across all modules
- Type hints throughout
- Docstrings on all public methods and classes