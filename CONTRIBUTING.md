# Contributing to chunk-norris

Thank you for considering a contribution. This document explains how to get set up, what kinds of contributions are most valuable, and how to submit your work.

---

## Table of contents

- [Getting started](#getting-started)
- [Running the tests](#running-the-tests)
- [Coding conventions](#coding-conventions)
- [Adding a new chunker](#adding-a-new-chunker)
- [Adding a new embedder](#adding-a-new-embedder)
- [Submitting a pull request](#submitting-a-pull-request)
- [Reporting bugs](#reporting-bugs)

---

## Getting started

**Prerequisites:** Python 3.10+, Git.

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/HaroldConley/chunk-norris.git
cd chunk-norris

# 2. Create a virtual environment
python -m venv .venv

# Activate — macOS/Linux
source .venv/bin/activate

# Activate — Windows
.venv\Scripts\activate

# 3. Install in editable mode with dev dependencies
pip install -e ".[dev]"

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
# (only needed for automatic question generation)
```

That's it. No database, no services, no API keys needed for the core library.

---

## Running the tests

```bash
# Run the full test suite
pytest

# Run with verbose output — shows each test name
pytest -v

# Run a specific file
pytest tests/chunkers/test_fixed.py -v

# Run a specific test class
pytest tests/chunkers/test_fixed.py::TestFixedChunkerInit -v

# Run with coverage report
pytest --cov=src/chunk_norris --cov-report=term-missing
```

All tests must pass before submitting a PR. If you add new code, add tests for it.

---

## Coding conventions

**Type hints everywhere.** Every function and method signature should have type hints — parameters and return types.

```python
# Good
def chunk(self, text: str) -> list[dict[str, Any]]:

# Not good
def chunk(self, text):
```

**Docstrings on every public method.** Follow the existing pattern — one-line summary, blank line, full description, Args section, Returns section.

```python
def chunk(self, text: str) -> list[dict[str, Any]]:
    """
    Splits the input text into chunks.

    Args:
        text (str): The input text to split.

    Returns:
        list[dict]: A list of chunk dicts, each with 'text' and 'metadata' keys.
    """
```

**No bare excepts.** Always catch specific exception types.

**Use `deepcopy` for metadata.** The `_create_chunk()` helper in `BaseChunker` handles this — use it rather than building chunk dicts manually.

**Python 3.10+ syntax.** Use `X | Y` instead of `Optional[X]` or `Union[X, Y]`. Use `list[str]` instead of `List[str]`.

**Line length: 88 characters.** Configured in `pyproject.toml`.

---

## Adding a new chunker

Adding a chunker is the most common contribution. Here is the complete pattern:

### 1. Create the file

Create `src/chunk_norris/chunkers/your_chunker.py`. Inherit from `BaseChunker` and implement two methods:

```python
from typing import Any
from chunk_norris.chunkers.base import BaseChunker


class YourChunker(BaseChunker):
    """
    One-line description of the strategy.

    Longer explanation of how it works and when to use it.

    Args:
        param_one (type): Description. Default: value.

    Example::

        chunker = YourChunker(param_one=value)
        chunks = chunker.chunk("Your document text here...")
    """

    def __init__(self, param_one: int = 256) -> None:
        if param_one <= 0:
            raise ValueError(f"param_one must be positive, got {param_one}")
        self.param_one = param_one

    def chunk(self, text: str) -> list[dict[str, Any]]:
        if not text or not text.strip():
            return []

        # Your chunking logic here
        chunks = []
        chunks.append(
            self._create_chunk(
                text="chunk text",
                metadata={
                    "chunk_index": 0,
                    "token_count": 10,
                    # add your strategy-specific metadata here
                },
            )
        )
        return chunks

    def __repr__(self) -> str:
        return f"YourChunker(param_one={self.param_one})"
```

### 2. Write the tests

Create `tests/chunkers/test_your_chunker.py`. At minimum, cover:

- `__init__` — valid values stored, invalid values raise `ValueError`
- `chunk("")` — returns empty list
- `chunk("some text")` — returns list of dicts with `text` and `metadata` keys
- Chunk indices are sequential
- Metadata contains all expected keys
- `__repr__` returns the right string
- Any behaviour specific to your strategy

### 3. Expose it in `__init__.py`

Add to `src/chunk_norris/__init__.py`:

```python
from chunk_norris.chunkers.your_chunker import YourChunker
```

And add `"YourChunker"` to `__all__`.

### 4. Add to `basic_usage.py` (optional)

If the chunker is generally useful, add an example configuration to `examples/basic_usage.py`.

---

## Adding a new embedder

Adding an embedder follows the same pattern.

### 1. Create the file

Create `src/chunk_norris/embeddings/your_embedder.py`:

```python
from chunk_norris.embeddings.base import BaseEmbedder, EmbeddingError


class YourEmbedder(BaseEmbedder):
    """Description of the embedding provider."""

    def __init__(self, model_name: str = "default-model") -> None:
        try:
            self._model = load_your_model(model_name)
        except Exception as e:
            raise EmbeddingError(
                message=f"Failed to load model '{model_name}'.",
                provider="YourProvider",
                original_error=e,
            ) from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("texts must not be empty.")
        try:
            return self._model.encode(texts).tolist()
        except Exception as e:
            raise EmbeddingError(
                message="Failed to generate embeddings.",
                provider="YourProvider",
                original_error=e,
            ) from e
```

Key requirements:
- `embed()` accepts a **list** of texts and returns a **list** of float vectors — one per input
- Wrap provider-specific errors in `EmbeddingError` — never let them leak out
- Load the model in `__init__`, not in `embed()` — expensive loading should happen once

### 2. Write the tests

Mock the underlying model to keep tests fast. See `tests/evaluator/test_metrics.py` for examples of how to mock an embedder.

---

## Submitting a pull request

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** — one logical change per PR. If you're adding a chunker and fixing a bug, open two PRs.

3. **Run the tests** and make sure everything passes:
   ```bash
   pytest
   ```

4. **Write a clear PR description** that explains:
   - What the change does
   - Why it's needed
   - How to test it manually if relevant

5. **Open the PR** against the `main` branch of `HaroldConley/chunk-norris`.

PRs that add new chunkers or embedders without tests will not be merged. PRs that break existing tests will not be merged.

---

## Reporting bugs

Open a GitHub issue at [github.com/HaroldConley/chunk-norris/issues](https://github.com/HaroldConley/chunk-norris/issues) with:

- A **minimal reproducible example** — the smallest possible code that triggers the bug
- The **expected behaviour** — what should have happened
- The **actual behaviour** — what happened instead, including the full error traceback
- Your **Python version** and **chunk-norris version** (`pip show chunk-norris`)

The more specific the report, the faster it gets fixed.

---

## What we're looking for

The most valuable contributions right now, in priority order:

1. **Bug fixes** — especially with a failing test that demonstrates the bug
2. **New chunking strategies** — Semantic chunker, Markdown-aware chunker
3. **New embedding providers** — OpenAI embeddings, Cohere, etc.
4. **Documentation improvements** — clearer explanations, better examples
5. **Performance improvements** — the evaluation loop processes questions one at a time; parallelisation would help for large question sets
