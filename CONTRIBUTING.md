# Contributing to rtvoice

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync --all-extras --dev
```

## Development

- Install pre-commit hooks: `uv run pre-commit install`
- Run tests: `uv run pytest`
- Format code: `uv run black .`

## Guidelines

- Keep changes focused; avoid unrelated refactors in the same PR.
- Write comments that explain *why*, not *what* — don't restate what the code
  already makes obvious.
- Match existing code style and structure before introducing new patterns.

## Pull requests

- Ensure `pytest` and pre-commit hooks pass before opening a PR.
- Describe the motivation for the change, not just what changed.
