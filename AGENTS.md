# Repository Guidelines

## Project Structure & Module Organization
- Core library lives in `src/state`, with CLI entrypoints in `state/__main__.py` and subcommands under `state/_cli`.
- Model configs and resources sit in `src/state/configs`, embeddings helpers in `src/state/emb`, and transition utilities in `src/state/tx`.
- Tests reside in `tests/` (`test_*.py`), runnable without extra fixtures. Example TOML configs are in `examples/`. Helper scripts live in `scripts/` for inference and embedding.
- Artifacts or scratch outputs should go in `tmp/` or a user-created path; keep `assets/` for checked-in visuals/resources only.

## Build, Test, and Development Commands
- Create/activate env and install in editable mode: `uv tool install -e .`.
- Run the CLI: `uv run state --help` (entrypoints `emb` and `tx`).
- Format/lint: `uv run ruff check .` (auto-fixes enabled by default config).
- Run tests: `uv run pytest` (adds `src/` to `PYTHONPATH` via standard layout).

## Coding Style & Naming Conventions
- Python 3.10–3.12; prefer type hints on public functions.
- Use 4-space indentation, 120-char max line length (`ruff.toml`), and avoid bare `except` (E722 is explicitly ignored—only use when necessary).
- Modules and files use `snake_case`; classes `CamelCase`; constants `UPPER_SNAKE_CASE`.
- Keep CLI options descriptive and align new configs with the existing TOML examples.

## Testing Guidelines
- Add unit tests alongside new features in `tests/` with filenames `test_*.py` and functions `test_*`.
- Cover edge cases around data loading, config parsing, and checkpoint handling; favor small fixtures over large data blobs.
- For regressions, reproduce with a failing test first, then implement the fix.

## Commit & Pull Request Guidelines
- Follow the short, imperative style seen in history (`chore: …`, `patch: …`, or focused message without trailing punctuation). Reference issue/PR numbers where applicable.
- PRs should explain the change, risks, and testing done (`uv run pytest`, `uv run ruff check .`). Include CLI examples if you changed commands or configs.
- Keep diffs scoped; split unrelated changes into separate PRs. Include screenshots or logs only when UI/output changes are relevant.

## Security & Configuration Tips
- Do not commit dataset paths or secrets; use environment variables or local config files kept out of git.
- Validate file paths in new CLI options and prefer existing config loaders under `state/_cli` to avoid duplicating logic.
