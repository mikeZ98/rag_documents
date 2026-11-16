# AGENTS.md — rag_documents

Single source of truth for every coding agent (Claude Code imports it via CLAUDE.md;
Cursor and Antigravity read AGENTS.md directly).

## Boundaries
- Lives on the KINGSTONE_SSD. All tool state stays in `./.cache/` (see `.envrc`);
  never write caches, venvs or downloads to `$HOME` or the internal disk.
- Python: **uv only** (`uv add`, `uv run`); commit `uv.lock`. No pip/poetry/conda.
- Project-specific notes (architecture, commands) live in `CLAUDE.md` — read them too.

## Definition of done
`qg` passes: ruff check · ruff format --check · mypy --strict · pytest.

## Conventions
- Full type hints; no `Any` on public APIs. Small pure functions, explicit errors.
- Conventional Commits. Git identity: mikeZ98 <145975972+mikeZ98@users.noreply.github.com>.
- Smallest correct change; keep modules isolated.
