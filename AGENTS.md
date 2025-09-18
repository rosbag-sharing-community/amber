# Repository Guidelines

## Project Structure & Module Organization
- Source: `amber_mcap/` (datasets, automation, importer, util, visualization). Native C++ lives under `amber_mcap/tf2/**` with `CMakeLists.txt`.
- Tests: `tests/` with small assets in `tests/images/`, `tests/video/`, and YAML configs in `tests/automation/`.
- Docs: `docs/` built with MkDocs. Project config in `pyproject.toml`; build helper `build.py`.

## Build, Test, and Development Commands
- Install (dev): `poetry install --with dev` (builds the `tf2_amber` extension via scikit-build/CMake). Optional groups: `--with apps,docs`.
- Rebuild C++ only: `poetry run python build.py` (copies built `.so` into the Poetry venv).
- Run tests: `poetry run pytest` or `poetry run pytest -k <pattern>`.
- Lint/format: `poetry run pre-commit run --all-files`.
- CLI entrypoint: `poetry run amber ...`.

## Coding Style & Naming Conventions
- Python: Black + isort, flake8 (88 cols, E203 ignored). Type hints required; `mypy` is strict. Use `snake_case` for functions/vars, `PascalCase` for classes, modules in `lower_snake_case`.
- C++: C++17 with `-Wall -Wextra -Wpedantic`. Follow existing TF2 code style; prefer descriptive names over abbreviations.

## Testing Guidelines
- Framework: `pytest` with coverage configured in `pyproject.toml`. Place tests under `tests/` named `test_*.py`.
- Keep tests deterministic and fast. Do not add large assets; reuse fixtures in `tests/`.
- Some tests require credentials or heavy models. S3-related tests are auto-skipped without `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`. Avoid network in unit tests unless guarded.

## Commit & Pull Request Guidelines
- Commits: short, imperative subject (≤ ~72 chars). Conventional prefixes encouraged: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `build:`.
- PRs: focused scope, clear description, rationale, and linked issues. Include screenshots/logs when relevant. Update docs (`docs/`) and CLI help for user-visible changes.

## Security & Configuration Tips
- Keep secrets in environment variables; never commit keys. Examples: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`.
- GPU dependencies are pinned to CUDA 12.4 wheels (`torch_cu124` source). Ensure environment compatibility or switch to CPU variants locally.
