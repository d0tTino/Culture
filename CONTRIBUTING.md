# Contributing to Culture.ai

Welcome! We're excited that you're interested in contributing to Culture.ai. This project is open to all contributors—whether you're fixing bugs, adding features, improving docs, or suggesting ideas.

## Getting Started
- **Read the [README.md](README.md)** for setup, prerequisites, and quickstart instructions.
- Install development tools:
- [Ruff](https://docs.astral.sh/ruff/) for linting/formatting
- [Mypy](http://mypy-lang.org/) for static type checking
- Install dependencies from `requirements-dev.txt` (includes `pytest-xdist`) in addition to `requirements.txt`.
- Set up your environment and dependencies as described in the README.
- Enable an [EditorConfig](https://editorconfig.org/) plugin in your editor if available to match the project's formatting settings.
- If you modify dependencies:
  1. Update `requirements.in` or `requirements-dev.in` as needed.
  2. Regenerate the lock files:
     ```bash
     pip-compile requirements.in -o requirements.txt --no-annotate --no-header
     pip-compile requirements-dev.in -o requirements-dev.txt --no-annotate --no-header
     ```
  3. Run `scripts/check_requirements.sh` to verify the files are in sync.
  4. Commit the updated `requirements*.txt` files.

## Code Style Guidelines
- **Follow [PEP 8](https://peps.python.org/pep-0008/)** for Python code style.
- **Type Annotations:** Use Python 3.10+ type hints (e.g., `str | None` instead of `Optional[str]`).
- **Formatting:**
  - Run `ruff format .` before committing.
  - Use `ruff check .` to catch lint issues.
- **Docstrings:**
  - Use Google or reStructuredText style for all public classes and methods.
  - Document parameters and return types.

## Development Workflow
- **Branching:** Create feature branches from `main` (e.g., `feature/your-feature` or `bugfix/your-bug`).
- **Commits:** Write clear, descriptive commit messages.
- **Pull Requests:**
  - Use a descriptive title and summary.
  - Reference related issues (e.g., `Closes #123`).
  - Ensure all tests pass before requesting review.
  - The **UI** workflow runs automatically for changes in `culture-ui`, executing
    `pnpm --filter culture-ui lint` and `pnpm --filter culture-ui type-check`.
    These steps must pass before your PR can merge.
  - At least one approval is required before merging.

## Testing Requirements
- **Write unit and integration tests** for new features and bug fixes.
- **Run tests locally:**
  ```bash
  python -m pytest tests/
  ```
- **Check coverage:**
  ```bash
  python -m pytest --cov=src --cov-report=term-missing tests/
  ```
- **Dependency lock check:** `scripts/check_requirements.sh` verifies that `requirements.txt` matches the output of `pip-compile`. The CI workflow runs this step automatically.
- See [docs/testing.md](docs/testing.md) for advanced test strategies, markers, and troubleshooting.

## Reporting Bugs & Requesting Features
- Use [GitHub Issues](https://github.com/d0tTino/Culture/issues) to report bugs or request features.
- For bugs, provide clear, reproducible steps and relevant logs or error messages.

## Code of Conduct
We are committed to a welcoming and inclusive environment. Please be respectful and constructive in all interactions. See [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/) for our code of conduct.

Thank you for helping make Culture.ai better! 
