# Contributing to LM Studio Auto Optimizer

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and inclusive in all interactions.

## How to Contribute

### Reporting Bugs

Before submitting a bug report:
1. Check existing issues to avoid duplicates
2. Use the bug report template
3. Include:
   - LM Studio version
   - Hardware specs (GPU, RAM, OS)
   - Steps to reproduce
   - Error logs (with sensitive info redacted)
   - Expected vs actual behavior

### Suggesting Features

Feature requests are welcome! Please:
1. Check existing feature requests
2. Describe the problem the feature solves
3. Provide use cases
4. Consider implementation complexity

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `pytest`
5. Run type checks: `mypy lm_optimizer`
6. Run linter: `ruff check lm_optimizer`
7. Format code: `ruff format lm_optimizer`
8. Submit PR with clear description

## Development Setup

```bash
# Clone and install in development mode
git clone <repo>
cd LM_Studio_Optimiser
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Style

- **Python**: 3.11+ with type hints
- **Formatting**: Ruff (line length 100)
- **Type checking**: mypy (strict mode)
- **Imports**: Absolute imports, organized by stdlib → third-party → local

### Type Hints

All new code must have type hints:

```python
def function(param: Type) -> ReturnType:
    ...
```

Use `Optional[Type]` for nullable, `list[Type]` for lists.

### Async Code

Use `async/await` for I/O operations. Prefer `asyncio` primitives.

```python
async def fetch_data(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()
```

## Testing

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest lm_optimizer/tests/test_core.py

# With coverage
pytest --cov=lm_optimizer

# Verbose
pytest -v
```

### Writing Tests

- Place tests in `lm_optimizer/tests/`
- Use `pytest` and `pytest-asyncio`
- Mock external dependencies (LM Studio, hardware)
- Test both success and failure paths

```python
@pytest.mark.asyncio
async def test_optimization_run(mock_client):
    optimizer = AdaptiveOptimizer(mock_client, ...)
    result = await optimizer.optimize(...)
    assert result.status == RunStatus.COMPLETED
```

## Documentation

- Update README.md for user-facing changes
- Update docstrings for API changes
- Add comments for complex logic
- Keep CHANGELOG.md updated

## Commit Messages

Follow conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat(optimizer): add Pareto frontier calculation
fix(hardware): handle missing GPU on macOS
docs(readme): update installation instructions
```

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release tag
4. Build and publish to PyPI

## Getting Help

- Open a GitHub issue for bugs/features
- Start a Discussion for questions
- Check existing docs first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.