# AGENTS.md

## General code guidelines

### Comments

Use inline comments where needed, don't add any comments at the top of any files.
This creates drift. Write brief docstrings describing only what the unit does,
don't let context from the calling places creep into a function's docstring and
comments.

### Tests

Use test driven development. Design components with testable interfaces that do
one thing well. Use unit tests inside components. Add integration tests for
actions between these interfaces.

### For Python

- Manage Python environments via pixi.
- Use mypy, and set type hints everywhere.
- Use ruff.
- Set up a pixi pre-commit task running unit tests, type checks, and lints.
