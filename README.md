# vla-chess

Simulated chess pick-and-place trajectory generation for VLA fine-tuning.

See `strategy-step-1.md` for the phase strategy and `implementation-plan-step-1.md` for the
implementation plan.

## Setup

```bash
pixi install            # create the environment from pixi.lock
pixi run install-hooks  # point git at .githooks (once per clone)
```

`core.hooksPath` is local git config and is not carried over by `git clone`, so
`pixi run install-hooks` has to be run once in every fresh checkout for the pre-commit gate
to be active.

## Tasks

| Task | Runs |
|---|---|
| `pixi run lint` | `ruff check` and `ruff format --check` |
| `pixi run format` | `ruff format` |
| `pixi run typecheck` | `mypy` (strict) over `src/` |
| `pixi run test` | unit tests, excluding tests marked `slow` |
| `pixi run test-slow` | only tests marked `slow` |
| `pixi run precommit` | `lint`, `typecheck`, `test` — what the git hook runs |

Tests marked `slow` (CPU-expensive) and `render` (needs the osmesa software renderer) are
opt-in and are not run by `precommit`.
