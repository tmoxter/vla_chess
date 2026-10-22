## Work in progress

Experimenting with hierarchical embodied agents, built on a SO-101 arm, LeRobot, and a fine-tuned VLA, currently SmolVLA. It plays physical chess against a human. It also serves as a testbed for some system 1 vs. system 2 inspired planning and execution collaboration (e.g. the planner needs tool usage to operate the chess engine) as well as affordance-weighted guidance (e.g. I want it to play on the pareto front of moves that land and moves that are good, since thats the only way I can beat it).

Source layout (`src/vla_chess/`):

| Module | Responsibility |
|---|---|
| `sim/` | MuJoCo scene, board + pieces, start positions |
| `grasp/` | minimum-tilt vertical grasp solver |
| `control/` | motion `planner` (IK/waypoints, tilted-move planning), `executor`, candidate `sampling` |
| `dataset/` | `episode` funnel runner, observation `capture`, `validate`, LeRobot writer |
| `sensor/` | camera rendering (osmesa CPU / EGL GPU) |
| `analysis/` | reachability, calibration, IK reports, benchmarks |
| `cli/` | `generate_dataset`, `eval_rollout`, probes and reference tools |

Note: "planner" in `control/` is **arm motion planning** (IK), distinct from the chess
engine's game planning.

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
| `pixi run -e eval eval-rollout` | closed-loop SmolVLA rollout in sim (eval env) |

Tests marked `slow` (CPU-expensive) and `render` (needs the osmesa software renderer) are
opt-in and are not run by `precommit`.


https://github.com/user-attachments/assets/2d835029-d066-4127-b617-9af2567fa565


