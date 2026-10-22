# vla-chess

**A robot that plays physical chess by combining a chess engine's *planning* with a
learned vision-language-action policy's *execution*.**

The long-term goal is an end-to-end system that, shown a chessboard, both **decides** the
move (a chess engine — the "what") and **physically makes** it (a fine-tuned VLA policy
driving an SO-101 arm to pick up the piece and place it on the target square — the "how").
The two halves are deliberately separate concerns:

- **Planning — the chess engine.** A standard engine (e.g. UCI/Stockfish-class) chooses a
  legal, strong move from the board state: `source → target`, plus captures/promotions.
  This is symbolic reasoning over the game, not motor control. *(Integration is future
  work; see Status.)*
- **Execution — the VLA policy.** Given the chosen move as a language instruction plus the
  arm's camera and proprioception, a fine-tuned **SmolVLA** policy outputs joint actions
  that carry out the pick-and-place in the physical (here, simulated) world. This is where
  the hard robotics problem lives, and it is what the project is building and validating now.

The engine turns a board into an intention; the VLA turns an intention into motion. The
interface between them is a move + an instruction string, which keeps each half independently
testable and swappable.

## Why this is hard (and interesting)

Chess *planning* is effectively solved. Reliable physical *execution* of an arbitrary move —
grasping a specific piece and placing it on a specific square, across the whole board, under
real sensing — is not. This repo is a focused testbed for that execution problem: a clean,
language-conditioned, repeatable pick-and-place task with unambiguous success criteria (did
the right piece end up settled on the right square), where the planning signal is trivial to
generate so all the difficulty is in the sensorimotor policy.

## Approach

The execution policy is trained by **imitation learning** on simulated demonstrations, in
three phases:

1. **Trajectory data generation (sim).** A scripted/IK "oracle" produces pick-and-place
   demonstrations in MuJoCo using privileged state — approach, descend, grasp, lift,
   transport, place, retreat — recorded as LeRobot episodes (wrist camera + 6-D joint/
   gripper state + action + instruction). A **sample-then-filter funnel** makes the planner
   a *candidate generator* and lets physics + a post-hoc validator decide what is kept, so
   tilted grasps and hard squares are harvested empirically rather than gated a priori.
   See `strategy-step-1.md`, `docs/sample-filter-pipeline.md`.
2. **Supervised fine-tuning.** Fine-tune a pretrained `lerobot/smolvla_base` on the dataset
   (rented GPU). See `strategy-step-2.md`, `runpod/`.
3. **Closed-loop evaluation (sim).** Reconstruct an episode's start scene and let the policy
   drive the arm autonomously at the dataset frame rate, scoring whether the piece lands.
   See `src/vla_chess/cli/eval_rollout.py`.

## Status

- ✅ **Phase 1 — data generation:** working. Scripted/IK oracle + tilted-grasp sampling
  funnel + validator + LeRobot writer; reachability/grasp analysis tooling.
- ✅ **Phase 2 — SFT:** working end to end on RunPod (EGL datagen render + GPU training).
  Two fine-tunes trained to low loss (~0.02).
- ⚠️ **Phase 3 — closed-loop:** **open problem.** Both fine-tunes learn the open-loop
  state→action map almost perfectly (train loss ~0.02, ~1° open-loop action error) yet
  **fail to execute the move closed-loop** — the arm drifts off the demo manifold at the
  grasp (classic covariate shift). More open-loop data diversity, longer training, frame
  rate, and action-chunk/smoothing knobs were all tried and **do not fix it**. Full
  write-up: **`docs/closed-loop-failure-report.md`**. Next lever is training-side, on the
  policy's own error distribution — **DAgger / on-policy relabeling** and **recovery-state
  coverage**.
- ⛔ **Planning (chess engine) integration:** not started — deferred until execution is
  reliable. The move source is currently a scripted instruction, which stands in for the
  engine's output.

## Architecture

```
chess engine ──move(src→tgt)──▶ instruction ──▶  VLA policy ──joint actions──▶  SO-101 arm (MuJoCo)
  (planning, future)                                (execution)                    │
                                                                                   ▼
oracle/IK ──▶ sample ──▶ plan(tilt) ──▶ execute ──▶ validate ──▶ LeRobot dataset ──▶ SFT ──▶ rollout eval
          (data-generation funnel: physics decides what is kept)
```

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

## Key documents

- `strategy-step-1.md` — Phase 1 (data generation) strategy; `implementation-plan-step-1.md`
- `strategy-step-2.md` — Phase 2 (SFT) strategy
- `experiment-2-plan.md` — the diverse-funnel experiment plan
- `docs/sample-filter-pipeline.md` — the sample-then-filter funnel
- `docs/closed-loop-failure-report.md` — findings from the two fine-tunes (the current wall)
- `runpod/` — pod scripts for GPU datagen (`datagen.sh`) and training (`train.sh`)
