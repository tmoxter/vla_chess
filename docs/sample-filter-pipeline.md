# Sample-then-filter trajectory generation

## The pivot

Datagen used to treat `plan_move` as an a-priori feasibility gate: it solved an exact fingers-down
pose over every anchor and raised `PlanInfeasible` if one did not exist, so every episode that
started was expected to succeed and a failure was a surprise. That framing throws away the squares
no vertical pose reaches — the outer files and the near ranks — and it makes tilted grasps
impossible, because a leaned wrist grazes the executor's 5 mm IK abort (`MAX_IK_ERROR`) and dies
mid-run even when it would have placed the piece cleanly.

The pipeline here inverts that. **The planner becomes a candidate generator; physics and a post-hoc
validator decide what is kept.** A move is sampled into many candidate grasps and placements, each
is executed under a relaxed IK abort, and only the runs that physically end with the piece settled
on its target square are written to the dataset. Correctness is decided empirically, not by an IK
residual.

## The funnel

1. **Sample** (`control/sampling.py`). `sample_candidates` draws `SampleKnobs.attempts` candidates
   for a move. The first is the clean baseline — minimum tilt, centred placement. Each later one
   leans the grasp somewhere in `[0, tilt_jitter_deg]` and offsets the placement foot within a disk
   of `place_offset_radius`, with its own seed. A `CandidateSpec` fully describes an attempt, so a
   run is reproducible and loggable from it.
2. **Plan, best-effort** (`control/planner.plan_tilted_move`). Each anchor takes whatever lean its
   square needs up to `max_tilt_deg`, found by a restart-based **minimum-tilt search**
   (`VerticalGraspSolver.solve_min_tilt`), collision-screened with the jaws wide open. The place
   aims the piece's **foot** (not the TCP) at its square and sets it down `RELEASE_GAP` above the
   surface, so a leaned piece still settles on target. Manipulation legs are Cartesian with a
   per-leg `approach_axis`; the two survey transits and the approach/return stay joint-space. It
   returns a `Plan` and a `PlanQuality` (the lean and residual of each anchor) and only raises
   `PlanInfeasible` when a square has no screened pose at all — a move that cannot be *attempted*,
   as opposed to attempted and rejected.
3. **Execute** (`control/executor.Executor.run`, `ik_abort=TILTED_IK_ABORT`). The 5 mm gate is
   widened to 30 mm: a tilted leg tracks its eased axis at a fraction of full weight and drifts
   ~20–24 mm off the interpolated position while still placing the piece. The guard now only kills
   a truly diverging rollout; physics judges the rest.
4. **Validate** (`dataset/validate.validate`). A run is a keep only when it was never aborted, the
   piece was gripped, and it ended settled and upright on its destination (the executor's own
   `placed`, which bounds the final lean to the 10° settled cap). Everything else is a rejection
   with a `RejectReason`, so a move's yield can be read back per cause.
5. **Keep / discard** (`dataset/episode.EpisodeRunner.run_funnel`). Each candidate is run once
   without rendering to find a keeper; only the accepted candidate is re-run with the cameras
   attached, so rendering — the expensive half — is paid only for an episode that is kept. The
   funnel returns a `FunnelOutcome`: the kept episode (or none) and the verdict on every attempt.

## The generator

`generate-dataset --funnel` runs the funnel over the Exp 2 move pool, `--seeds` slots each,
`--attempts` candidates per slot (`--tilt-jitter`, `--place-offset` shape the spread). It writes a
`funnel-manifest.json` beside the captures: total slots, episodes kept, and per-move
attempts/kept/reject tallies, so the yield and the hard squares are visible.

## What it buys

Moves the fingers-down planner could not place now place through physics: outer files (`a7→a6`,
`h7→h6`) and two-step pushes (`a7→a5`), verified end to end. Extreme near-rank squares (`a4→a3` at
~34°) still fail — as they should — and the validator drops them with a reason rather than
poisoning the dataset.

## Provenance

This productionises the `scratch_tilt_pickplace_v2.py` prototype's confirmed pieces (min-tilt
restart search, foot-on-target placement, hover back-off along the approach axis) and resolves the
two failure modes it was bracketed by: pure joint-space between anchors swings the piece through
uncontrolled arcs (knocks it off-square), and Cartesian tilted tracking grazes the old 5 mm gate.
The fix is Cartesian tilt-tracking with the abort relaxed — the piece stays on its path and the
graze is tolerated. See `experiment-2-plan.md`, "Step 2 progress".
