# Experiment 2 — Generalization across pawn-forward moves

## Goal

Move from memorization to **spatial generalization**: a policy that can move a pawn forward on
squares it was never trained on, by learning across many different pawn-forward moves — mixing
**one-step and two-step** forward moves.

## What Exp 1 established

- Pipeline works end-to-end (data → fine-tune → closed-loop sim eval).
- Overfit (8 episodes, center files c7/d7/e7/f7) **reproduces trained trajectories** in closed loop.
- **Fails to generalize** to a held-out square (g7→g6): "ghost motion" — replays memorized
  center-file trajectories, cycles the gripper over the wrong location, never traps the g-pawn.
- Loss converged by ~8k steps; we no longer chase loss→0 — Exp 2 stops on a held-out metric.

## Design decisions (defaults; ✅ = chosen, ❓ = needs confirmation)

- ✅ **fps = 10** (keep, to isolate the diversity variable; revisit smoothness later).
- ✅ **batch = 32** (L4 had huge VRAM headroom at 16).
- ✅ **No fixed step count** — eval-driven stopping on held-out rollout success (budget ~15–30k,
  checkpoint every 1–2k).
- ✅ **Distance encoded in the instruction** — "move the pawn one field forward" /
  "move the pawn two fields forward". Image+state alone can't disambiguate 1 vs 2 from the same
  start, so language must carry it (also tests language grounding).
- ✅ **Black pawns only** (rank 7 down the board), as in Exp 1 — keep scope controlled.
- ❓ **Episode target / datagen effort** — ~120 overnight locally vs rent a many-core CPU box.
- ❓ **Held-out split** — settled in Phase 1 step 3 below.

## Phase 1 — Data (the crux)

### Step 1 — Reachability
Run the repo's reachability probe to get the squares the SO-101 arm can actually pick and place.
This constrains the valid move set (not all 64 squares are reachable). Output: the reachable
square set (and any per-square grasp feasibility).

### Step 2 — Move pool (1- and 2-step)
Enumerate valid pawn-forward moves within reach:
- **1-step:** `src(file, rank) -> (file, rank-1)` for black, for every reachable src whose target is
  also reachable and on-board.
- **2-step:** `src -> (file, rank-2)`. Chess-legal 2-step is only from the pawn's home rank (rank 7,
  i.e. 7→5); we restrict 2-step to rank-7 starts to keep it interpretable, both squares reachable.
- Each move gets its distance-specific instruction string.

### Step 3 — Train / held-out split
Reserve a subset of squares/files **entirely** for eval (never trained) — the generalization
measurement, like g7 was in Exp 1. Record the exact split. Ensure held-out set includes at least
one full file and some 2-step moves.

### Step 4 — Volume & generation
Datagen is the bottleneck (~7 min/episode single-process, 2 CPUs). Coverage of distinct
(square, distance) pairs matters more than many seeds. Generate (resumable) → LeRobot v3.0 →
push new Hub repo `chess-pawns-multi-v1`, version-tagged. Parallelize across processes or rent a
CPU box if the count is large.

## Phase 1 results (steps 1–3) — 2026-09-03

**Step 1 — feasibility (SUPERSEDED 2026-09-05 — see "Grasp feasibility: vertical is too strict"
below).** The old FK-reachability framing here (61/64 "reachable", "reachability does not
constrain") was misleading and is retired: FK reach is not the binding constraint — **grasp
feasibility is**, and it is far stricter than reach.

### Grasp feasibility: vertical is too strict (finding, 2026-09-05)

- **Reach ≠ grasp.** The board is ~fully FK-reachable, but that says nothing about whether a
  piece can be picked up and placed. Do not cite the reachability probe (`docs/reachability.md`)
  or `docs/grasp-poses.md`'s fingers-down count as if the graspable set were settled — both are
  superseded framings.
- **Strictly vertical (fingers-down, ≤3° off-vertical) is unacceptable:** it yields only ~4
  usable pick-and-place moves and rank 4 is entirely absent. Vertical is dead as *the* grasp
  constraint.
- **Allowing the wrist to tilt off vertical is the fix.**
- **RECOUNT (2026-09-06, `scratch_recount_tilt.py`, supersedes the ~37 number):** position-first
  IK finds a *continuum* of position-valid poses at different tilts, so the honest per-square
  figure is the **minimum** tilt over many restarts, not whatever one seed lands on. The old
  `scratch_fullboard_feasibility.py` reported single-seed tilts and both under- and over-counted:
  it marked a4/h4 "out" at ~49° and d4/e4 "44°/borderline", when their true minima are 24° and
  **7.8°**. With min-tilt search (40 restarts, err 0 mm across the block), the kinematic ceiling is:

  | tilt budget | graspable fields |
  |---|---|
  | ≤ 45° | **50** |
  | ≤ 50° | 52 |
  | ≤ 55° | 54 |
  | ≤ 60° | 54 |
  | position-reachable | 58 (a1,b1,c1,f1,g1,h1 out) |

  Per-square minimum tilt (deg), rank 8 top → 1 bottom, files a–h:

  ```
  8    5  7  3  0  0  4  7  2
  7    7  2  6 10  8  5  4  5
  6    0  7  5  3  3  1  6  0
  5   10  2  2  4  4  2  2 10
  4   24 16 10  8  8 10 16 25
  3   41 33 28 25 25 28 33 40
  2   65 53 47 44 44 47 53 65
  1    —  —  —  80 80  —  —  —
       a  b  c  d  e  f  g  h
  ```

- **45° is no longer the binding constraint.** Ranks 3–8 (48 fields) are *all* graspable at
  ≤41°; rank 4 needs only ≤25°. The only squares that need >45° are the rank-2 edges (b2/c2/f2/g2
  at 47–53°, a2/h2 at 65°) and d1/e1 (~80°, effectively a side-grab). Loosening 45°→55° buys just
  +4 fields — the rank-2 edges sitting directly in front of the shoulder, i.e. exactly the poses
  collision + physics are most likely to trim. So **50 fields at ≤45° already clears the 30–40
  target with margin**; the binding constraint is now collision/physics screening, not the tilt
  cap. Keep 45° (a cleaner, lower-tilt dataset) unless physics trims ranks 3–8 harder than
  expected, in which case 55° is the cheap next step.
- **This is a kinematic ceiling only** — IK, no collision screen, no grasp+place physics.
  Collision + physics will trim it; treat 50 as an upper bound comfortably above the 30–40 target.
- **Open engineering work before a dataset:** (1) ✅ grasp solver relaxed to take the natural
  per-square tilt up to the cap (`solve()`/`solve_square()` gained `max_approach_deg`/`weight`,
  Step 1 done 2026-09-06); (2) 🔶 make transport/release tilt-aware so pieces land on the target
  square — a naive tilted grasp picks d6 but drops it on c5; (3) physics-validate the feasible
  field set to get the true fields, then build the move pool and split over them.

### Step 2 progress (tilt-aware transport/release, 2026-09-06, in progress)

Prototype: `scratch_tilt_pickplace_v2.py`. Confirmed pieces and design decisions, several
findings that will shape the production planner:

- **Min-tilt is seed-dependent.** A single position-first (weight=0) solve lands on a *random*
  tilt from the continuum; the per-square minimum needs a restart search that keeps the
  lowest-tilt seed (the recount method). Collision-screen the grasp with **jaws wide open**
  (`gripper_joint_range[1]`), matching `solve_reference_grasps`; screening at the near-closed
  standoff opening wrongly rejects clean vertical poses (d6 came back 39–49° instead of 3°).
- **Foot-on-target placement + low release (Tobi's hint) works.** The gripped piece leans with
  the wrist, so foot = TCP + `grasp_height·axis`; aim the *foot* at the square (TCP =
  foot_target − `grasp_height·axis`) and release ~3 mm above the surface. Must **re-solve IK at
  that foot-compensated point** — solving over the square centre and commanding the shifted point
  left place/hover anchors 16–78 mm off and diverged the descent.
- **Hover must back off *along the approach axis*, not straight up**, and ≤~40 mm (vertical hover
  ~65 mm = the fingers-down ceiling; 60 mm put it at 85 mm, unreachable → descent from garbage).
- **Executor now honours a per-leg approach axis** (`Waypoint.approach_axis`, eased across the
  leg, tracked at `TILT_TRACK_WEIGHT`=0.3). Backward-compatible (None ⇒ fingers-down); 48/48
  control+grasp unit tests still pass. **Not yet wired into `plan_move`** — inert until Step 2
  lands.
- **Verdict so far:** vertical moves place correctly through the new path (d6→d5 ✓). Genuinely
  tilted moves (a4→a3, h4→h3 at 24–41°) are bracketed by two failure modes: **Cartesian** tilted
  tracking holds the piece path but diverges at ~5–6 mm (right at the 5 mm `MAX_IK_ERROR` gate) on
  descend/transport; **pure joint-space** never diverges but swings the piece through uncontrolled
  arcs between IK branches (d6→e6, a4 lost). **Next:** hybrid — joint-space for the short axial
  legs (descend/lift/place/release) between branch-consistent anchors, Cartesian only for
  transport; anchors solved as tracked translations of the grasp/place pose so they stay in one
  IK branch. Open question if that still grazes the gate: a slightly looser `MAX_IK_ERROR` for
  tilted legs, or split transport into two waypoints.

**Step 2 & 3 below are the OLD vertical-grasp pool (24 moves / 16 trained), kept for history but
SUPERSEDED 2026-09-05** — that pool assumed fingers-down feasibility and "safely graspable like
Exp 1", which the finding above retired. The real pool will be built over the tilted-grasp
feasible field set (~50 candidates at ≤45°, to be physics-validated), not these.

**Step 2 — Move pool (24 moves):** black pawns, forward = decreasing rank. Files a–h ×
{1-step from rank 7 (7→6), 1-step from rank 6 (6→5), 2-step from rank 7 (7→5)}. Start squares on
ranks 6–7, targets on ranks 5–6 (mid-board, safely graspable like Exp 1). Instructions:
"move the pawn one field forward" (1-step) / "move the pawn two fields forward" (2-step).

**Step 3 — Train / held-out split:**
- **Held-out (8 moves, eval-only — the rollout harness builds these on the fly, so no dataset is
  generated for them):**
  - Unseen files: `b7→b6, b6→b5, b7→b5`, `g7→g6, g6→g5, g7→g5`
  - Unseen square on a seen file: `d6→d5`
  - Unseen 2-step on a seen file: `f7→f5`
- **Trained (16 moves):** files {a, c, d, e, f, h} × {7→6, 6→5, 7→5}, minus `d6→d5` and `f7→f5`.
  - Trained 2-step: `a7→a5, c7→c5, d7→d5, e7→e5, h7→h5`.
- Tests three generalization axes: **unseen file** (b, g), **unseen square on a seen file**
  (d6→d5), **unseen distance on a seen file** (f7→f5).

**Step 4 (needs a call):** episodes = 16 trained moves × N seeds (per-seed board-pose jitter is the
augmentation). ~4 seeds → 64 eps, ~6 seeds → 96 eps. Held-out moves are NOT generated.

**Datagen CLI — DONE (2026-09-04):** `generate_dataset.py` gained a `--move-pool` mode:
`EXP2_TRAINED_MOVES` (the 16 trained pushes), each repeated `--seeds` times (default 6 → 96 eps),
single-pawn board per move, distance carried in the label via `forward_instruction()`
(1-step → "move the pawn one field forward", 2-step → "two fields forward"). Verified: 96 eps, 6×
each move, held-out `d6→d5`/`f7→f5`/`b`/`g` all absent. 16 datagen tests + mypy + ruff clean.
Run: `pixi run -e datagen generate-dataset --move-pool --seeds 6 --fps 30 --out data/chess_pawns_multi --repo-id <hub>/chess-pawns-multi-v1`.

**Pool EXPANDED 2026-10-10 (31 trained / 3 held-out) — SUPERSEDES the 16/8 split above.** Once the
tilted-grasp funnel plus the wrist-roll fix were in, a physics sweep (real screened plan + dry-run
on the single-pawn layout, baseline candidate per move) showed the outer files and a deeper rank
place cleanly, so the pool was broadened:
- **Trained (31):** mid-board on **all eight files** {7→6, 6→5, 7→5}; plus **rank-4 depth** —
  one-step `5→4` on {a,b,c,d,f} and two-step `6→4` on {c,d,e,f}. Every entry grasps+places on the
  baseline candidate. The right-edge rank-4 pushes `g5→g4`/`h5→h4` do **not** place on baseline (a
  left/right asymmetry from the shoulder offset) and are excluded until the sampling funnel is shown
  to recover them.
- **Held-out (3, eval-only):** `d6→d5` (unseen square), `f7→f5` (unseen 2-step), `e5→e4` (unseen
  depth: a rank-4 one-step the policy only ever saw reached via a two-step).
- **Tradeoff:** training b/g gives up Exp 2's original unseen-**file** generalization axis; the
  held-out set now probes unseen **square** and unseen **depth** instead. Reach frontier confirmed:
  clean down to rank 4 (centre), rank 3 is where pieces reach the square but topple, rank ≤2 is
  past the 45° tilt cap / out of position reach.

**Datagen cost (measured 2026-09-03):** datagen is ~100% **render-bound**. Local osmesa render is
~2.6 s/frame on 2 CPUs; sim is ~0.41 ms/mj_step (~3 s/episode, negligible). So wall time scales
~linearly with fps (frames ∝ fps): ~7 min/ep @10fps, ~14 @20, ~21 @30. 96 eps @30fps ≈ 17 h
locally — too slow.

**Decision — GPU datagen path (chosen):** MuJoCo physics is always CPU; only rendering uses the
GPU, so running datagen on a GPU pod *is* the "CPU-sim / GPU-render" split, in one process (no
cross-machine render RPC — not worth it). On a GPU, render drops 2.6 s → ~ms, so an episode goes
~7 min → ~5 s; 96 eps @30fps ≈ ~8 min. `sensor/render.py` now accepts `MUJOCO_GL=egl` (GPU)
alongside `osmesa` (local CPU, default, unchanged) — a **parallel** option, nothing removed. Plan:
generate at **30 fps on the GPU pod** (also natively fixes the Exp 1 jitter). Remaining to wire:
pod setup installs the package + datagen deps + EGL and sets `MUJOCO_GL=egl`; validate egl renders
match osmesa (pixel parity) before trusting the dataset; then generate + push. Note: 30 fps makes
CPU rollout eval ~3× slower, so eval likely moves to GPU too.

## Phase 2 — Training

- Reuse `runpod/` tooling on a fresh GPU pod.
- **Fix the Exp 1 disk crash:** checkpoints + metrics → `/workspace` (big persistent volume), not
  the 30 GB root overlay.
- `smolvla_base`, batch 32, held-out val split, checkpoint every 1–2k, eval-driven stop.
- Emit a metrics JSONL on `/workspace` during training (step, loss, lr, grad_norm, timing).

**Pod plumbing wired 2026-10-19.**
- **Pod-side datagen:** `runpod/datagen.sh` builds an EGL env (CPU torch + lerobot[dataset]==0.6.1
  + mujoco + the package) and runs `generate-dataset` under `MUJOCO_GL=egl`, parity-probing first.
- **Hub push + v3.0 tag:** `generate-dataset --push` serializes then calls
  `writer.push_dataset`, which pushes via `LeRobotDataset.push_to_hub(tag_version=True)` — that
  creates the `v3.0` tag `get_safe_version` needs (the old `upload_folder` gap).
- **`runpod/exp2.env`:** batch 32, `/workspace` output + log, fresh dataset/policy repos, funnel
  `--one-step-only --single-task` datagen args. Sourced by `bootstrap.sh`; does not touch the
  Exp-1 overfit defaults baked into `train.sh`.
- **lerobot pinned to 0.6.1** in `pod_setup.sh` (matches the datagen/eval pixi pins).
- **Smoke test (`runpod/smoke.env`):** a tiny fingers-down vertical slice run through the exact
  same pipe to de-risk it before the real harvest —
  `DATAGEN=1 ENV_FILE=runpod/smoke.env ./runpod/bootstrap.sh <host> <port>`.
- **metrics JSONL — DEFERRED.** lerobot-train has no native JSONL hook (its only logger is W&B and
  the console line abbreviates step counts via `format_big_number`), so `train.sh` persists the
  full console log to `${METRICS_DIR}/train.log` on `/workspace` for now; a faithful structured
  `metrics.jsonl` needs a small lerobot logging shim, not yet built.

## Phase 3 — Evaluation (quantitative)

- Add **programmatic success detection** to `eval_rollout.py`: final pawn square via
  `board.world_to_square` vs target → success/fail (no more eyeballing).
- **Batch rollouts:** trained squares (sanity) + all held-out squares → **success rate trained vs
  held-out** as the headline metric. Videos kept as artifacts.

**Implemented 2026-10-19.** `Executor.placement(data, source, target) -> PlacementCheck`
(final_square + settled + placed) factors the oracle's own final-square/settled test out of `run`,
so the rollout scores a move exactly the way datagen does. `eval_rollout` now prints SUCCESS/FAIL
per rollout and takes `--moves "c7:c6,d6:d5,g7:g6"` to load the policy once, drive each move, and
report a trained-vs-held-out success rate (one mp4+npz per move, exit 1 if any fail). Unit-tested
in `tests/unit/control/test_executor_placement.py` + `tests/unit/cli/test_eval_rollout.py`.

## MLflow tracking

Local MLflow (SQLite) is set up (`track` pixi env, `mlflow/mlflow.db`, `pixi run -e track
mlflow-server`). Flow: pod writes `metrics.jsonl` to `/workspace` during training → scp back → local
ingest script replays it into the SQLite DB (`log_metric(..., step=)`) with params + artifacts.
One master DB accumulates every experiment; local eval logs straight into it.

## Open decisions

- Episode target / datagen effort (local overnight vs rented CPU box).
- Final held-out split (from Phase 1 step 3).
