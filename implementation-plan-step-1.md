# VLA Chess — Phase 1 Implementation Plan: Trajectory Data Generation

Implementation plan for the phase described in `strategy-step-1.md`. Scope is strictly the
two Phase 1 deliverables: (1) a scripted/IK trajectory **generation engine**, and (2) a
repeatable **quality/validity monitoring workflow** for the data it produces. No SFT
training run, no RL, no real SO-101 hardware, no teleop tooling.

---

## 0. Ground rules, assumptions, and decisions

These are the choices this plan commits to. Each is flagged where it could reasonably be
revisited; anything marked **OPEN** should be resolved before the step that consumes it.

### Environment constraints (from the strategy doc, re-verified on this host)

| Constraint | Value | Consequence for the plan |
|---|---|---|
| CPU | 2 vCPU | One simulation process. No multiprocessing pool of MuJoCo instances. |
| RAM | ~3.9 GB total, ~2.3 GB available | Hard RSS budget of **1.6 GB** for the generator; stream frames to disk, never buffer a full batch in memory. |
| GPU | none | `MUJOCO_GL=osmesa` software rendering only. Never `egl`, never `glfw`. |
| Disk | ~206 GB free | Ample; video-encoded dataset for ~1000 episodes is expected in the 5–25 GB range. |

Because rendering is the throughput bottleneck on CPU, **every design choice that trades
control frequency or episode count against render cost is made explicitly in Step 2 and
re-calibrated in Step 15**, not guessed up front. Image resolution is the one knob excluded
from that trade: §0.10 fixes it at 512², and only an RSS-ceiling breach reopens it.

### Technical decisions

1. **Simulator**: MuJoCo (Python bindings), single process, `MUJOCO_GL=osmesa`.
2. **Robot model**: the SO-101 / SO-ARM100 MJCF (5 revolute joints + 1 gripper joint), so
   the action space produced now matches the arm the project intends to deploy on later.
   **RESOLVED**: source is
   [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100), path
   `Simulation/SO101/`, license Apache-2.0. Meshes and MJCF were generated via
   onshape-to-robot from the CAD model; `scene.xml` is the entry point and includes one of
   two calibration variants (`so101_new_calib.xml`, zero at mid-range — default; or
   `so101_old_calib.xml`, zero at full horizontal extension). Step 3 vendors this into
   `assets/so101/` (submodule or vendored copy, pinned to a commit SHA, not tracked
   upstream) and pulls in `so101_new_calib.xml`.
   Two known gaps to close in Step 3, not deferred further: base collision meshes were
   removed upstream (simulation issues) and must be replaced with simple convex proxies
   for our contact/pick-place needs; and the gripper's linear joint (0 = closed, 100 =
   open) is not yet reflected in the upstream MJCF/URDF and must be added or corrected
   before Step 5 depends on it.
   No fallback model: if the vendored asset fails Step 3's compile/reachability acceptance
   criteria, Step 3 fails hard and blocks the plan until the SO-101 asset is fixed. The
   project depends on matching the arm it will deploy on — silently substituting a
   different arm would produce a dataset that doesn't serve that goal.
3. **Action space**: absolute target joint positions, 6-dim `[j1..j5, gripper]`, in the
   same units/space as `observation.state`. This mirrors what a real follower arm consumes
   and is what SmolVLA is normally fine-tuned on. Deltas are *not* used.
4. **IK**: [`mink`](https://github.com/kevinzakka/mink) (differential IK via joint-limited
   least-squares over MuJoCo models), used directly rather than a hand-written DLS solver.
   Rationale: maintained, tested solver purpose-built for this exact MuJoCo-model-to-pose
   problem; avoids spending Phase 1 budget re-deriving damped least-squares, nullspace
   biasing, and singularity handling that `mink` already provides. Step 5 wraps it behind
   the same `IKSolver`/`IKResult` interface so downstream steps are solver-agnostic.
5. **Dataset format**: `LeRobotDataset` v2.1+ (parquet episode chunks + mp4 video streams +
   `meta/` json). Written through a thin `DatasetWriter` interface so the underlying
   implementation (lerobot API vs. direct parquet/mp4 emission) can be swapped without
   touching the generation engine.
6. **Episode granularity**: one episode = one complete pick **and** place, carrying a single
   task string (e.g. `"move the pawn from b2 to b4"`). Per-phase sub-instructions
   (`"pick up the piece on b2"`, `"place the piece on b4"`) are stored as frame-level
   metadata with phase boundaries, so a phase-segmented variant of the dataset can be
   emitted later without regenerating anything.
7. **Determinism**: every episode is fully described by a serializable `EpisodeSpec`
   containing its master seed. Given the same spec, the engine must reproduce the episode
   bit-for-bit (physics and sampled randomization; rendered pixels compared with a small
   tolerance). This is what makes failures debuggable and the validator meaningful.
8. **Language**: Python 3.11+ managed by pixi; mypy strict, ruff, pytest — per `AGENTS.md`.
9. **Cameras**: wrist-mounted only — the overhead `top` camera is dropped. Measured on this
   host (`docs/render-budget.md`), two cameras cost noticeably more render time than one at
   every resolution tested; dropping to one camera buys back a meaningful fraction of
   throughput (see §0.10 for how that throughput is spent). This is a real capability
   trade-off, not free: Step 11's `PIECE_VISIBLE_AT_START` check was designed against the
   overhead camera's stable, always-on-board view. A wrist-mounted camera's view depends on
   the arm's start pose and may not frame the source square at frame 0. Step 11 must replace
   that check with a privileged-state predicate (source piece pose is on-board and within
   tolerance of the intended square at frame 0, checked from `MjData`, not pixels) rather
   than a pixel-segmentation check — tracked as a required change, not deferred. Any other
   per-episode or aggregate check that assumed two camera keys (e.g. dataset integrity's
   "every episode has both camera keys") is a single-key check now.
10. **Resolution**: fixed at **512×512**, not chosen to clear an fps threshold. SmolVLA
    compresses every frame to a fixed 64 visual tokens via PixelShuffle regardless of input
    resolution (512² is literally SmolVLA's own published reference example), so there is no
    model-side compute cost to a higher resolution, and published SmolVLA/LeRobot recipes
    commonly train at 256–400px, not 128px. Generating at 128² and calling it done — the
    original plan, before this was raised — would have meant upsampling permanently blurry,
    low-detail sim frames to whatever resolution the real camera's frames get downsampled to
    at inference, creating a real train/deploy domain gap (missing edge and texture detail,
    not just a smaller array) independent of whatever the fps math said. That risk is worse
    than a slower generation run, so it is not a trade this plan makes to hit a throughput
    number.
    RSS is a different kind of constraint and stays a **hard, non-negotiable ceiling**: this
    host physically has ≤ 1.6 GB to give the generator, so if 512² peak RSS exceeds that
    ceiling, Step 2 is genuinely blocked and Step 15 must explicitly recalibrate (documented
    resolution/episode-length reduction) — that is a real fail-hard case, not the same thing
    as the fps trade-off above. fps, in contrast, is not a physical ceiling: a low measured
    fps at 512² is recorded and feeds Step 15's episode-count and capture-fps calibration
    (fewer episodes, lower capture Hz, longer wall-clock budget), not a reason to shrink the
    image. `MIN_CAMERA_FPS` (4.0) is kept only as a reference point in the recorded rationale,
    not as a gate that picks the resolution.

### Proposed repo layout

```
vla_chess/
  pyproject.toml            # pixi manifest + ruff/mypy/pytest config
  src/vla_chess/
    __init__.py
    config.py               # typed config objects, defaults, seed handling
    render.py               # osmesa renderer wrapper + camera specs
    assets/                 # MJCF scene fragments, meshes, textures
    scene.py                # scene assembly, board<->world mapping
    randomize.py            # EpisodeSpec + seeded samplers (scene + motion level)
    kinematics.py           # FK, Jacobian, DLS IK, joint-limit handling
    planner.py              # waypoint plan construction with motion randomization
    executor.py             # control loop, interpolation, gripper, phase tracking
    observe.py              # observation/action assembly, decimation
    instructions.py         # language instruction templates
    episode.py              # Episode dataclass + EpisodeRunner
    validate/
      episode.py            # per-episode checks
      batch.py              # aggregate checks
      diversity.py          # diversity metrics
      report.py             # markdown/HTML report rendering
    writer.py               # DatasetWriter (LeRobot format)
    cli/
      generate.py           # batch generation entrypoint
      validate_dataset.py   # aggregate validation entrypoint
      inspect_episode.py    # single-episode debug/replay entrypoint
  tests/
    unit/                   # fast, no rendering, tiny synthetic MJCF fixtures
    integration/            # marked `slow`, real MuJoCo + osmesa, low-res
  docs/
    runbook.md
    dataset-card.md
```

### Data schema (target, fixed at Step 12)

| Key | Shape / type | Notes |
|---|---|---|
| `observation.images.wrist` | `(H, W, 3)` uint8 | wrist-mounted camera (sole camera — see §0.9) |
| `observation.state` | `(6,)` float32 | 5 joint positions + gripper position |
| `action` | `(6,)` float32 | commanded absolute joint targets for next step |
| `task` | str | episode-level instruction |
| `timestamp`, `frame_index`, `episode_index`, `index` | — | LeRobot standard fields |

Episode-level metadata stored alongside (in `meta/episodes*.jsonl` extras or a sidecar
`meta/generation.jsonl`): full `EpisodeSpec`, seed, source/target squares, piece type,
phase boundary frame indices, validator verdict and metric values, engine git SHA.

---

## 1. Repository and toolchain scaffolding

**What gets built**: the pixi environment and quality gate that everything else is
developed inside. `pyproject.toml` with pixi dependencies (`python`, `mujoco`, `mink`,
`numpy`, `ffmpeg`, dev: `pytest`, `pytest-cov`, `mypy`, `ruff`), pixi tasks `test`, `test-slow`,
`typecheck`, `lint`, `format`, and a composite `precommit` task, wired to a git
`pre-commit` hook. Empty `src/vla_chess/` package with `py.typed`. Pytest markers `slow`
and `render` registered so CPU-expensive tests are opt-in.

**Interface**: `pixi run precommit` → runs ruff check, ruff format --check, mypy strict on
`src/`, pytest on `tests/unit`. Exit code 0/non-zero.

**Depends on**: nothing.

### Acceptance Criteria
- `pixi install` completes on this host from a clean checkout.
- `pixi run precommit` exits 0 on the scaffold (with at least one trivial unit test present).
- `mypy --strict src/` reports 0 errors; the config actually enables strict mode
  (`disallow_untyped_defs`, `warn_return_any`, `no_implicit_optional`).
- A deliberately introduced type error and a deliberately introduced lint error each cause
  `pixi run precommit` to exit non-zero (verified once, then reverted).
- `git commit` with the hook installed runs `precommit` and blocks on failure.
- `pytest -m "not slow"` collects and passes; `pytest -m slow` collects 0 tests without error.

---

## 2. Headless rendering layer and performance budget

**What gets built**: `render.py` — a thin wrapper over `mujoco.Renderer` that asserts the
`osmesa` backend, owns camera definitions (name, resolution, fovy, pose), and renders one
RGB frame per camera per call with an explicit, reused output buffer (no per-frame
allocation). Plus a benchmark script that measures achievable render throughput and RSS on
this host across a small resolution grid, and writes the result to
`docs/render-budget.md`.

**Interface**:
```python
class CameraSpec:  # name, width, height, fovy, attached_body | freejoint pose
class Renderer:
    def __init__(self, model: mujoco.MjModel, cameras: Sequence[CameraSpec]) -> None
    def render(self, data: mujoco.MjData) -> dict[str, np.ndarray]  # uint8 (H,W,3)
    def close(self) -> None
```

**Depends on**: Step 1.

**Why this is a separate, early step**: resolution is fixed by §0.10 (512×512, matching
SmolVLA's reference resolution), but the render budget measured here still determines
control frequency and realistic episode count at that resolution. Every later step's cost
estimate hangs off the number produced here, so it is measured before anything is built on
top of it.

### Acceptance Criteria
- `MUJOCO_GL` is verified to be `osmesa` at import time; constructing a `Renderer` with any
  other backend raises a clear `RuntimeError` naming the required value.
- Unit test: rendering a fixture MJCF containing a single red box against a black
  background yields an array of the requested `(H, W, 3)` uint8 shape whose mean red
  channel exceeds mean green and blue by ≥ 20 counts (i.e. the pixels are actually the
  scene, not noise or a blank buffer).
- Unit test: two consecutive renders of the same `MjData` are bit-identical; a render after
  translating the box by 5 cm differs in ≥ 1% of pixels.
- Benchmark reports, for at least resolutions 128², 224², 256², 320², 512²: wrist-camera
  frames/sec and peak process RSS (the production camera count, per §0.9 — not a synthetic
  multi-camera sweep). Results committed to `docs/render-budget.md`.
- The working point is **512×512, fixed** (per §0.10 — not chosen by searching for the
  highest fps-passing resolution). At 512², peak RSS **must be ≤ 1.6 GB** — this is a hard
  ceiling on this host; exceeding it blocks Step 2 and requires an explicit, documented
  recalibration in Step 15, not a silent resolution drop. The measured fps at 512² is
  recorded against the 4 fps reference point for context but does not gate the resolution
  choice — a shortfall is absorbed by Step 15's episode-count/capture-fps calibration.
- No memory growth: rendering 2000 frames in a loop increases RSS by < 50 MB from the
  100th frame onward.

---

## 3. Scene assets and board coordinate model

**What gets built**: the MuJoCo scene — arm + table + chessboard + pieces — assembled from
MJCF fragments in `assets/`, and `scene.py`, which is responsible for (a) building an
`MjModel` from a scene description and (b) the algebraic mapping between chess square names
and world poses.

Must declare `<visual><global offwidth="512" offheight="512"/></visual>` (or larger): MuJoCo
defaults the offscreen framebuffer to 480px, and `mujoco.Renderer` raises a hard `ValueError`
constructing a context above that size. Found the hard way benchmarking `bench_scene.xml` at
512² (§0.10) — every scene MJCF that gets rendered at the target resolution needs this
explicitly, it is not inherited from anywhere else.

Includes: vendoring the SO-101 MJCF from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)`/Simulation/SO101/`
(Apache-2.0, pinned to a commit SHA) into `assets/so101/`, using `so101_new_calib.xml`;
adding simple convex proxy collision geometry for the base (upstream ships without base
collision meshes) and fixing up the gripper's linear joint (0=closed, 100=open) mapping,
which is not yet correct in the upstream files; a parametric board body (8×8 squares, known
square pitch, placed on the table with a settable position + yaw); simple convex piece
geometries (at minimum pawn/rook/bishop/knight/queen/king as primitive-composed bodies with
free joints), each with mass/friction properties in a plausible range; the wrist camera
from Step 2 (per §0.9, the sole camera — no overhead `top` camera).

**Interface**:
```python
@dataclass(frozen=True)
class BoardPose:  # world xy position, yaw, square_pitch
class Board:
    def square_to_world(self, square: str, board: BoardPose) -> np.ndarray  # (3,) centre, on board surface
    def world_to_square(self, xyz: np.ndarray, board: BoardPose) -> str | None
    def squares(self) -> list[str]  # "a1".."h8"
def build_scene(desc: SceneDescription) -> tuple[mujoco.MjModel, SceneIndex]
# SceneIndex: name -> body/joint/geom/site ids for arm, gripper site, pieces, board
```

**Depends on**: Steps 1–2.

### Acceptance Criteria
- `build_scene` returns a model that compiles and steps 1000 times with no MuJoCo warnings
  and no `NaN` in `data.qpos`/`data.qvel`.
- Vendored SO-101 asset: base body has convex collision geometry (a dropped test object
  rests on/collides with the base rather than passing through it), and the gripper joint's
  `[0, 100]` range is verified to correspond to closed/open by driving it to each extreme
  and checking the fingertip site separation increases monotonically with joint value.
- Unit test (pure math, no simulator): `world_to_square(square_to_world(s, pose)) == s` for
  all 64 squares across ≥ 20 randomized board poses (position within the reachable table
  region, yaw in [-π, π]).
- Unit test: a point offset by 0.45 × square pitch from a square centre still maps to that
  square; offset by 0.55 × pitch maps to an adjacent square. A point 3 pitches off the board
  maps to `None`.
- Unit test: `square_to_world` returns centres whose pairwise spacing along board axes
  equals the configured pitch to within 1e-9 m, and whose z equals board surface height.
- Integration test (`slow`): a piece dropped 2 cm above a square settles within 200 sim
  steps to a resting pose whose xy is within 5 mm of the square centre and whose tilt from
  vertical is < 5°, i.e. pieces stand up rather than roll or fall through the board.
- Every square reachable by the arm is enumerated by a reachability probe and the result is
  recorded in `docs/reachability.md`. **If some squares are unreachable at the nominal board
  pose, the board placement is adjusted until ≥ 60 of 64 squares are reachable, or the
  unreachable set is explicitly excluded from sampling in Step 4** — no silent failures.
- Scene loads and renders through Step 2's `Renderer` producing a visibly non-uniform image
  (pixel standard deviation > 10 counts on the wrist camera).

---

## 4. Episode specification and seeded randomization (scene level)

**What gets built**: `randomize.py` — the `EpisodeSpec` dataclass (the complete, serializable
description of one episode) plus a seeded sampler producing specs. Covers the scene-level
domain randomization the strategy doc requires: board position/orientation, piece style,
lighting, plus camera jitter, distractor pieces, and physical property jitter.

Randomized fields and their ranges (ranges are config, defaults recorded in `config.py`):
board xy ±3 cm, board yaw ±15°, piece type from the enabled set, piece visual style/colour,
piece scale ±5%, piece mass/friction ±20%, light position/direction/diffuse intensity,
ambient level, table/board texture variant, the wrist camera's pose jittered ±1 cm / ±2°,
0–6 distractor pieces on other squares, source and target squares.

**Interface**:
```python
@dataclass(frozen=True)
class EpisodeSpec:
    seed: int
    source_square: str
    target_square: str
    piece_type: str
    board: BoardPose
    lighting: LightingSpec
    cameras: dict[str, CameraJitter]
    physics: PhysicsJitter
    distractors: tuple[DistractorSpec, ...]
    motion: MotionSpec        # filled in by Step 6's sampler
    def to_json(self) -> str
    @classmethod
    def from_json(cls, s: str) -> EpisodeSpec

class SpecSampler:
    def __init__(self, cfg: RandomizationConfig, master_seed: int) -> None
    def sample(self, index: int) -> EpisodeSpec
```

Square-pair sampling is **coverage-driven, not uniform-iid**: the sampler maintains counts
and prefers under-represented source and target squares, so a finite batch does not leave
squares unvisited by chance.

**Depends on**: Step 3.

### Acceptance Criteria
- Determinism: `SpecSampler(cfg, seed).sample(i)` returns an identical spec across processes
  and runs, for 100 values of `i`. Round-trip `EpisodeSpec.from_json(spec.to_json()) == spec`.
- Independence: two samplers with different master seeds produce specs differing in every
  randomized field for ≥ 95% of indices.
- Range compliance: over 10 000 sampled specs, every randomized scalar lies inside its
  configured range, and the min/max observed are each within 5% of the range bounds
  (i.e. the ranges are actually exercised, not clustered at the centre).
- Coverage: over a simulated 640-spec batch, each of the reachable squares appears as source
  ≥ 5 times and as target ≥ 5 times, and `max_count / min_count ≤ 2.0`.
- Source ≠ target in 100% of specs; distractor squares never collide with source or target.
- Applying a spec to a scene is verified: `apply_spec(model, data, spec)` yields a board yaw
  matching the spec to within 1e-6 rad and a source piece resting on the source square
  (per Step 3's `world_to_square`).
- Unit tests run with no rendering and complete in < 5 s total.

---

## 5. Kinematics: FK, Jacobian, and IK via `mink`

**What gets built**: `kinematics.py` — forward kinematics for the gripper site (via
`mujoco.mj_kinematics`/site lookup) and an `IKSolver` that wraps `mink`'s differential IK
(frame task on the gripper site, joint-limit and posture-bias tasks) to map a target
end-effector pose to joint angles. The SO-101's 5 DoF cannot generally satisfy a full 6-DoF
pose, so orientation is configured as a weighted task, not a hard constraint, using `mink`'s
task-weighting rather than a hand-rolled nullspace bias.

**Interface**:
```python
@dataclass(frozen=True)
class IKResult:
    qpos: np.ndarray          # (n_joints,)
    position_error: float     # metres
    orientation_error: float  # radians
    iterations: int
    converged: bool

class IKSolver:
    def __init__(self, model, site_id, joint_ids, cfg: IKConfig) -> None
    def solve(self, target_pos, target_quat, q_init) -> IKResult
    def fk(self, qpos) -> tuple[np.ndarray, np.ndarray]  # pos, quat
```

**Depends on**: Step 3.

### Acceptance Criteria
- Round-trip unit test: for 500 random joint configurations sampled inside limits, compute
  FK, then solve IK from a perturbed initial guess (±0.3 rad per joint). **≥ 98% converge to
  position error ≤ 2 mm**, and median iteration count ≤ 50 (`mink` solver iterations).
- Solves that fail return `converged=False` — the solver never returns a silently bad pose,
  and never returns `NaN` (asserted over the same 500 cases plus 100 deliberately
  unreachable targets 1.5 m away).
- All returned `qpos` lie within model joint limits, inclusive, for every test case
  (`mink`'s joint-limit task configured, and re-verified independently in the test).
- Singularity robustness: at a fully extended (near-singular) configuration, the solver's
  per-iteration joint step norm stays ≤ a configured cap (no blow-up); test asserts no step
  exceeds 0.5 rad.
- Posture bias: for a target reachable by multiple configurations, the returned solution's
  distance to the neutral posture is lower than that of the same solve with `mink`'s posture
  task weight set to zero, in ≥ 80% of 100 trials.
- Determinism: identical inputs give bit-identical `IKResult` across runs.
- Performance: median solve time ≤ 5 ms on this host (IK must not become the bottleneck
  alongside rendering).
- If the 98% convergence criterion is not met after tuning `mink` task weights and damping,
  that is a scene/target-range problem to fix (e.g. reachability bounds from Step 3), not a
  solver swap — `mink` is the committed solver, not a candidate.

---

## 6. Waypoint planning with motion-level randomization

**What gets built**: `planner.py` — construction of the pick-and-place waypoint sequence
(approach above source → descend → close gripper → lift → transport → descend over target →
release → retreat) in Cartesian space, with the strategy doc's motion-level diversity
injected per episode.

Randomized per episode (into `MotionSpec`, part of `EpisodeSpec` so it stays reproducible):
- **approach azimuth** — the horizontal direction the gripper comes in from, sampled over
  the full circle subject to reachability, and **approach elevation/tilt** ±20°;
- **approach and retreat heights** — pre-grasp hover height, lift height, transport height,
  each independently sampled over a configured range;
- **transport path shape** — straight-line vs. arc vs. via a randomized mid-waypoint, with
  arc curvature sampled;
- **waypoint timing** — per-segment duration sampled independently, plus a randomized
  velocity profile (trapezoidal vs. minimum-jerk, with random acceleration fraction);
- **grasp-closure timing** — gripper close command offset relative to arrival at the grasp
  pose, sampled in a window that includes both slightly-early and slightly-late closure;
- **waypoint jitter** — small isotropic noise on every intermediate waypoint position and on
  wrist roll;
- **start pose** — the arm's initial joint configuration perturbed around home.

Grasp pose selection accounts for piece geometry (grasp height as a fraction of piece height,
also jittered).

**Interface**:
```python
@dataclass(frozen=True)
class Waypoint:
    position: np.ndarray; quat: np.ndarray
    gripper: float            # commanded opening in [0, 1]
    duration: float           # seconds to reach from previous waypoint
    profile: VelocityProfile
    phase: Phase              # APPROACH|DESCEND|GRASP|LIFT|TRANSPORT|PLACE|RELEASE|RETREAT

class Planner:
    def plan(self, spec: EpisodeSpec, scene: SceneIndex, data: mujoco.MjData) -> list[Waypoint]
```

**Depends on**: Steps 3, 4, 5 (uses IK reachability checks to reject infeasible sampled
approach directions and resample, up to a bounded number of attempts).

### Acceptance Criteria
- Structural unit tests: every plan contains all eight phases in the correct order; gripper
  command is open before `GRASP`, closed from `GRASP` through `PLACE`, open from `RELEASE`
  onward; grasp waypoint xy is within 5 mm of the source piece xy; release waypoint xy is
  within 5 mm of the target square centre.
- Safety unit tests: no waypoint has z below the board surface; every transport-phase
  waypoint clears the tallest board piece height by ≥ 1 cm; total planned duration is within
  the configured `[min, max]` episode-length window.
- Feasibility: for 200 sampled specs, ≥ 95% of plans have **every** waypoint IK-solvable to
  ≤ 2 mm; plans that don't are returned as `PlanInfeasible` with the offending waypoint
  identified, never silently truncated.
- **Diversity unit tests (this is the core requirement of the strategy doc and is tested
  explicitly, not assumed):** over 1000 sampled plans —
  - approach azimuth, binned into 12 sectors of 30°, has **every reachable sector holding
    ≥ 3% of episodes** and no sector holding > 20%;
  - lift height, transport height, per-segment duration, and grasp-closure offset each have
    a coefficient of variation ≥ 0.10;
  - all three transport path shapes appear, each with frequency ≥ 15%;
  - no two plans in the 1000 are identical under a rounded-waypoint hash (1 mm / 10 ms
    quantization);
  - the mean pairwise L2 distance between time-normalized, resampled end-effector paths
    exceeds a floor recorded in `docs/diversity-baseline.md` (floor is set from the first
    measured run and thereafter regression-tested against).
- Determinism: same spec → identical plan.

---

## 7. Trajectory execution and control loop

**What gets built**: `executor.py` — converts a waypoint plan into an executed rollout.
Interpolates each segment according to its velocity profile, solves IK at the control rate,
issues position commands to the arm actuators and the gripper, steps physics, tracks phase
boundaries, and detects grasp success/failure. Runs the sim at a physics timestep with a
lower control rate (decimation), and a lower observation rate still (Step 8).

Also owns the abort conditions: NaN state, joint-limit violation, piece knocked off the
board, excessive contact force, IK divergence, wall-clock timeout.

**Interface**:
```python
@dataclass
class ExecutionResult:
    states: np.ndarray          # (T, 6) observed joint+gripper positions at control steps
    actions: np.ndarray         # (T, 6) commanded targets
    phase_boundaries: dict[Phase, tuple[int, int]]
    grasped: bool
    placed: bool
    abort_reason: AbortReason | None
    diagnostics: ExecutionDiagnostics   # max tracking error, max contact force, sim steps

class Executor:
    def run(self, model, data, plan: list[Waypoint], ik: IKSolver,
            on_control_step: Callable[[int, mujoco.MjData], None] | None) -> ExecutionResult
```

The `on_control_step` callback is how Step 8 hooks in observation capture without the
executor knowing anything about cameras or datasets.

**Depends on**: Steps 5, 6.

### Acceptance Criteria
- Integration test (`slow`): a nominal plan for a pawn from `d2` to `d4` executes with
  `grasped=True`, `placed=True`, `abort_reason=None`, and the piece's final resting xy within
  1.0 cm of the `d4` centre and tilt < 10°.
- The piece is verified as actually gripped, not pushed: during the `TRANSPORT` phase the
  piece's z stays ≥ 2 cm above the board surface for the whole phase, and its xy tracks the
  gripper site xy to within 2 cm.
- Tracking quality: max joint tracking error (commanded vs. achieved) ≤ 0.05 rad over the
  whole episode on the successful nominal case.
- Abort paths are individually tested and each returns the correct `AbortReason` with
  `states`/`actions` truncated consistently (equal lengths, no partial frame): unreachable
  target injected, piece removed from scene before grasp, wall-clock timeout forced with a
  1 s cap.
- `states` and `actions` have equal length T, contain no `NaN`/`inf`, and every action is
  within joint limits.
- Determinism: two runs of the same spec produce `states`/`actions` arrays identical to
  within 1e-9.
- Resource guard: an executed episode's peak RSS delta is < 200 MB, and running 20 episodes
  back-to-back in one process shows no monotonic RSS growth beyond 100 MB total.

---

## 8. Observation capture and framing

**What gets built**: `observe.py` — assembles per-frame observation records at the dataset
frame rate (a decimation of the control rate, chosen from Step 2's budget), calling the
Step 2 renderer and packing state/action/timestamp/phase. Handles the action-alignment
convention explicitly (frame `t` carries the observation at time `t` and the action commanded
to reach `t+1`) and the final-frame handling.

**Interface**:
```python
@dataclass
class Frame:
    images: dict[str, np.ndarray]
    state: np.ndarray; action: np.ndarray
    timestamp: float; frame_index: int; phase: Phase

class ObservationRecorder:
    def __init__(self, renderer: Renderer, fps: float, control_hz: float) -> None
    def on_control_step(self, step: int, data: mujoco.MjData) -> None   # matches Executor hook
    def frames(self) -> list[Frame]
```

**Depends on**: Steps 2, 7.

### Acceptance Criteria
- Unit test with a stub renderer: for a 10 s rollout at 50 Hz control and 15 Hz capture,
  exactly `round(10 * 15)` frames are produced; timestamps are monotonically increasing with
  spacing `1/15 ± 1e-6` s; `frame_index` is contiguous from 0.
- Alignment test: with a synthetic executor emitting `state[t] = t` and `action[t] = t+1`,
  the recorded frames satisfy `action[i] == state[i+1]` for all `i < T-1` — i.e. the
  convention is enforced by test, not by comment.
- Every frame contains the `wrist` camera key, uint8, of the configured shape.
- Image sanity per frame: pixel std > 5 counts and mean in `[10, 245]` — rejects all-black
  (renderer failure) and all-white (blown-out lighting) frames at the source.
- Wrist camera moves: over an episode, mean absolute inter-frame pixel difference on `wrist`
  exceeds that of a static-camera baseline, confirming the camera is actually attached to
  the moving body.
- Memory: frames are written through to the caller and the recorder holds at most a
  configured `max_buffered_frames` (default sized to ≤ 150 MB); exceeding it raises rather
  than silently growing.

---

## 9. Language instruction generation

**What gets built**: `instructions.py` — deterministic-per-seed generation of the episode
task string and the two phase sub-instructions, from a template bank with varied phrasing so
the policy is not conditioned on one canonical sentence.

Covers: piece-name vs. generic-"piece" references, square naming, and several verb/phrasing
templates (e.g. `"move the {piece} from {src} to {dst}"`, `"pick up the piece on {src} and
place it on {dst}"`, `"put the {piece} on {src} down on {dst}"`).

**Interface**:
```python
class InstructionGenerator:
    def episode_instruction(self, spec: EpisodeSpec) -> str
    def phase_instructions(self, spec: EpisodeSpec) -> tuple[str, str]  # pick, place
```

**Depends on**: Step 4.

### Acceptance Criteria
- Determinism: same spec → same strings.
- Correctness: for 1000 specs, the generated string contains the correct source square, the
  correct target square, in that order, and never contains a square not in the spec (regex
  extraction of all square tokens is asserted against the spec).
- Variety: ≥ 6 distinct templates observed over 1000 specs, none exceeding 30% frequency.
- Piece naming matches `spec.piece_type` whenever a piece noun is used.
- All strings are lowercase-normalized, ≤ 100 characters, ASCII, with no double spaces or
  trailing punctuation inconsistency (asserted).
- Phase instructions match the strategy doc's examples in form: pick instruction references
  only the source square, place instruction only the target square.

---

## 10. Episode runner (engine integration)

**What gets built**: `episode.py` — the single function that turns an `EpisodeSpec` into an
`Episode`, wiring Steps 3–9: build/reset scene → apply spec → settle physics → plan →
execute with observation capture → attach instructions and metadata. This is the generation
engine's public entry point.

Owns model reuse across episodes: because the board pose, camera jitter, piece styles etc.
change per episode, the runner decides what can be mutated on an existing `MjModel`
(positions, colours, lights, masses, camera poses) versus what forces a rebuild (piece count
changes). Rebuild frequency directly affects throughput, so this is measured.

**Interface**:
```python
@dataclass
class Episode:
    spec: EpisodeSpec
    frames: list[Frame]
    instruction: str
    phase_instructions: tuple[str, str]
    execution: ExecutionResult
    engine_version: str

class EpisodeRunner:
    def __init__(self, cfg: EngineConfig) -> None
    def run(self, spec: EpisodeSpec) -> Episode | FailedEpisode
    def close(self) -> None
```

`FailedEpisode` carries the spec, the failure reason, and whatever partial diagnostics
exist — failures are first-class data, not exceptions, because Step 14 reports on them.

**Depends on**: Steps 3–9.

### Acceptance Criteria
- Integration test (`slow`): 5 specs from 5 different seeds all run end-to-end; ≥ 4 return
  `Episode` (not `FailedEpisode`) with `placed=True`.
- Reproducibility: running the same spec twice gives identical states/actions (1e-9) and
  images identical in ≥ 99.9% of pixels with max per-pixel difference ≤ 2 counts.
- Cross-episode isolation: running spec A, then spec B, then spec A again reproduces the
  first A result to the same tolerance — i.e. no state leaks between episodes.
- Failures do not raise: a spec engineered to be infeasible returns a `FailedEpisode` with a
  populated reason, and the runner remains usable for the next spec afterwards.
- Throughput measured and recorded in `docs/render-budget.md`: seconds per episode, and the
  fraction spent in rendering vs. physics vs. IK. **Recorded, with a target of ≤ 120 s per
  episode at the chosen working point**; if exceeded, Step 15's calibration must reduce
  capture fps, episode duration, or episode count to meet it before the full run — not
  resolution, which §0.10 fixes independently of throughput.
- Steady-state RSS over 20 consecutive episodes stays ≤ 1.6 GB with no monotonic growth.

---

## 11. Per-episode validator

**What gets built**: `validate/episode.py` — the first half of Deliverable 2. A pure
function over an `Episode` returning a structured verdict, run **immediately after
generation, before the episode is written to the dataset**. Failing episodes are quarantined
(spec + verdict + a debug artifact) rather than dropped silently, so the failure modes are
themselves data.

Checks, each an independently testable predicate with a named code:

*Task success* — `TASK_PLACED` (final piece xy within tolerance of target square centre,
tilt < 15°, resting/at rest), `TASK_GRASPED`, `NO_DISTRACTOR_DISTURBED` (no other piece moved
> 1 cm), `PIECE_ON_BOARD` (final pose inside board bounds).

*Physical plausibility* — `NO_NAN`, `JOINT_LIMITS`, `NO_PENETRATION` (max contact penetration
depth below threshold), `NO_EXCESSIVE_FORCE`, `VELOCITY_BOUNDED`, `NO_TELEPORT` (per-frame
end-effector displacement below a kinematic bound).

*Data integrity* — `LENGTH_CONSISTENT` (frames vs. states vs. actions), `TIMESTAMPS_MONOTONIC`,
`ACTION_STATE_ALIGNED` (achieved state tracks the previous action within tolerance),
`EPISODE_LENGTH_IN_RANGE`, `ALL_PHASES_PRESENT`, `SPEC_ROUNDTRIP` (spec serializes and
re-parses).

*Visual integrity* — `IMAGES_NOT_BLANK`, `IMAGES_NOT_STATIC` (episode has real motion:
mean inter-frame difference above floor), `PIECE_ON_BOARD_AT_START` (source piece pose is
on-board and within tolerance of the intended square at frame 0 — checked from privileged
`MjData` state, not pixels; per §0.9, the wrist camera's frame-0 framing is pose-dependent
so pixel segmentation can't be the check here), `GRIPPER_VISIBLE` in the wrist view,
`EXPOSURE_IN_RANGE`.

*Language integrity* — `INSTRUCTION_MATCHES_SPEC` (squares/piece in text agree with the spec).

**Interface**:
```python
@dataclass(frozen=True)
class CheckResult: code: str; passed: bool; value: float | None; threshold: float | None; message: str
@dataclass(frozen=True)
class EpisodeVerdict:
    valid: bool                    # all blocking checks passed
    checks: tuple[CheckResult, ...]
    severity: Literal["ok", "warn", "reject"]

def validate_episode(ep: Episode, cfg: ValidationConfig) -> EpisodeVerdict
```

Every check records its **measured value and threshold**, not just a boolean — that is what
makes Step 14's aggregate distributions possible.

**Depends on**: Step 10.

### Acceptance Criteria
- Every check has at least one unit test with a **synthetic passing episode** and at least
  one with a **synthetic failing episode** built by mutating the passing fixture (inject a
  NaN, shift the final piece pose 5 cm, blank the images, freeze the arm, swap the
  instruction's target square, truncate the action array). Test fails if any check code lacks
  both cases — enforced by a test that enumerates check codes and asserts coverage.
- No false positives on the golden set: 10 hand-inspected known-good episodes (produced in
  Step 15's pilot and committed as fixtures with downsampled images) all return `valid=True`.
- No false negatives on the failure set: 10 known-bad episodes (deliberately induced: missed
  grasp, dropped piece, wrong square, blank render, aborted execution) all return
  `valid=False` with the expected primary check code.
- Validation is pure and fast: no simulator or file I/O; runs in ≤ 200 ms per episode.
- Thresholds live in `ValidationConfig` with documented defaults; changing a threshold via
  config demonstrably flips a borderline fixture's verdict.
- Quarantine round-trip: a rejected episode's saved artifact can be reloaded and re-validated
  to the identical verdict.

---

## 12. LeRobot dataset writer

**What gets built**: `writer.py` — appends validated episodes to a `LeRobotDataset` on disk
(features schema, per-frame append, per-episode commit, video encoding via ffmpeg at CPU,
`meta/` files), plus the sidecar `meta/generation.jsonl` carrying each episode's full spec,
verdict, and engine git SHA. Supports resuming into an existing dataset directory.

**Interface**:
```python
class DatasetWriter(Protocol):
    def open(self, root: Path, features: FeatureSpec, fps: float, resume: bool) -> None
    def add_episode(self, ep: Episode, verdict: EpisodeVerdict) -> int   # returns episode_index
    def finalize(self) -> DatasetStats
class LeRobotWriter(DatasetWriter): ...
```

Video encoding is a known CPU cost; the writer measures and reports encode time per episode
so Step 15 can decide encoding preset vs. throughput.

**Depends on**: Steps 10, 11.

### Acceptance Criteria
- **Round-trip is the primary criterion**: after writing 3 episodes, loading the directory
  with `LeRobotDataset(root)` succeeds; `len(dataset)` equals the total frame count;
  `dataset.num_episodes == 3`; and for 20 randomly sampled frames, `observation.state`,
  `action`, `timestamp`, and `task` match the source `Episode` values to within 1e-5.
- Decoded video frames match the source images: mean absolute pixel error ≤ 4 counts and
  PSNR ≥ 32 dB per sampled frame (lossy encoding is acceptable; silent corruption is not).
- Schema: `meta/info.json` declares exactly the features in §0's table with correct shapes,
  dtypes, and the configured fps; a schema-assertion test compares against a committed
  expected-schema fixture.
- Resume: writing 2 episodes, closing, reopening with `resume=True`, and writing 2 more
  yields a dataset with 4 episodes, contiguous `episode_index` 0–3 and globally contiguous
  `index`.
- Crash safety: killing the process mid-episode leaves a dataset that either excludes that
  episode entirely or is repairable by a documented `--repair` path; verified by an
  integration test that truncates the in-progress episode files and then reopens.
- `meta/generation.jsonl` has exactly one record per written episode, each containing a spec
  that `EpisodeSpec.from_json` parses and that the engine can re-run.
- Rejected episodes are **never** present in the main dataset (asserted by writing a mixed
  batch of 3 valid + 2 invalid and checking `num_episodes == 3` and the quarantine dir
  holding 2).

---

## 13. Batch generation CLI

**What gets built**: `cli/generate.py` — the unattended driver. Single process, no
parallelism (per the strategy doc's RAM constraint). Loops: sample spec → run → validate →
write or quarantine → log → checkpoint.

Features: `--episodes N`, `--master-seed`, `--out`, `--config`, `--resume`, `--max-hours`,
`--dry-run`; a checkpoint file recording completed spec indices so an interrupted run
resumes without duplicating or skipping; structured JSONL progress logging (one record per
episode with timing, verdict, and check values); a rolling console summary (episodes done,
success rate, eps/hour, ETA, RSS); an **RSS guard** that aborts cleanly if memory exceeds a
configured ceiling; and a **quality tripwire** that halts the run if the rolling success rate
over the last 25 episodes falls below a configured floor (default 70%) — the strategy doc's
"catch bad data close to its source" requirement, enforced at run time rather than
post hoc.

**Depends on**: Steps 10–12.

### Acceptance Criteria
- `--dry-run --episodes 100` samples and validates 100 specs (plans included) without
  stepping physics or rendering, in < 60 s, reporting the projected square coverage.
- `--episodes 3` on a clean directory produces a loadable dataset (Step 12's criteria) and a
  JSONL log with exactly 3 records containing per-episode wall time and verdict.
- Resume correctness: `--episodes 6` interrupted with SIGINT after ~3 episodes, then re-run
  with `--resume`, yields exactly 6 episodes with no duplicate spec seeds (asserted against
  `meta/generation.jsonl`).
- SIGINT handling is graceful: the current episode is abandoned, the dataset is left
  consistent, and the exit message names the checkpoint file. Exit code is distinct from a
  crash.
- RSS guard: run with an artificially low ceiling → the process exits cleanly with a named
  reason and a consistent dataset, not an OOM kill.
- Quality tripwire: with the validator forced to reject everything, the run halts within 25
  episodes with a clear message and non-zero exit code.
- `--max-hours` is honoured to within one episode's duration.
- Only one MuJoCo process exists at any time (asserted by the test harness inspecting child
  processes); CPU use stays within 2 vCPU without oversubscription (thread env vars pinned:
  `OMP_NUM_THREADS`, `MKL_NUM_THREADS`).
- Re-running with the same master seed and episode count into a fresh directory reproduces
  identical `meta/generation.jsonl` specs.

---

## 14. Aggregate/batch validation and reporting

**What gets built**: the second half of Deliverable 2 — `validate/batch.py`,
`validate/diversity.py`, `validate/report.py`, and `cli/validate_dataset.py`. Operates on an
assembled dataset directory (a batch or the full dataset), independently of the generator,
so it can be run at any time and on any dataset.

**Aggregate checks**:
- *Integrity*: dataset loads; frame counts match metadata; every episode has its wrist
  camera stream decodable end-to-end (decode a sampled subset fully, all episodes at least
  partially); no duplicate `index`; fps consistent; no episode below/above the length bounds;
  every dataset episode has a matching `generation.jsonl` record and vice versa.
- *Statistical*: per-dimension distributions of `state` and `action` (min/max/mean/std,
  percentiles) with out-of-range and constant-dimension detection (a dimension with near-zero
  variance means a joint never moved — a real and easy-to-miss failure); action smoothness
  (per-step delta percentiles, outlier count); NaN/inf sweep.
- *Coverage*: source/target square histograms over the board, piece-type histogram,
  square-pair coverage, instruction-template histogram, board-pose and lighting coverage —
  each with a min-count and a max/min-ratio criterion.
- *Diversity* (the strategy doc's central risk): approach-azimuth entropy and sector
  occupancy; distributions of lift/transport height, segment durations, grasp-closure offset;
  path-shape mix; **near-duplicate detection** via pairwise distance between time-normalized
  end-effector paths and via action-sequence hashing — reporting the count of pairs below a
  similarity threshold; per-episode length variance.
- *Visual*: per-episode and dataset-level image statistics (mean/std luminance, colour
  histograms) with outlier flagging; a blank/frozen-frame sweep across all frames (cheap
  proxy statistics precomputed at generation time in Step 11, re-verified on a sample here).
- *Language*: instruction↔spec agreement re-checked across the whole dataset; vocabulary and
  template distribution; no instruction referencing a square absent from the episode.
- *Cross-check against per-episode verdicts*: recompute a sample of Step 11's checks from the
  written dataset and assert agreement with the stored verdict — this catches the writer
  silently corrupting data after validation passed.

**Outputs**:
- `report.md` + `report.html` with tables and plots (square-coverage heatmap over the 8×8
  board, approach-azimuth polar histogram, per-joint action distributions, episode-length
  histogram, near-duplicate scatter, failure-reason breakdown, throughput over time);
- a **contact sheet**: sampled frames (first/mid/last of N random episodes, plus every
  flagged episode) tiled into a PNG for fast human review — the human-in-the-loop half of
  the workflow;
- `report.json`, machine-readable, with every metric and a top-level `pass`/`fail`;
- exit code 0 only if all blocking aggregate criteria pass.

**Interface**:
```python
def validate_batch(root: Path, cfg: BatchValidationConfig) -> BatchReport
def write_report(report: BatchReport, out_dir: Path) -> None
# CLI: pixi run validate-dataset --root data/chess_v1 --out reports/ [--sample-frames 200]
```

**Depends on**: Step 12 (needs a written dataset); can be developed in parallel with 13
against a small fixture dataset.

### Acceptance Criteria
- Runs on a 500-episode dataset in ≤ 15 minutes on this host, with peak RSS ≤ 1.2 GB
  (streams episodes; never loads the whole dataset into memory) — asserted by test on a
  synthetic dataset of equivalent size.
- **Fault-injection suite**: for each of these corruptions applied to a good fixture dataset,
  validation exits non-zero and the report names the correct failing metric —
  (a) one episode's video replaced with black frames; (b) one episode's actions replaced with
  a constant; (c) 40 episodes duplicated verbatim (near-duplicate detector must flag them);
  (d) all episodes forced to a single approach azimuth (diversity criterion must fail);
  (e) 20 squares removed from the coverage; (f) one instruction's target square edited;
  (g) `info.json` fps changed to disagree with timestamps; (h) an episode present in the
  parquet but missing from `generation.jsonl`.
- A clean fixture dataset passes with exit code 0 and no flags.
- Every metric in `report.json` has an explicit threshold recorded next to it — no metric is
  reported without a pass/fail rule, and no rule is applied that isn't in the report.
- The board heatmap, polar histogram, and contact sheet are produced as files and are
  non-empty; the contact sheet includes 100% of flagged episodes.
- The report is reproducible: two runs on the same dataset produce identical `report.json`
  (given a fixed sampling seed).
- Determinism of sampling: `--sample-frames` uses a seeded RNG recorded in the report.

---

## 15. Pilot batch, calibration, and full generation run

**What gets built**: no new modules — the empirical loop that turns the engine into a
dataset. Three sub-stages, each gated:

**15a — Pilot (25 episodes).** Run at the Step 2 working point. Validate per-episode and in
aggregate. Human-review the contact sheet. Record throughput, success rate, and RSS. Curate
the golden fixture sets promised in Step 11 (10 good, 10 bad episodes) from this batch.

**15b — Calibration.** Adjust the knobs against measured evidence: capture fps, episode
duration bounds, episode count, randomization ranges (widen anything the Step 14 report shows
as under-dispersed; narrow anything driving failures), IK/controller gains, validator
thresholds (only with justification recorded), video encoding preset. Image resolution is
**not** in this set: §0.10 fixes it at 512², and it is reopened only if the RSS ceiling is
breached, as an explicitly documented recalibration. Re-run a 25-episode pilot after changes.
Iterate until the gate below passes. Every knob change is committed with the report that
motivated it.

**15c — Full run.** Generate the target batch as a single-process, resumable run under
`nohup`/systemd-style supervision, with periodic incremental aggregate validation (Step 14
run on the partial dataset every ~100 episodes) rather than only at the end. Target size:
**500 episodes minimum, 1000 if throughput allows**, sized from the measured seconds-per-
episode against an acceptable wall-clock budget. Finish with a full Step 14 report, a
`docs/dataset-card.md` (contents, schema, randomization ranges, known limitations, engine
git SHA, validation summary), and `docs/runbook.md` (how to regenerate, resume, validate,
and interpret the report).

**Depends on**: Steps 1–14.

### Acceptance Criteria
- **Pilot gate (must pass before 15c)**: per-episode success rate ≥ 85% over 25 episodes;
  Step 14 aggregate validation exits 0 on the pilot; every diversity criterion from Step 6
  and Step 14 passes on real generated data (not just sampled plans); measured throughput
  ≤ 120 s/episode; peak RSS ≤ 1.6 GB; human review of the contact sheet finds no systematic
  visual artifact (recorded as a signed-off note in `docs/pilot-review.md`).
- Golden fixtures committed and Step 11's golden-set tests (no false positives/negatives) pass
  against them.
- **Full-run completion criteria**: ≥ 500 valid episodes written; overall rejection rate ≤ 15%
  with every rejection reason categorized in the report; incremental validations all exit 0;
  final Step 14 report exits 0.
- Coverage on the final dataset: every reachable square appears as source ≥ 5 times and as
  target ≥ 5 times; `max/min` count ratio ≤ 3.0; every enabled piece type ≥ 5% of episodes.
- Diversity on the final dataset: all Step 14 diversity criteria pass; **zero near-duplicate
  episode pairs** below the similarity threshold.
- Reproducibility spot-check: 5 episodes re-generated from their stored specs match the
  dataset's stored states/actions to within 1e-6 and images to Step 10's tolerance.
- `docs/dataset-card.md` and `docs/runbook.md` exist and are accurate; a reviewer following
  the runbook can regenerate a 3-episode dataset and validate it without asking questions.
- The full dataset directory loads with `LeRobotDataset` in a fresh process, and iterating
  100 random samples yields correctly shaped tensors with no decode errors.

---

## Validation Strategy

This section is the plan for Deliverable 2. It is not a bolt-on to the generator; it is a
gate the generator's output has to pass through at three distinct points. The organizing
principle from the strategy doc is *catch bad data close to its source*, so validation is
staged from cheapest/earliest to most expensive/latest, and each stage can independently
fail a run.

### Stage A — Pre-generation (before any physics runs)

Cheapest possible rejection: specs and plans are validated before an episode is simulated.

- `EpisodeSpec` field ranges and internal consistency (source ≠ target, distractors distinct,
  squares reachable) — Step 4.
- Plan structural, safety, and IK-feasibility checks — Step 6. An infeasible plan is
  resampled or the spec skipped, and the skip is logged with a reason.
- `--dry-run` mode runs this stage over an entire planned batch in under a minute,
  producing projected coverage and diversity statistics **before** committing hours of CPU.
  This is the intended first action before any long run.

Failure handling: skipped specs are logged with reasons and counted; a skip rate above a
configured ceiling aborts the run (the sampler or the scene geometry is wrong).

### Stage B — Per-episode, immediately after generation (Step 11)

Every episode is validated the moment it is generated and **before it is written to the
dataset**. Checks span five families — task success, physical plausibility, data integrity,
visual integrity, language integrity — each producing a measured value, a threshold, and a
pass/fail code (not just a boolean), because those values become Stage C's distributions.

- **Valid** episodes are written to the dataset with their verdict recorded in
  `meta/generation.jsonl`.
- **Invalid** episodes are written to a quarantine directory with their spec, verdict, and a
  debug artifact (downsampled frames + state/action arrays), never to the main dataset. They
  are retained deliberately: the distribution of failure reasons is diagnostic information
  about the engine, and quarantined episodes are replayable from their spec.
- **Warn**-severity episodes are written but tagged, so Stage C can report on them.

Two run-time tripwires make this stage act on the generator rather than merely observe it
(Step 13):
1. **Rolling success rate** over the last 25 episodes below the configured floor halts the
   run. A long unattended run must not spend six hours producing garbage.
2. **RSS ceiling** breach ends the run cleanly rather than being OOM-killed mid-write.

The validator is a pure function with no I/O and no simulator, which is what makes it
testable in isolation, cheap enough to run on every episode, and reusable in Stage C.

### Stage C — Aggregate, once a batch or the full dataset is assembled (Step 14)

Run via `cli/validate_dataset.py` on any dataset directory, at any time — after every ~100
episodes during a long run, after each batch, and once on the completed dataset. It is
independent of the generator, so it also validates datasets produced by an older engine
version.

It answers questions that are invisible episode-by-episode:

1. **Integrity at rest** — does the written dataset load, decode, and self-agree? This is
   where writer bugs surface: an episode can be perfectly valid in memory and corrupt on
   disk. A sampled re-computation of Stage B's checks *from the written dataset* is compared
   against the stored verdicts; disagreement is a blocking failure.
2. **Distributional health** — per-dimension state/action statistics, constant-dimension
   detection, smoothness outliers, NaN sweeps. A joint that never moved, or an action
   dimension with a stuck value, fails here and nowhere else.
3. **Coverage** — square, piece-type, square-pair, board-pose, lighting, and instruction
   template histograms against min-count and max/min-ratio thresholds. Ensures the sampler's
   coverage-driven design actually delivered.
4. **Diversity** — the strategy doc's explicit failure mode ("a large but uniform dataset
   generalizes worse than a small diverse one"). Measured, not assumed: approach-azimuth
   sector occupancy and entropy, dispersion of heights/timings/grasp offsets, path-shape mix,
   and **near-duplicate detection** over time-normalized end-effector paths and hashed action
   sequences. Insufficient diversity fails the dataset even if every individual episode is
   perfect — this is the single most important aggregate criterion in Phase 1.
5. **Visual health** — dataset-level luminance/colour statistics with outlier flagging, and a
   blank/frozen-frame sweep.
6. **Language health** — instruction↔spec agreement across the whole dataset, vocabulary and
   template distribution.

Output is three artifacts: a machine-readable `report.json` (every metric with its threshold
and verdict, exit code 0 only if all blocking criteria pass), a human-readable
`report.md`/`report.html` with plots (board coverage heatmap, approach-azimuth polar
histogram, per-joint action distributions, episode-length histogram, failure-reason
breakdown), and a **contact sheet PNG** tiling sampled frames plus every flagged episode.

### Stage D — Human review (bounded, but not optional)

Automated checks cannot catch "the motion looks unnatural" or "the gripper always approaches
from the same visually odd angle." The contact sheet and a handful of rendered episode videos
are reviewed by a human at the pilot gate (15a) and once on the final dataset, with the
outcome recorded in `docs/pilot-review.md`. This is a fixed, small, scheduled cost — not
open-ended manual spot-checking, which is exactly what the strategy doc says to avoid. It is
also where the later human-teleop comparison (out of scope for Phase 1) would eventually
plug in.

### Meta-validation: testing the validators

The validators are code, so they get the same TDD treatment as the engine (per `AGENTS.md`):

- Every per-episode check has a passing fixture *and* a mutation-derived failing fixture; a
  meta-test enumerates check codes and fails if any lacks both.
- The aggregate validator is tested by **fault injection**: eight specific corruptions of a
  known-good fixture dataset (blank video, constant actions, duplicated episodes, collapsed
  azimuth diversity, missing square coverage, edited instruction, inconsistent fps, missing
  metadata record), each of which must be caught and correctly named.
- Golden sets from the pilot batch guard against both false positives and false negatives.
- Thresholds live in typed config objects with documented defaults; a threshold change must
  demonstrably flip a borderline fixture, so no threshold is unfalsifiable.

### Where each stage runs

All stages run on this host, in a single process, under the osmesa/2-vCPU/1.6 GB-RSS
constraints. Stage A is seconds. Stage B is milliseconds per episode. Stage C streams the
dataset from disk and is budgeted at ≤ 15 minutes per 500 episodes, so running it every ~100
episodes during a long generation run costs a few percent of total wall-clock — cheap
insurance against discovering a corrupt dataset after a multi-day run.

---

## Risk register

| Risk | Signal | Mitigation |
|---|---|---|
| osmesa rendering too slow for a useful dataset at the fixed 512² resolution | Step 2 benchmark well below 4 fps at 512² | No longer resolved by shrinking the image (§0.10 fixes resolution regardless of fps) — absorbed by Step 15's episode-count/capture-fps calibration instead. Only resolution's own hard ceiling is RSS > 1.6 GB, which is a real blocker, not a throughput one. Residual: Step 10's full-pipeline (physics+IK+render) throughput on the real scene is still unmeasured and could still make episode counts impractically low. |
| Vendored SO-101 MJCF fails Step 3 acceptance (base collisions, gripper joint fix, compile/reachability) | Step 3 blocked | Hard stop. Fix the asset (patch collisions/joint mapping, or escalate upstream) before proceeding — no silent substitution. |
| `mink` IK fails to converge near the board surface | Step 5 convergence < 98% | Treated as a scene/reachability problem (target range, joint limits), not a solver swap — tune task weights and Step 3/4 reachability bounds first. |
| Grasping is unreliable (pieces slip) | Step 7/15a success rate < 85% | Tune gripper friction/geometry, grasp height, and closure force; widen piece geometry tolerance; these are scene parameters, not code changes. |
| Diversity criteria fail despite randomization | Step 6/14 diversity checks red | Ranges are config; widen and re-run the cheap Stage A dry-run to confirm before regenerating. |
| RAM ceiling hit during long runs | RSS guard trips | Frames stream to the writer; model reuse audited in Step 10; guard exits cleanly and the run resumes. |
| Long run dies mid-way | Process gone | Checkpoint + `--resume` (Step 13), verified by test, plus incremental Stage C validation so a corrupt run is caught early. |

## Out of scope (restated)

Not planned here, per `strategy-step-1.md`: the SFT training script or hyperparameters, RL
fine-tuning, real SO-101 hardware/calibration/deployment, and human-teleop tooling setup
(gym-hil etc.). The teleop sanity-check dataset is a later, non-blocking activity; this plan
only ensures the dataset format and validation tooling would accept such episodes unchanged.
