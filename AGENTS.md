# AGENTS.md

## General code guidelines

### Comments

Use inline comments where needed, don't add any comments at the top of any files.
This creates drift. Describe only what the unit itself does — context from the calling
places does not belong in a function's comments or its docstring.

### Docstrings

Use **Google style**: an imperative one-line summary, then `Args:` / `Returns:` / `Raises:`
sections where the function takes arguments, returns something, or raises on purpose. Types
live in the signature — mypy enforces them there — so don't repeat them in the docstring.
Describe what a parameter *means* and what the return value *is*, including units and frames
wherever the name alone doesn't say.

Every function and method gets one, with two exceptions:

- **Unit tests.** The test name and its assertions are the documentation. Add a docstring only
  where the setup encodes something a reader could not otherwise infer.
- **Local helpers with a leading underscore** that do no complex multi-step work and whose name
  already says what they do. Those get a single-line docstring and no sections — and the name
  has to earn it. A private helper that *does* do multi-step work is documented in full like
  anything else.

```python
def solve_square(square: str, prefer: Configuration | None = None) -> Pose:
    """Solve a vertical grasp pose for a board square.

    Args:
        square: Algebraic square name, e.g. "e4".
        prefer: Configuration the arm is coming from; the nearest acceptable pose to it
            is returned. Pass this whenever the arm is already somewhere.

    Returns:
        Grasp pose in the world frame.

    Raises:
        ValueError: If no acceptable pose reaches the square.
    """
```

This is a convention, not a lint rule: ruff's `D`/`DOC` rules are deliberately left off for
now, so nothing will catch a missing docstring for you. It may become enforced later.

### Tests

Use test driven development. Design components with testable interfaces that do
one thing well. Use unit tests inside components. Add integration tests for
actions between these interfaces.

### For Python

- Manage Python environments via pixi.
- Use mypy, and set type hints everywhere.
- Use ruff.
- Set up a pixi pre-commit task running unit tests, type checks, and lints.

## Layout

```
src/vla_chess/
  paths.py     asset root; every asset path resolves from here, never from a module's __file__
  sim/         the world: scene assembly, board/square geometry, contacts, starting positions
  grasp/       the IK — vertical (fingers-down) grasp solving and the solved reference set
  control/     planner (move -> waypoints) and executor (waypoints -> simulated motion)
  sensor/      what the policy sees: offscreen rendering
  dataset/     episode recording and training-trajectory production
  viz/         output for humans: contact sheets, captions, video
  analysis/    instrumentation: benchmarks, probes and the reports under docs/
  cli/         argparse shells; no logic worth testing lives here
```

Imports point **down** this list and never back up: `sim` imports nothing else in the package,
`grasp` and `sensor` build on `sim`, `control` on `sim` + `grasp`, `dataset` on everything above
it. `viz`, `analysis` and `cli` sit outside the stack and may import anything. If a change needs
an edge that points up, the module is in the wrong package — move it rather than adding the
import. This is a convention, not a lint rule; `import-linter` can enforce it later if it drifts.

A CLI is argparse plus wiring. The function it calls must be importable and testable without
argparse — if you find yourself testing a CLI, the logic is in the wrong tier.

Everything here was one flat module per name until the 2026-08-20 reorganisation, so older
design docs say `scene.grip_radius` where the code now says `sim.scene.grip_radius`. Leaf module
names are unchanged and those references still resolve by name, with one exception: the mink
solver formerly at `kinematics.IKSolver` is now `analysis.reference_ik.IKSolver`. It is a
cross-check for `ik_report`, not a runtime dependency — nothing outside `analysis/` imports it.

## Scene conventions

These are the project's standing assumptions for training and simulation, encoded as the
defaults in `src/vla_chess/sim/scene.py`. Treat them as the baseline; a scene that departs from
them is an experiment, not the norm, and should say so at the call site.

- **The arm plays black, from behind the black back rank.** The arm's `base` is pinned at the
  world origin by the vendored asset, so this is expressed as `DEFAULT_BOARD_YAW = pi/2`
  rather than by moving the arm.
- **The wrist camera is the observation.** `DEFAULT_WRIST_CAMERA_MOUNT` is the over-the-jaws
  resting mount; the other mounts in that module are comparison baselines, not live options.
- **Scenes start at rest.** Build with `build_scene`, then get state from `make_data` rather
  than `mujoco.MjData(model)` directly — a bare `MjData` sits at the arm's XML home pose,
  which the policy should never observe from.

Note that the arm is blind to ranks 7-8 and the a6/h6 corners at rest (46/64 squares in
frustum) — its own home ranks. That is a property of the seat, not a bug, and the policy is
expected to move to see them.

- **Grasps come in from straight above, and the finger axis is the site's local x.** The TCP
  site's z axis is the gripper's roll, not its approach: constraining z and calling it
  "top-down" selects side approaches. Use `grasp.VerticalGraspSolver` (or
  `grasp.finger_axis_site_frame`) rather than writing a fresh orientation constraint. Reaching
  the target is the easy half — screen candidate poses for collisions and restart, or the
  solver will hand back a pose that reaches the square through the arm's own body.
- **A square has a continuum of grasps, so say which one you want.** The residual leaves
  `wrist_roll` free: every solution reaches the square equally well and they sit a half-turn
  apart, so taking the first acceptable one is a coin toss that a transit then pays for in
  wrist rotation — 250 degrees of it on the episode that surfaced this. Pass `prefer` (the
  configuration the arm is coming from) to `solve`/`solve_square` anywhere a pose is solved for
  an arm that is already somewhere, and it returns the nearest acceptable pose that is not
  sitting on the roll stop. Judge a pose's headroom with `free_margin`, on roll alone: the other
  four joints are pinned by the target, and `wrist_flex` is within 1.5 degrees of its stop in
  every fingers-down pose over the home ranks. See `docs/trajectory.md`.
- **The hand is collided as slabs of its own meshes, and the two have to stay in step.** MuJoCo
  collides a mesh as one convex hull, and the fixed finger is an L whose hull fills the mouth a
  gripped piece occupies: the drawn finger stood 10.6 mm clear of a pawn the engine was holding,
  so the observation showed a hand gripping nothing and the recorded jaw angle was one that grips
  only in simulation. `scene._split_hand_collision` cuts each jaw's collision mesh across its own
  long axis and hulls the slabs (`FINGER_SLAB_CUTS`, `JAW_SLAB_CUTS`); visual geoms are never
  touched. Jaw commands, the pinch offset and the approach opening come from
  `scene.FIXED_FACE_INSET` and `JAW_GAP_TABLE`, measured by `pixi run gripper-probe`, which
  slides a piece across the jaws and bisects each face for first contact. Re-run it and re-pin
  both if anything about the hand's collision geometry changes — it is a fixed point (the pinch
  offset moves the pose it measures at), so iterate until it stops moving. Anything asking what
  the real hand would hit — `collision.piece_clearance`, occlusion in `survey.framing` — reads
  the visual geoms instead. See `docs/trajectory.md`.
- **The jaws close on the widest thing between the fingers.** Not on whatever the TCP sits
  beside: the fingers span 8 mm below the TCP to 26 mm above it, and a pawn's head is inside
  that band and wider than its stem. Use `scene.grip_radius`.
- **There is exactly one board pose: `scene.DEFAULT_BOARD_POSE` (x = 0.300).** Everything
  solved against the board — the reference grasps above all — is solved against it, and a
  second board distance has already been in this tree once: reference poses from one distance
  miss by ~20 mm at another, which looks right in a render and grips air. The value sits in a
  ~4 mm window with a hard constraint on each side (d8/e8 self-collision below it, birds-eye
  framing above it), so do not re-tune it against a reach or coverage count. It was signed off
  by eye, and `pixi run render-reference-views` regenerates exactly what was looked at — if the
  pose ever moves, re-inspect those and get the new ones signed off too. See
  `docs/grasp-poses.md` and `docs/reference-views.md`. The d8/e8 argument for this distance was
  made against a grasp frame since corrected twice, and it survives both: at the current frame
  those two squares still solve, and the reference set's 28 of 64 loses `b5`/`g5` instead.

### Running `pixi run test-slow`

**Do not run this.** It is not part of CI, not part of `pixi run precommit`, and not part of
any routine verify-and-commit loop. A commit is green without it. "I changed scene/render code,
so I should probably check the slow suite" is *not* a sufficient reason — that reflex is what
this rule exists to stop.

Run it only when one of these is true:

- The user explicitly asks for it by name.
- A specific slow test is the direct deliverable of the current task (you are writing or fixing
  that test, and no fast test can express the assertion).

In every other case, rely on `pixi run precommit`. If you believe a change needs slow-suite
coverage, say so and let the user decide — don't just run it.

When it is genuinely warranted, invoke it on its own: never chained with `&&`/`;`, never
batched alongside other pixi tasks, never in parallel with another command in the same turn.
On this host it is slow enough (real MuJoCo + osmesa, 2 vCPUs, no GPU — see
`docs/render-budget.md`) to regularly exceed a single tool-call timeout, and re-invoking it or
wrapping it in an outer timeout while a prior run is still in flight is what actually causes
runs to get killed or orphaned, not the suite itself hanging. Run it, wait for that one
command to finish, then move on.
