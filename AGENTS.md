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

## Scene conventions

These are the project's standing assumptions for training and simulation, encoded as the
defaults in `src/vla_chess/scene.py`. Treat them as the baseline; a scene that departs from
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
- **Size anything the hand does against the hull, not the mesh.** MuJoCo collides meshes as
  convex hulls and the fixed finger is a fork whose hull fills it, so its contact face stands
  11.6 mm inboard of the face the mesh draws. Jaw commands, the pinch offset and the approach
  opening all come from `scene.FIXED_FACE_INSET`, `GAP_AT_ZERO` and `GAP_PER_RADIAN`, which are
  measured by closing on a real piece and bisecting for first contact — a grip sized off mesh
  geometry commanded 6 mm of interference and looked correct in every render. Screening is the
  other way round: `collision.piece_clearance` uses the true meshes, because the question there
  is whether the real hand would hit something. See `docs/trajectory.md`.
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
  made against a grasp frame since corrected, and does not hold at the current one; the board
  has been left where the eye put it, pending that re-inspection (`docs/trajectory.md`).

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
