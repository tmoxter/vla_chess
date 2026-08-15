# VLA Chess — Phase 1 Strategy: Trajectory Data Generation

Target picture for the first phase of the project: producing a training dataset of
pick-and-place trajectories, in simulation, for supervised fine-tuning (SFT) of
SmolVLA. No implementation yet — this captures the *what* and *why*, not the *how*.

## Goal of this phase

Generate a simulated dataset of chess piece pick-and-place episodes, labeled with
language instructions (e.g. "pick up the piece on b2" / "place the piece on b4"),
suitable for fine-tuning a pretrained SmolVLA checkpoint via LeRobot. This dataset
is the input to the SFT training run (which happens on a rented GPU, out of scope
for this phase).

## Deliverable of this phase

Two things, not one:

1. A data-generating engine for trajectories — the scripted/IK cold-start pipeline
   described below, capable of producing pick-and-place episodes across squares,
   board configurations, and piece types on demand.
2. Tooling to monitor and evaluate the quality and validity of the data that engine
   produces — not a one-off manual spot-check, but a repeatable workflow that lets
   quality be checked at multiple points along the generation process (e.g. per-
   episode validity right after generation, and again in aggregate once a batch or
   the full dataset is assembled). The point is to catch bad data close to its
   source rather than discovering it only once it shows up as a failed SFT run.

Both are in scope for this phase; neither is optional. The generation engine
without a way to evaluate its output is not a trustworthy deliverable on its own.

## Cold start: scripted/IK-generated trajectories, no teleop required

The first batch of training data does not require a human operator, a leader arm,
or any physical hardware. It's generated as follows:

- Use privileged, ground-truth simulator state (exact piece pose, exact target
  square pose) to script an "oracle" pick-and-place motion via inverse kinematics:
  approach above the source square, descend, close the gripper, lift, transport
  above the target square, descend, release, retreat.
- Sweep this across many squares, board positions/orientations, and piece types,
  fully unattended, to produce hundreds to thousands of episodes.
- Record camera images, proprioceptive state, the resulting action stream, and the
  text instruction for each episode.
- This does **not** make the trained policy dependent on IK at runtime. IK is only
  ever used offline, inside the simulator, to author demonstrations. The SmolVLA
  checkpoint only ever learns the mapping from (image, state, instruction) to
  action — it never calls IK itself, and the resulting policy stays board-agnostic
  at inference time.
- This is a standard, proven technique (the same pattern behind RoboMimic/MimicGen-
  style sim datasets): script an oracle policy from privileged state, then use its
  rollouts as demonstrations for a vision+language policy that has no access to
  that privileged state.

### Known defect: the oracle does not currently grasp (found 2026-08-15)

The cold start above assumes the scripted oracle can pick a piece up. As of the first
complete `d7 -> d6` episode it cannot, and the cause is a frame convention rather than
anything in the physics: `gripperframe`, the site every pose in this project is solved
for, sits at the point where the two fingers *meet when shut*, flat against the fixed
finger's face — not on the centreline between jaws opened wide enough to admit a piece.
Solving IK to put that point on a piece's axis parks the fixed finger inside the piece.
The hand model itself is sound. `docs/trajectory.md` carries the measurements.

Restoring the oracle, in dependency order. The order matters more than the items:

1. **Repair the validity check before the thing it checks.** `contacts_at_pose` excludes
   every hand-vs-target contact on the grounds that touching the target is the point of a
   grasp, which is what let a finger sit 9 mm inside a pawn through a full sign-off. Bound
   the exclusion by depth, and add the predicate that is missing altogether: after closing,
   both the fixed finger *and* the moving jaw must be in contact. Doing this first means
   the re-solve in (3) is screened by something that can tell a grasp from an impalement;
   doing it later means re-solving twice. This is the phase's second deliverable — quality
   tooling — catching a defect at its source, and it is the case study for why that
   deliverable is not optional.
2. **Pin the pinch frame.** A second site in the `gripper` body, offset along the closing
   axis by at least the stem radius. It cannot be the jaw midline, which moves with the
   opening; the fixed finger's face is what is actually invariant. Preferred convention:
   one offset sized to the largest stem (12.5 mm) plus margin, verified against every piece
   type, letting closure do the centring — one frame and one asset rather than a per-piece
   family.
3. **Re-solve what was derived against the old frame.** `assets/reference_grasps.json` and
   the reachability envelope both move. Expect collateral: `DEFAULT_BOARD_X = 0.295` was
   pinned partly on the black queen and king being liftable, which was measured against the
   broken frame. If that argument shifts, `scene.py`'s standing rule applies — board
   placement changes on a visual re-inspection, never on a coverage count alone.
4. **Stop commanding the mechanical stop.** `GRIPPER_CLOSED = 0` drives the servo into
   saturation, ~43 N of pinch on a 15 g piece. Command a grip target sized to the piece
   instead (the opening where the jaw gap equals the stem diameter, less ~1 mm of squeeze).
5. **Re-run and let the check from (1) say what broke.** Whether the grasp survives lift and
   transport is not predictable from the static numbers and is not worth arguing about in
   advance.

Tracked separately, because the grasp fix does not touch them: transport clearance (hover
caps near ~70 mm against a 92 mm tallest piece, so no transport leg near the arm clears an
occupied square — this bounds which moves are generatable at all, and therefore dataset
diversity); Step 7's acceptance criterion, which names a move that has no grasp pose at
either end; and wrist-roll freedom, which is an observation-quality problem rather than a
grasp one, since the stems are circular.

### Noise/diversity requirement — do not skip this

Naive waypoint-IK scripting produces very uniform motion: same approach angle,
same grasp timing, same retreat height, every episode. A large but uniform dataset
generalizes worse than a small diverse one (established finding: ~70 diverse demos
outperform ~500 near-identical ones for sim-to-real transfer). A purely scripted
dataset risks being an even more extreme version of the "uniform data" failure
case, just at higher volume.

Mitigation: inject randomization into *how* each episode is executed, not just
*where* it happens —
- vary approach angle and trajectory shape per episode
- vary waypoint timing / grasp-closure timing
- perturb start pose and intermediate waypoints with small noise
- randomize board position/orientation, piece style, and lighting across episodes
  (scene-level domain randomization, in addition to motion-level randomization)

Goal: a family of varied motions toward the same goal, not one canonical motion
replicated at scale.

### Where human demos still fit (later, not blocking)

A small number of real gamepad/keyboard-teleoperated demos (no leader arm hardware
needed) are planned as a later sanity check against the scripted dataset — not a
prerequisite for the cold start. Purpose: catch any unnatural pattern the scripted
motion might have baked in (e.g. an artificial retreat height) that happens to
work in sim but wouldn't be representative of how the task should actually be
performed.

## Where this runs: MuJoCo on this machine

This host (the Dedibox this agent runs on) has been confirmed as: ~3.8GB RAM
(~2.4GB free), 2 vCPUs, no GPU, ~206GB free disk.

- MuJoCo is the right simulator for this box. Its physics core is lightweight by
  design and runs fine on hardware far smaller than a GPU workstation — the
  available RAM here is sufficient for a single (non-parallelized) simulation
  process.
- Headless rendering without a GPU: use the `osmesa` backend
  (`MUJOCO_GL=osmesa`) — pure CPU software rendering. Not `egl` (requires a GPU)
  and not `glfw` (requires an actual display; won't run headless at all).
- Run one simulation process at a time given the tight RAM headroom — do not
  batch-parallelize multiple MuJoCo instances concurrently on this box.
- The full scripted/IK data-generation harness — running episodes, saving
  images/state/actions/instructions into dataset format — is intended to run
  entirely on this machine. No live viewer or human interaction is needed for
  this cold-start batch, so there's no requirement to stream a display anywhere.
- GPU-parallelized simulators (ManiSkill, Genesis) are explicitly out of scope for
  this machine — they exist specifically for GPU-parallelized simulation, which
  this box doesn't have. If that path is ever pursued, it runs on the same rented
  GPU instance used for training, not here.

## Explicitly out of scope for this phase

- Model architecture/training implementation details (SFT script, hyperparameters)
- RL fine-tuning stage (planned as a later phase, on top of the SFT baseline)
- Real hardware (SO-101), calibration, or deployment
- The human-teleop tooling setup itself (gym-hil configuration, etc.)
