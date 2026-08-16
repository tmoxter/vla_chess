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

### The oracle grasps (fixed 2026-08-16)

The cold start above assumes the scripted oracle can pick a piece up. It could not until this
was worked through, and the run of failures is worth keeping, because every one of them was
invisible until the fix for the previous one exposed it. In order: the TCP sat on the fixed
finger rather than between the jaws; the contact screen forgave the resulting impalement; the
arm's base was pinned at its torque limit by a collision pair MuJoCo does not filter; the grip
was sized off meshes the simulator does not collide; the jaws bore on the pawn's head rather
than its stem; the hand opened wide enough to sweep the neighbouring square; opening the jaw did
not release a piece the fixed finger was still touching; and friction resolved too softly to
hold anything. `docs/trajectory.md` carries the measurement behind each.

`d7 -> d6` now completes: grasped, lifted, carried, placed upright on the target square, arm
back at the survey pose, no other piece moved by more than the 0.09 mm settling floor. The same
plan works for `b7`, `c7` and `e7`. `docs/episode-views/` holds the wrist camera the dataset
carries and a witness view that shows the piece leaving the board.

Two consequences are open, and both are placement decisions rather than code:

1. **The reference set and the board pose.** The corrected pinch site sits ~8 mm further out,
   which costs `d8` and `e8` — 28 of 64 squares against the committed 30. `DEFAULT_BOARD_X`
   is at 0.300 *because* the black queen and king became liftable there, and that was measured
   against the broken frame. Recovering them means moving the board out again and re-searching
   the survey pose, which `scene.py`'s standing rule says happens on a visual re-inspection and
   never on a coverage count alone. Until that inspection, `assets/reference_grasps.json` is
   solved against a frame the episode pipeline no longer uses.
2. **Transport clearance.** Hover caps near ~70 mm against a 92 mm tallest piece, so no
   transport leg near the arm clears an occupied square. This bounds which moves are generatable
   at all, and therefore dataset diversity. `d7 -> d6` works because nothing stands between the
   two squares.

Also still open, unchanged by the grasp work: Step 7's acceptance criterion, which names a move
with no grasp pose at either end; and wrist-roll freedom, which is an observation-quality problem
rather than a grasp one, since the stems are circular.

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
