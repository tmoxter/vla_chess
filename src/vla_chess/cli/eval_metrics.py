"""Quantitative closed-loop metrics for a fine-tuned SmolVLA policy, measured in parallel.

Where `eval_rollout` drives one move and renders an mp4, this harness runs *many* rollouts of a
move and reports rates with confidence intervals. It separates the two things that "success" had
been conflating:

  * **grasp** — did the policy ever close both jaws on the piece (and lift it), and
  * **placement** — did the piece come to rest, upright, on the target square, and
  * **placement at done** — was it standing there when the arm, having let go of it, came back
    to rest (`control.done`): the score a deployment without a fixed episode length would get.

and it measures **trajectory displacement** of each policy rollout against the scripted expert's
trajectory for the same move (joint-space and TCP-space, plus terminal piece error).

It also provides the **expert baseline**: the scripted oracle's own per-frame actions replayed
through the *identical* eval stepping (the fps -> substep cadence used for the policy, not the
executor's native 50 Hz control loop) and scored the same way. That is the sanity floor — if the
expert itself does not place under the eval cadence, the eval, not the policy, is the problem.

Parallelism: a worker pool loads the policy once per process and runs a share of the rollouts. The
binding resource is **GPU memory** (one ~1.8 GB fp32 model copy + CUDA context per worker, so a
16 GB card holds ~5) and **CPU cores** (each rollout is a CPU-bound sim+render loop, ~1 worker per
vCPU) — pick `--workers` as the min of the two. The GPU compute itself is nearly idle.

Runs in the `eval` pixi env. Example (1000 rollouts of the one move that worked, 5 workers):

    pixi run -e eval python -m vla_chess.cli.eval_metrics \
        --repo-id hugginggeorge/smolvla-chess-exp2 --revision 020000 \
        --source d7 --target d6 --fps 30 --n-action-steps 10 \
        --rollouts 1000 --workers 5 --device cuda --out runpod/artifacts/metrics-d7d6.json

    # expert baseline (10 reps of the scripted oracle through the same eval setting):
    pixi run -e eval python -m vla_chess.cli.eval_metrics --expert-baseline \
        --source d7 --target d6 --fps 30 --reps 10 --out runpod/artifacts/expert-d7d6.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco

from vla_chess.cli.eval_rollout import (
    SUBSTEPS_PER_SECOND,
    _load_policy,
    _observation,
    _spec_for,
    parse_moves,
    substeps_per_frame,
)
from vla_chess.cli.generate_dataset import EXP2_TRAINED_MOVES, one_step_moves
from vla_chess.control.done import DoneDetector
from vla_chess.control.executor import Executor
from vla_chess.control.planner import MoveSpec
from vla_chess.dataset.episode import (
    WRIST_CAMERA,
    EpisodeRunner,
    git_engine_version,
    move_scene_description,
)
from vla_chess.dataset.observe import DEFAULT_FPS
from vla_chess.dataset.rollouts import RecordedRollout, rollout_path, save_rollout
from vla_chess.dataset.takeover import HIGH_RELEASE, GripTrace, mid_air_releases
from vla_chess.rl.env import LIFT_CLEARANCE_M
from vla_chess.sensor.render import Renderer
from vla_chess.sim.positions import single_pawn_position
from vla_chess.sim.scene import (
    BIRDS_EYE_ARM_QPOS,
    DEFAULT_BOARD_POSE,
    build_scene,
    make_data,
)
from vla_chess.sim.snapshot import capture_state


def _load_local(mdir: str, device: str) -> tuple[Any, Any, Any]:
    """Load a SmolVLA checkpoint straight from a local directory, no Hub round-trip.

    The Exp-2 final model lives only on disk (it was never pushed to the Hub), so the sweep loads
    the flat `pretrained_model` directory directly rather than through `snapshot_download`.

    Args:
        mdir: Path to a `pretrained_model` directory (config.json + model.safetensors + processors).
        device: Torch device to place the policy on.

    Returns:
        The policy and its pre/post processors, ready for `select_action`.
    """
    import torch
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    torch.set_num_threads(1)  # one thread per worker; the pool, not torch, provides the parallelism
    cfg = SmolVLAConfig.from_pretrained(mdir)
    cfg.device = device
    cfg.use_amp = False
    policy = SmolVLAPolicy.from_pretrained(mdir, config=cfg)
    policy.eval()
    policy.to(device)
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=mdir,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


# Points the policy and expert trajectories are resampled to on a shared normalized time axis
# before they are compared, so a 30 fps rollout and a differently-lengthed expert trace line up
# regardless of their raw step counts.
TRAJECTORY_SAMPLES = 100


@dataclass(frozen=True)
class RolloutMetrics:
    """Per-rollout outcome for one policy draw of a move.

    The displacement fields are NaN when the sweep ran without an expert reference.
    """

    index: int
    move: str  # "source->target"
    grasped: bool  # both jaws ever bore on the piece
    lifted: bool  # ... while the piece was clear of the board
    placed: bool  # came to rest upright on the target square
    opened_mid_air: bool  # the jaws were opened on the piece higher than HIGH_RELEASE
    slipped_mid_air: bool  # the piece left jaws still commanded closed, higher than HIGH_RELEASE
    settled: bool
    final_square: str | None
    terminal_to_target_m: float  # piece resting distance from the target-square centre
    terminal_to_expert_m: float  # piece resting distance from the expert's final piece pose
    joint_disp_mean: float  # mean per-sample arm-joint L2 vs expert (rad)
    joint_disp_max: float
    tcp_disp_mean: float  # mean per-sample TCP L2 vs expert (m)
    tcp_disp_max: float
    done_frame: int | None  # frame the move became done on by `control.done`, None if never
    placed_at_done: bool  # the piece stood on the target, released, on that frame


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    """Point estimate and Wilson score 95% interval for a binomial rate.

    The Wilson interval is used rather than the normal approximation because the rates here sit
    near 0, where the naive interval undercovers and can dip below zero.

    Args:
        successes: Number of successful trials.
        total: Number of trials.
        z: Standard-normal quantile for the interval (1.96 for 95%).

    Returns:
        The observed rate and the lower and upper interval bounds, each in [0, 1]. A zero-trial
        input returns all zeros.
    """
    if total == 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _resample(trace: np.ndarray, samples: int) -> np.ndarray:
    """Resample a (T, D) trajectory to `samples` points on a normalized [0, 1] time axis.

    Length-independent alignment: the policy rollout and the scripted expert run for different
    numbers of steps (30 fps vs the executor's 50 Hz control), so both are stretched onto the same
    axis before any per-point distance is taken.

    Args:
        trace: Per-step values, shape (T, D), T >= 1.
        samples: Number of output points.

    Returns:
        The resampled trajectory, shape (samples, D).
    """
    length = len(trace)
    if length == 1:
        return np.repeat(trace[:1], samples, axis=0)
    source = np.linspace(0.0, 1.0, length)
    target = np.linspace(0.0, 1.0, samples)
    return np.stack([np.interp(target, source, trace[:, d]) for d in range(trace.shape[1])], axis=1)


@dataclass(frozen=True)
class _Reference:
    """The scripted expert's trace for a move, used as the displacement baseline."""

    joint_samples: np.ndarray  # (TRAJECTORY_SAMPLES, 5) arm joints, resampled
    tcp_samples: np.ndarray  # (TRAJECTORY_SAMPLES, 3) TCP position, resampled
    final_piece_xyz: np.ndarray  # (3,) where the expert left the piece
    per_frame_actions: np.ndarray  # (F, 6) expert action per dataset frame, for replay
    grasped: bool
    placed: bool


def _build_move_scene(spec: MoveSpec) -> tuple[Any, Any, Any, Executor, int]:
    """Build the lone-pawn scene for a move and its executor, as the policy rollout sees it.

    Returns:
        The model, scene index, fresh data, an executor bound to them, and the body id of the
        piece being moved.
    """
    pieces = single_pawn_position(spec.source)
    description = move_scene_description(spec, pieces, DEFAULT_BOARD_POSE, BIRDS_EYE_ARM_QPOS)
    model, index = build_scene(description)
    data = make_data(model, index)
    executor = Executor(model, index)
    piece_body = index.piece_body_id[spec.source]
    return model, index, data, executor, piece_body


def expert_reference(spec: MoveSpec, fps: float, seed: int = 0) -> _Reference:
    """Run the scripted oracle for the move and distil the trace the policy is scored against.

    The scripted oracle on the default board — the scene a policy rollout of this move also runs
    in, so the displacement is policy-vs-expert. The seed drives the grasp solver's random
    restarts: `seed=0` is the canonical reference the policy sweep measures against; the expert
    baseline varies the seed per rep so its ten runs are independent draws rather than one
    trajectory scored ten times.

    Args:
        spec: The move to generate the expert for.
        fps: Capture frame rate; sets the per-frame action trace used for replay and comparison.
        seed: Grasp-solver seed for this expert draw.

    Returns:
        The expert reference: resampled joint and TCP traces, its final piece pose, its per-frame
        action trace, and its own grasp/placement verdict.
    """
    runner = EpisodeRunner(
        fps=fps, pieces=single_pawn_position(spec.source), engine_version=git_engine_version()
    )
    episode = runner.run(spec, seed=seed)
    _, _, _, executor, _ = _build_move_scene(spec)
    states = np.stack([f.state for f in episode.frames])  # (F, 6)
    joints = states[:, :5]
    tcp = np.stack([executor.tcp_position(q) for q in joints])  # (F, 3)
    return _Reference(
        joint_samples=_resample(joints, TRAJECTORY_SAMPLES),
        tcp_samples=_resample(tcp, TRAJECTORY_SAMPLES),
        final_piece_xyz=_expert_final_piece(spec),
        per_frame_actions=np.stack([f.action for f in episode.frames]),
        grasped=bool(episode.execution.grasped),
        placed=bool(episode.execution.placed),
    )


def _expert_final_piece(spec: MoveSpec) -> np.ndarray:
    """Target-square centre as the expert's intended final piece position.

    The expert places cleanly, so the target-square centre is its resting pose to within the
    settle tolerance; using the geometric centre avoids re-stepping the sim to read the body.
    """
    _, index, _, _, _ = _build_move_scene(spec)
    return np.asarray(index.board.square_to_world(spec.target, index.board_pose), dtype=float)


def _drive_once(
    policy: Any,
    pre: Any,
    post: Any,
    spec: MoveSpec,
    max_frames: int,
    frame_substeps: int,
    index_label: int,
    reference: _Reference | None,
    record: Path | None = None,
) -> RolloutMetrics:
    """Drive one policy rollout of a move, logging grasp, placement and displacement.

    Mirrors `eval_rollout.drive` but additionally polls the two-jaw grasp each frame, tracks the
    piece height to tell a lift from a grounded touch, and scores the trajectory against the
    scripted expert reference. No frames are rendered to disk — only the metrics are kept.

    Args:
        policy, pre, post: The loaded SmolVLA policy and its processors.
        spec: The move to drive.
        max_frames: Frames to drive before scoring.
        frame_substeps: Sim substeps advanced per policy step (from `substeps_per_frame(fps)`).
        index_label: This rollout's index, carried into the result for logging.
        reference: The expert trace to measure displacement against; None skips the
            displacement and reports it as NaN.
        record: Directory to save the rollout's per-frame simulator snapshots under, as
            `rollout-<index>.npz`; nothing is saved when None.

    Returns:
        The rollout's grasp/placement/displacement metrics.
    """
    import torch

    model, index, data, executor, piece_body = _build_move_scene(spec)
    renderer = Renderer(model, [WRIST_CAMERA])
    surface = index.board.surface_height
    instruction = spec.instruction()
    policy.reset()

    joints: list[np.ndarray] = []
    snapshots: list[np.ndarray] = []
    held: list[bool] = []
    heights: list[float] = []
    jaw_commands: list[float] = []
    grasped = lifted = False
    done = DoneDetector()
    placed_at_done = False
    for _ in range(max_frames):
        if record is not None:
            snapshots.append(capture_state(model, data))
        frame = np.array(renderer.render(data)[WRIST_CAMERA.name], copy=True)
        state = executor._state(data)
        joints.append(state[:5].astype(np.float64))
        obs = _observation(frame, state, instruction)
        with torch.inference_mode():
            action = post(policy.select_action(pre(obs)))
        action_np = action.squeeze(0).to(torch.float32).cpu().numpy()
        executor._write_ctrl(data, action_np[:5], float(action_np[5]))
        for _ in range(frame_substeps):
            mujoco.mj_step(model, data)
        holding = executor._holding(data, piece_body)
        height = float(data.xpos[piece_body][2]) - surface
        held.append(holding)
        heights.append(height)
        jaw_commands.append(float(data.ctrl[index.gripper_actuator_id]))
        if holding:
            grasped = True
            if height > LIFT_CLEARANCE_M:
                lifted = True
        if done.done_frame is None and done.update(executor._arm_qpos(data), holding):
            placed_at_done = bool(executor.placement(data, spec.source, spec.target).placed)

    releases = mid_air_releases(
        GripTrace(
            gripped=np.array(held, dtype=bool),
            height=np.array(heights),
            jaw_command=np.array(jaw_commands),
        ),
        HIGH_RELEASE,
    )
    check = executor.placement(data, spec.source, spec.target)
    piece_xyz = np.asarray(data.xpos[piece_body], dtype=float)
    target_xyz = np.asarray(index.board.square_to_world(spec.target, index.board_pose), dtype=float)

    if record is not None:
        save_rollout(
            rollout_path(record, index_label),
            RecordedRollout(
                source=spec.source,
                target=spec.target,
                instruction=instruction,
                snapshots=np.stack(snapshots),
                grasped=grasped,
                placed=bool(check.placed),
                final_square=check.final_square,
                fps=SUBSTEPS_PER_SECOND / frame_substeps,
            ),
        )

    if reference is None:
        joint_l2 = tcp_l2 = np.array([np.nan])
        expert_xy = np.full(2, np.nan)
    else:
        policy_joints = _resample(np.stack(joints), TRAJECTORY_SAMPLES)
        policy_tcp = _resample(
            np.stack([executor.tcp_position(q) for q in np.stack(joints)]), TRAJECTORY_SAMPLES
        )
        joint_l2 = np.linalg.norm(policy_joints - reference.joint_samples, axis=1)
        tcp_l2 = np.linalg.norm(policy_tcp - reference.tcp_samples, axis=1)
        expert_xy = reference.final_piece_xyz[:2]

    return RolloutMetrics(
        index=index_label,
        move=f"{spec.source}->{spec.target}",
        grasped=grasped,
        lifted=lifted,
        placed=bool(check.placed),
        opened_mid_air=any(r.opened for r in releases),
        slipped_mid_air=any(not r.opened for r in releases),
        settled=bool(check.settled),
        final_square=check.final_square,
        terminal_to_target_m=float(np.linalg.norm(piece_xyz[:2] - target_xyz[:2])),
        terminal_to_expert_m=float(np.linalg.norm(piece_xyz[:2] - expert_xy)),
        joint_disp_mean=float(joint_l2.mean()),
        joint_disp_max=float(joint_l2.max()),
        tcp_disp_mean=float(tcp_l2.mean()),
        tcp_disp_max=float(tcp_l2.max()),
        done_frame=done.done_frame,
        placed_at_done=placed_at_done,
    )


# --- worker-pool plumbing: each process loads the policy once and keeps it in a module global ---
_WORKER: dict[str, Any] = {}


def _init_worker(
    repo_id: str,
    revision: str,
    device: str,
    n_action_steps: int | None,
    moves: list[tuple[str, str]],
    episode: int,
    fps: float,
    max_frames: int,
    local_checkpoint: str | None,
    with_reference: bool,
    record: Path | None,
) -> None:
    """Load the policy once in this worker process and cache it with the moves it drives."""
    if local_checkpoint is not None:
        policy, pre, post = _load_local(local_checkpoint, device)
    else:
        policy, pre, post = _load_policy(repo_id, revision, device)
    if n_action_steps is not None:
        policy.config.n_action_steps = n_action_steps
    _WORKER.update(
        policy=policy,
        pre=pre,
        post=post,
        specs=[_spec_for(episode, source, target) for source, target in moves],
        references={},
        with_reference=with_reference,
        fps=fps,
        max_frames=max_frames,
        frame_substeps=substeps_per_frame(fps),
        record=record,
    )


def _run_index(index_label: int) -> RolloutMetrics:
    """Run one rollout in a pool worker using the process-cached policy.

    Rollout `i` drives move `i` modulo the move list, so every move gets an even share. Each
    move's expert reference is built the first time this worker drives it.
    """
    w = _WORKER
    spec = w["specs"][index_label % len(w["specs"])]
    reference = None
    if w["with_reference"]:
        key = (spec.source, spec.target)
        if key not in w["references"]:
            w["references"][key] = expert_reference(spec, w["fps"])
        reference = w["references"][key]
    return _drive_once(
        w["policy"],
        w["pre"],
        w["post"],
        spec,
        w["max_frames"],
        w["frame_substeps"],
        index_label,
        reference,
        w["record"],
    )


def sweep_moves(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Resolve which moves a sweep drives from its command line.

    Args:
        args: Parsed options carrying `moves`, `trained_one_step`, `source` and `target`.

    Returns:
        The (source, target) pairs, in the order rollouts cycle through them.

    Raises:
        ValueError: If no move is named, or more than one way of naming them is used.
    """
    named = [args.moves is not None, args.trained_one_step, args.source is not None]
    if sum(named) != 1:
        raise ValueError(
            "name the moves with exactly one of --source/--target, --moves, --trained-one-step"
        )
    if args.moves is not None:
        return parse_moves(args.moves)
    if args.trained_one_step:
        return one_step_moves(EXP2_TRAINED_MOVES)
    if args.target is None:
        raise ValueError("--source needs --target")
    return [(args.source, args.target)]


def run_policy_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """Run `args.rollouts` policy rollouts of one move across a worker pool and aggregate.

    Returns:
        A report dict: grasp/lift/placement rates with Wilson 95% intervals, displacement
        summaries, and the full per-rollout records.
    """
    import multiprocessing as mp

    moves = sweep_moves(args)
    init = (
        args.repo_id,
        args.revision,
        args.device,
        args.n_action_steps,
        moves,
        args.episode,
        args.fps,
        args.max_frames,
        args.local_checkpoint,
        not args.no_reference,
        args.record_states,
    )
    started = time.monotonic()
    records: list[RolloutMetrics] = []
    if args.workers <= 1:
        _init_worker(*init)
        for i in range(args.rollouts):
            records.append(_run_index(i))
            print(f"[{int(time.monotonic() - started)}s] {i + 1}/{args.rollouts}", flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_init_worker, initargs=init) as pool:
            for done, rec in enumerate(pool.imap_unordered(_run_index, range(args.rollouts)), 1):
                records.append(rec)
                print(f"[{int(time.monotonic() - started)}s] {done}/{args.rollouts}", flush=True)

    records.sort(key=lambda r: r.index)
    n = len(records)
    grasp_p, grasp_lo, grasp_hi = wilson_interval(sum(r.grasped for r in records), n)
    lift_p, lift_lo, lift_hi = wilson_interval(sum(r.lifted for r in records), n)
    place_p, place_lo, place_hi = wilson_interval(sum(r.placed for r in records), n)
    opened_p, opened_lo, opened_hi = wilson_interval(sum(r.opened_mid_air for r in records), n)
    slipped_p, slipped_lo, slipped_hi = wilson_interval(sum(r.slipped_mid_air for r in records), n)
    done_p, done_lo, done_hi = wilson_interval(sum(r.done_frame is not None for r in records), n)
    at_done_p, at_done_lo, at_done_hi = wilson_interval(sum(r.placed_at_done for r in records), n)

    def summary(values: list[float]) -> dict[str, float]:
        a = np.asarray(values)
        if np.all(np.isnan(a)):
            return {"mean": math.nan, "median": math.nan, "p95": math.nan, "max": math.nan}
        return {
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)),
            "max": float(a.max()),
        }

    per_move: dict[str, dict[str, int]] = {}
    for r in records:
        tally = per_move.setdefault(
            r.move,
            {
                "rollouts": 0,
                "grasped": 0,
                "lifted": 0,
                "placed": 0,
                "opened_mid_air": 0,
                "placed_at_done": 0,
            },
        )
        tally["rollouts"] += 1
        tally["grasped"] += int(r.grasped)
        tally["lifted"] += int(r.lifted)
        tally["placed"] += int(r.placed)
        tally["opened_mid_air"] += int(r.opened_mid_air)
        tally["placed_at_done"] += int(r.placed_at_done)

    return {
        "mode": "policy_sweep",
        "move": ",".join(f"{s}->{t}" for s, t in moves),
        "per_move": per_move,
        "record_states": None if args.record_states is None else str(args.record_states),
        # A local checkpoint is loaded instead of the Hub repo, so the repo id and revision would
        # name a policy that was never scored.
        "repo_id": None if args.local_checkpoint else args.repo_id,
        "revision": None if args.local_checkpoint else args.revision,
        "local_checkpoint": args.local_checkpoint,
        "engine_version": git_engine_version(),
        "fps": args.fps,
        "n_action_steps": args.n_action_steps,
        "max_frames": args.max_frames,
        "rollouts": n,
        "workers": args.workers,
        "wall_seconds": round(time.monotonic() - started, 1),
        "grasp_rate": {"p": grasp_p, "ci95": [grasp_lo, grasp_hi]},
        "lift_rate": {"p": lift_p, "ci95": [lift_lo, lift_hi]},
        "placement_rate": {"p": place_p, "ci95": [place_lo, place_hi]},
        "opened_mid_air_rate": {"p": opened_p, "ci95": [opened_lo, opened_hi]},
        "slipped_mid_air_rate": {"p": slipped_p, "ci95": [slipped_lo, slipped_hi]},
        # A move is done once the arm let go and came back to rest (`control.done`); scored there,
        # a move that never got done within max_frames is a failure, as it would be on the arm.
        "done_rate": {"p": done_p, "ci95": [done_lo, done_hi]},
        "placement_at_done_rate": {"p": at_done_p, "ci95": [at_done_lo, at_done_hi]},
        "terminal_to_target_m": summary([r.terminal_to_target_m for r in records]),
        "joint_disp_mean_rad": summary([r.joint_disp_mean for r in records]),
        "tcp_disp_mean_m": summary([r.tcp_disp_mean for r in records]),
        "records": [asdict(r) for r in records],
    }


def _replay_expert_through_eval(
    spec: MoveSpec, actions: np.ndarray, frame_substeps: int
) -> tuple[bool, bool, bool, str | None]:
    """Replay a scripted action trace through the policy eval's stepping and score it.

    The expert's per-frame actions are written one per frame and stepped `frame_substeps` substeps
    each — the exact cadence the policy rollout runs at, which differs from the executor's native
    50 Hz control loop. This is what "run the expert through the same eval setting" measures.

    Returns:
        Whether the piece was grasped (two jaws), lifted clear, placed, and its final square.
    """
    model, index, data, executor, piece_body = _build_move_scene(spec)
    surface = index.board.surface_height
    grasped = lifted = False
    for action in actions:
        executor._write_ctrl(data, action[:5], float(action[5]))
        for _ in range(frame_substeps):
            mujoco.mj_step(model, data)
        if executor._holding(data, piece_body):
            grasped = True
            if float(data.xpos[piece_body][2]) - surface > LIFT_CLEARANCE_M:
                lifted = True
    check = executor.placement(data, spec.source, spec.target)
    return grasped, lifted, bool(check.placed), check.final_square


def run_expert_baseline(args: argparse.Namespace) -> dict[str, Any]:
    """Replay the scripted oracle through the eval cadence `args.reps` times and aggregate.

    Each rep regenerates the oracle with its own seed, so the ten runs are independent expert
    draws, then replays that draw through the eval cadence. A clean recipe is `reps`/`reps`; any
    failure is the eval cadence (fps -> substeps) failing to preserve a trajectory the executor's
    own 50 Hz loop placed — exactly the floor to rule out before blaming the policy.

    Returns:
        A report dict: grasp and placement counts over the reps and the per-rep records.
    """
    spec = _spec_for(args.episode, args.source, args.target)
    frame_substeps = substeps_per_frame(args.fps)
    reps: list[dict[str, Any]] = []
    native_grasped = native_placed = 0
    started = time.monotonic()
    for r in range(args.reps):
        reference = expert_reference(spec, args.fps, seed=r)
        native_grasped += int(reference.grasped)
        native_placed += int(reference.placed)
        grasped, lifted, placed, final = _replay_expert_through_eval(
            spec, reference.per_frame_actions, frame_substeps
        )
        reps.append(
            {
                "rep": r,
                "grasped": grasped,
                "lifted": lifted,
                "placed": placed,
                "final_square": final,
            }
        )
        print(
            f"[{int(time.monotonic() - started)}s] expert rep {r + 1}/{args.reps}: "
            f"{'PLACED' if placed else 'fail'} (grasp={grasped}, ended {final})",
            flush=True,
        )
    return {
        "mode": "expert_baseline",
        "move": f"{args.source}->{args.target}",
        "fps": args.fps,
        "reps": args.reps,
        "frame_substeps": frame_substeps,
        "native_executor_grasped": native_grasped,
        "native_executor_placed": native_placed,
        "grasp_count": sum(x["grasped"] for x in reps),
        "placement_count": sum(x["placed"] for x in reps),
        "records": reps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="hugginggeorge/smolvla-chess-exp2")
    parser.add_argument("--revision", default="020000")
    parser.add_argument("--source", default=None, help="start square, e.g. d7")
    parser.add_argument("--target", default=None, help="target square, e.g. d6")
    parser.add_argument(
        "--moves",
        default=None,
        help="policy sweep over several moves, as source:target pairs (e.g. 'c7:c6,d7:d6'); "
        "rollouts cycle through them",
    )
    parser.add_argument(
        "--trained-one-step",
        action="store_true",
        help="policy sweep over every one-step pawn push the funnel datasets train on",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="skip the scripted expert reference; displacement metrics are reported as NaN",
    )
    parser.add_argument(
        "--record-states",
        type=Path,
        default=None,
        help="save each rollout's per-frame simulator snapshots under this directory, for "
        "expert takeovers",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    # 25 s at 30 fps. The expert's own episode is ~16-17 s, so a tighter budget scores a policy
    # that is merely slower than the expert while the piece is still in its hand.
    parser.add_argument("--max-frames", type=int, default=750)
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="override the policy's action-chunk horizon (e.g. 10)",
    )
    parser.add_argument("--rollouts", type=int, default=1000, help="policy rollouts to run")
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="parallel worker processes; one model copy (~2.5-3 GB VRAM) each",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--local-checkpoint",
        default=None,
        help="load the policy from this local pretrained_model dir instead of the "
        "Hub (the Exp-2 final model lives only on disk)",
    )
    parser.add_argument("--reps", type=int, default=10, help="expert-baseline repetitions")
    parser.add_argument(
        "--expert-baseline",
        action="store_true",
        help="replay the scripted oracle through the eval cadence instead of running the policy",
    )
    parser.add_argument("--out", type=Path, required=True, help="where the JSON report is written")
    args = parser.parse_args()
    if args.expert_baseline and (args.source is None or args.target is None):
        parser.error("--expert-baseline needs --source and --target")

    report = run_expert_baseline(args) if args.expert_baseline else run_policy_sweep(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}", flush=True)
    if args.expert_baseline:
        print(
            f"=== expert baseline: placed {report['placement_count']}/{report['reps']}, "
            f"grasped {report['grasp_count']}/{report['reps']} ===",
            flush=True,
        )
    else:
        g, p = report["grasp_rate"], report["placement_rate"]
        d = report["placement_at_done_rate"]
        print(
            f"=== grasp {g['p']:.1%} (95% CI {g['ci95'][0]:.1%}-{g['ci95'][1]:.1%}) | "
            f"placement {p['p']:.1%} (95% CI {p['ci95'][0]:.1%}-{p['ci95'][1]:.1%}) | "
            f"placement at done {d['p']:.1%} (95% CI {d['ci95'][0]:.1%}-{d['ci95'][1]:.1%}) "
            f"over {report['rollouts']} rollouts ===",
            flush=True,
        )


if __name__ == "__main__":
    main()
