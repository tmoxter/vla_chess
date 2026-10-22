"""Generate a batch of pick-and-place episodes into one LeRobot dataset, resumably.

Two phases. First, every episode is generated and written to its own capture directory; a run
that is interrupted and restarted skips the captures already on disk, so the expensive physics is
never repeated. Second, once the captures exist, they are serialized into a LeRobot dataset in a
single pass that can be rerun at will — it writes to a temporary directory and swaps it into place
only when complete.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from vla_chess.analysis.reachability import (
    STEP4_XY_JITTER,
    STEP4_YAW_JITTER_DEG,
    sample_board_pose,
)
from vla_chess.control.planner import MoveSpec, PlanInfeasible
from vla_chess.control.sampling import DEFAULT_ATTEMPTS, SampleKnobs
from vla_chess.dataset.capture import load_capture, save_capture
from vla_chess.dataset.episode import CandidateSelection, EpisodeRunner, git_engine_version
from vla_chess.dataset.observe import DEFAULT_FPS
from vla_chess.sim.positions import single_pawn_position
from vla_chess.sim.scene import DEFAULT_BOARD_POSE, PiecePlacement, PieceType

if TYPE_CHECKING:
    # writer imports lerobot, which lives only in the datagen env; keep it out of this module's
    # import so the batch driver and its pure helpers still load under the default env.
    from vla_chess.dataset.writer import DatasetStats

# The pawn-forward moves that plan at the current grasp frame: a piece one square forward with
# nothing between source and target to foul the transport. These are the four centre files —
# the outer files (a, b, g, h) have no clean fingers-down hover over the rank-6 target, which is
# farther from the arm than the rank-7 source. Confirmed by probing plan_move over all eight
# files; d7->d6 is the fully-executed reference in docs/trajectory.md.
WORKING_MOVES: Final = (("c7", "c6"), ("d7", "d6"), ("e7", "e6"), ("f7", "f6"))

# The label the lone-pawn demonstrations carry, in place of the coordinate-naming default. With
# a single pawn on the board there is nothing to disambiguate it from, so the task is the push
# itself; the squares still cycle through WORKING_MOVES so "forward" is not tied to one file.
SINGLE_PAWN_INSTRUCTION: Final = "move the pawn one field forward"

# Exp 2 distance-specific labels. Image + state alone can't tell a one- from a two-square push from
# the same start, so the distance has to ride in the language — which also probes grounding.
ONE_STEP_INSTRUCTION: Final = "move the pawn one field forward"
TWO_STEP_INSTRUCTION: Final = "move the pawn two fields forward"

# The Exp 2 trained move pool (see experiment-2-plan.md, Phase 1 step 3), expanded 2026-10-10 once
# the tilted-grasp funnel and the wrist-roll fix opened up the outer files and a deeper rank. Black
# pawns pushed forward. Two bands, every entry physics-validated (real screened plan + dry-run on
# the funnel's single-pawn layout — it grasps and places, not just plans):
#   mid-board, ALL eight files {7->6, 6->5 one-step, 7->5 two-step};
#   rank-4 depth where it still places — one-step 5->4 on the left/centre files {a,b,c,d,f} and
#   two-step 6->4 on the centre files {c,d,e,f}. The right-edge rank-4 pushes (g5->g4, h5->h4) do
#   not grasp/place on the baseline candidate — a left/right asymmetry from the shoulder offset —
#   so they are left out until the sampling funnel is shown to recover them.
# Minus the held-out moves below, never generated here. Expanding to train b/g trades away Exp 2's
# original unseen-FILE generalization axis; EXP2_HELD_OUT keeps an unseen-SQUARE and unseen-DEPTH
# probe in its place, rebuilt live by the rollout harness rather than trained.
EXP2_HELD_OUT: Final = (
    ("d6", "d5"),  # unseen one-step square on a trained file
    ("f7", "f5"),  # unseen two-step push on a trained file
    ("e5", "e4"),  # unseen depth: rank-4 one-step the policy only saw reached via a two-step
)
EXP2_TRAINED_MOVES: Final = (
    ("a7", "a6"),
    ("a6", "a5"),
    ("a7", "a5"),
    ("b7", "b6"),
    ("b6", "b5"),
    ("b7", "b5"),
    ("c7", "c6"),
    ("c6", "c5"),
    ("c7", "c5"),
    ("d7", "d6"),
    ("d7", "d5"),
    ("e7", "e6"),
    ("e6", "e5"),
    ("e7", "e5"),
    ("f7", "f6"),
    ("f6", "f5"),
    ("g7", "g6"),
    ("g6", "g5"),
    ("g7", "g5"),
    ("h7", "h6"),
    ("h6", "h5"),
    ("h7", "h5"),
    # rank-4 one-step (left/centre; g, h fail the baseline candidate)
    ("a5", "a4"),
    ("b5", "b4"),
    ("c5", "c4"),
    ("d5", "d4"),
    ("f5", "f4"),
    # rank-4 two-step (centre)
    ("c6", "c4"),
    ("d6", "d4"),
    ("e6", "e4"),
    ("f6", "f4"),
)

DEFAULT_EPISODES: Final = 30
DEFAULT_SEEDS: Final = 6
DEFAULT_OUT: Final = Path("data/chess_pick_place")


def forward_instruction(source: str, target: str) -> str:
    """Phrase a pawn-forward move by its distance, one or two squares.

    Args:
        source: Algebraic start square, e.g. "d7".
        target: Algebraic destination square on the same file, e.g. "d5".

    Returns:
        The one- or two-field forward label matching the rank distance.

    Raises:
        ValueError: If source and target are not on the same file, or the forward distance is
            neither one nor two squares.
    """
    if source[0] != target[0]:
        raise ValueError(f"{source}->{target} is not a same-file pawn push")
    distance = abs(int(source[1]) - int(target[1]))
    if distance == 1:
        return ONE_STEP_INSTRUCTION
    if distance == 2:
        return TWO_STEP_INSTRUCTION
    raise ValueError(f"{source}->{target} is neither a one- nor two-square push")


def one_step_moves(moves: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    """Keep only the single-square forward pushes from a move pool.

    The first fine-tune demonstrates one task — a pawn pushed one square forward — under a single
    constant instruction (see `--single-task`). A two-square push cannot honestly carry that label,
    and image+state alone can't tell it apart from a one-square push, so the two-steppers are
    dropped here rather than relabelled into contradictory supervision.

    Args:
        moves: The (source, target) pawn pushes to filter, each a same-file forward move.

    Returns:
        The subset whose forward distance is exactly one rank, in input order.
    """
    return [(s, t) for s, t in moves if abs(int(s[1]) - int(t[1])) == 1]


def build_move_pool_specs(
    moves: Sequence[tuple[str, str]] = EXP2_TRAINED_MOVES,
    seeds: int = DEFAULT_SEEDS,
) -> list[MoveSpec]:
    """Lay out a move pool with a distance-specific label, repeated across seeds.

    Every move is generated `seeds` times; the per-episode seed (assigned downstream by the
    generator) only reseeds the grasp solver's random restarts on a fixed scene, so these repeats
    are near-identical — the board-pose and random-candidate diversity lives in the `--funnel
    --diverse` path, not here. Episodes are grouped by seed slot so an interrupted, resumed run
    refills in a stable order.

    Args:
        moves: The (source, target) pawn pushes to demonstrate, each a same-file forward move.
        seeds: How many jittered repeats of the whole pool to generate.

    Returns:
        `len(moves) * seeds` specs, one per episode, in generation order.
    """
    return [
        MoveSpec(
            source=source,
            target=target,
            piece_type=PieceType.PAWN,
            instruction_override=forward_instruction(source, target),
        )
        for _ in range(seeds)
        for source, target in moves
    ]


def build_specs(
    episodes: int,
    moves: Sequence[tuple[str, str]] = WORKING_MOVES,
    instruction: str | None = None,
) -> list[MoveSpec]:
    """Lay out the moves for a batch, cycling through the working squares.

    Args:
        episodes: How many episodes the batch holds.
        moves: The (source, target) squares to cycle through, all pawns.
        instruction: Label to stamp on every spec, overriding the coordinate-naming default;
            the default phrasing is kept when omitted.

    Returns:
        One spec per episode, in generation order.
    """
    return [
        MoveSpec(
            source=moves[i % len(moves)][0],
            target=moves[i % len(moves)][1],
            piece_type=PieceType.PAWN,
            instruction_override=instruction,
        )
        for i in range(episodes)
    ]


def capture_dir(captures_root: Path, index: int) -> Path:
    """Name the capture directory for one episode index."""
    return captures_root / f"ep-{index:04d}"


def pending_indices(captures_root: Path, episodes: int) -> list[int]:
    """List the episode indices that still need generating.

    Args:
        captures_root: Directory the per-episode captures live under.
        episodes: Total number of episodes in the batch.

    Returns:
        The indices whose capture directory does not yet exist, in order.
    """
    return [i for i in range(episodes) if not capture_dir(captures_root, i).exists()]


def generate(
    specs: Sequence[MoveSpec],
    captures_root: Path,
    master_seed: int,
    fps: float,
    pieces_for: Callable[[MoveSpec], Sequence[PiecePlacement] | None] | None = None,
) -> None:
    """Generate every missing episode into its capture directory.

    Episodes whose capture already exists are skipped, so an interrupted run resumes where it
    stopped. A spec that cannot be planned is logged and left uncaptured rather than aborting
    the batch.

    Args:
        specs: The moves to generate, in order; the seed of episode i is `master_seed + i`.
        captures_root: Directory the per-episode captures are written under.
        master_seed: Seed of the first episode.
        fps: Capture frame rate.
        pieces_for: Board layout to run a spec against, one per spec — used when the layout
            depends on the move, as the lone-pawn slice's does. When omitted, or when it
            returns None, the runner's default opening position is used.
    """
    engine_version = git_engine_version()
    total = len(specs)
    for index, spec in enumerate(specs):
        target = capture_dir(captures_root, index)
        head = f"[{index + 1}/{total}] {spec.source}->{spec.target}"
        if target.exists():
            print(f"{head} already captured, skipping", flush=True)
            continue
        pieces = pieces_for(spec) if pieces_for is not None else None
        runner = EpisodeRunner(fps=fps, pieces=pieces, engine_version=engine_version)
        print(f"{head} generating (seed {master_seed + index})...", flush=True)
        started = time.monotonic()
        try:
            episode = runner.run(spec, seed=master_seed + index)
        except PlanInfeasible as error:
            print(f"{head} could not be planned, skipping: {error}", flush=True)
            continue
        if not episode.frames:
            print(f"{head} produced no frames, skipping", flush=True)
            continue
        save_capture(target, episode)
        result = episode.execution
        print(
            f"{head} captured {len(episode.frames)} frames in {time.monotonic() - started:.0f} s "
            f"(grasped={result.grasped} placed={result.placed})",
            flush=True,
        )


FUNNEL_MANIFEST_NAME: Final = "funnel-manifest.json"


@dataclass
class MoveYield:
    """Running tally of one move's funnel outcomes across its seed slots."""

    slots: int = 0
    kept: int = 0
    attempts: int = 0
    rejects: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, object]:
        """Render the tally for the manifest.

        Returns:
            The slot, kept, attempt counts and the reject histogram as plain JSON types.
        """
        return {
            "slots": self.slots,
            "kept": self.kept,
            "attempts": self.attempts,
            "rejects": dict(self.rejects),
        }


def generate_funnel(
    moves: Sequence[tuple[str, str]],
    seeds: int,
    knobs: SampleKnobs,
    captures_root: Path,
    master_seed: int,
    fps: float,
    instruction: str | None = None,
    selection: CandidateSelection = CandidateSelection.FIRST,
    board_xy_jitter: float = 0.0,
    board_yaw_jitter_deg: float = 0.0,
) -> dict[str, object]:
    """Generate one kept episode per (move, seed slot) through the sample-then-filter funnel.

    Every slot samples up to `knobs.attempts` candidates and keeps one that places the piece — the
    first under `FIRST` selection, one drawn at random under `RANDOM`; a slot that keeps none
    captures nothing and is logged as a miss. When a board jitter is set, each slot also draws its
    own board pose from the Step 4 envelope, so the kept episodes spread over board placements as
    well as grasp poses rather than repeating one fixed scene. Slots whose capture already exists
    are skipped, so an interrupted run resumes. The per-move yield — attempts spent, episodes kept,
    and why the rejects failed — is returned and written beside the captures.

    Args:
        moves: The (source, target) pawn pushes to demonstrate, each a same-file forward move.
        seeds: How many slots of the whole pool to generate.
        knobs: The candidate distribution each slot draws from.
        captures_root: Directory the per-episode captures are written under.
        master_seed: Base seed; slot i draws candidates from `master_seed + i * knobs.attempts`.
        fps: Capture frame rate.
        instruction: Constant task label to stamp on every kept episode, overriding the
            distance-specific default; the one-/two-field phrasing is kept when omitted.
        selection: Whether each slot keeps the first accepted candidate or one at random.
        board_xy_jitter: Largest per-slot board-centre offset per planar axis, metres; zero
            pins every slot to the nominal board pose.
        board_yaw_jitter_deg: Largest per-slot board-yaw offset, degrees; zero pins the yaw.

    Returns:
        The yield manifest: total slots, kept count, and per-move attempts/kept/reject tallies.
    """
    engine_version = git_engine_version()
    total = seeds * len(moves)
    kept = 0
    per_move: dict[str, MoveYield] = {}
    reason_totals: Counter[str] = Counter()
    index = 0
    for slot in range(seeds):
        for source, target in moves:
            name = f"{source}->{target}"
            stats = per_move.setdefault(name, MoveYield())
            head = f"[{index + 1}/{total}] {name} slot {slot}"
            target_dir = capture_dir(captures_root, index)
            if target_dir.exists():
                print(f"{head} already captured, skipping", flush=True)
                index += 1
                continue
            move = MoveSpec(
                source=source,
                target=target,
                piece_type=PieceType.PAWN,
                instruction_override=(
                    instruction if instruction is not None else forward_instruction(source, target)
                ),
            )
            board_pose = (
                sample_board_pose(
                    np.random.default_rng([master_seed, index]),
                    xy_jitter=board_xy_jitter,
                    yaw_jitter_deg=board_yaw_jitter_deg,
                )
                if board_xy_jitter > 0.0 or board_yaw_jitter_deg > 0.0
                else DEFAULT_BOARD_POSE
            )
            runner = EpisodeRunner(
                fps=fps,
                pieces=single_pawn_position(source),
                board_pose=board_pose,
                engine_version=engine_version,
            )
            base_seed = master_seed + index * knobs.attempts
            print(f"{head} sampling up to {knobs.attempts} candidates...", flush=True)
            started = time.monotonic()
            outcome = runner.run_funnel(move, knobs, base_seed=base_seed, selection=selection)

            stats.slots += 1
            stats.attempts += len(outcome.attempts)
            for verdict in outcome.attempts:
                if not verdict.accepted and verdict.reason is not None:
                    stats.rejects[verdict.reason.value] += 1
                    reason_totals[verdict.reason.value] += 1

            if outcome.accepted and outcome.episode is not None and outcome.episode.frames:
                save_capture(target_dir, outcome.episode)
                kept += 1
                stats.kept += 1
                print(
                    f"{head} kept after {len(outcome.attempts)} attempt(s) in "
                    f"{time.monotonic() - started:.0f} s",
                    flush=True,
                )
            else:
                print(
                    f"{head} no keeper in {len(outcome.attempts)} attempt(s), skipping",
                    flush=True,
                )
            index += 1

    manifest: dict[str, object] = {
        "slots": total,
        "kept": kept,
        "attempts_per_slot": knobs.attempts,
        "selection": selection.value,
        "board_xy_jitter": board_xy_jitter,
        "board_yaw_jitter_deg": board_yaw_jitter_deg,
        "reason_totals": dict(reason_totals),
        "per_move": {name: stat.as_dict() for name, stat in per_move.items()},
    }
    (captures_root / FUNNEL_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return manifest


def serialize(captures_root: Path, out: Path, repo_id: str | None = None) -> DatasetStats:
    """Write every capture into one LeRobot dataset, in index order.

    The dataset is built in a temporary directory and swapped into `out` only once finalized,
    so a dataset that exists at `out` is always complete and this step can be rerun freely.

    Args:
        captures_root: Directory holding the per-episode captures.
        out: Where the finished dataset goes; replaced if it already exists.
        repo_id: LeRobot repository id stamped into the dataset; the writer's default is used
            when omitted.

    Returns:
        The episode and frame counts of the written dataset.
    """
    from vla_chess.dataset.writer import DEFAULT_REPO_ID, LeRobotWriter

    captures = sorted(path for path in captures_root.glob("ep-*") if path.is_dir())
    if not captures:
        raise FileNotFoundError(f"no captures to serialize under {captures_root}")

    staging = out.with_name(out.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    writer = LeRobotWriter(staging, repo_id=repo_id or DEFAULT_REPO_ID)
    for path in captures:
        writer.add_episode(load_capture(path))
    stats = writer.finalize()

    if out.exists():
        shutil.rmtree(out)
    staging.rename(out)
    return stats


def push(out: Path, repo_id: str, *, private: bool) -> None:
    """Push a serialized dataset to the Hub, tagged for training.

    Args:
        out: Directory the dataset was serialized into.
        repo_id: Hub repository id to push to; also the id stamped into the dataset.
        private: Whether the Hub repository is created private.
    """
    from vla_chess.dataset.writer import push_dataset

    visibility = "private" if private else "public"
    print(f"pushing {out} to Hub repo {repo_id} ({visibility})...", flush=True)
    push_dataset(out, repo_id, private=private)
    print(f"pushed {repo_id} with the v3.0 version tag", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the batch generator's command line.

    Args:
        argv: Command line without the program name; `sys.argv` is read when omitted.

    Returns:
        The parsed options.
    """
    parser = argparse.ArgumentParser(
        prog="generate-dataset",
        description="Generate a batch of pick-and-place episodes into one LeRobot dataset, "
        "resuming from per-episode captures if interrupted.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where the dataset goes")
    parser.add_argument(
        "--captures",
        type=Path,
        default=None,
        help="where per-episode captures are staged; defaults to '<out>-captures'",
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument(
        "--move-pool",
        action="store_true",
        help="generate the Exp 2 trained move pool (16 pawn-forward moves across files "
        "a/c/d/e/f/h, one- and two-step, distance carried in the instruction) repeated "
        "--seeds times; overrides --episodes. Single-pawn boards, like --single-pawn.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_SEEDS,
        help=f"jittered repeats of the move pool when --move-pool is set (default {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--funnel",
        action="store_true",
        help="sample-then-filter mode: for each move in the Exp 2 pool, sample tilted-grasp "
        "candidates and keep the first that physically places the piece, repeated --seeds slots. "
        "Uses the relaxed tilted IK abort and writes a yield manifest. Single-pawn boards.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=DEFAULT_ATTEMPTS,
        help=f"candidates sampled per slot in --funnel mode (default {DEFAULT_ATTEMPTS})",
    )
    parser.add_argument(
        "--one-step-only",
        action="store_true",
        help="restrict the --funnel pool to single-square pushes; drops the two-step moves so a "
        "single constant label stays truthful. Intended with --single-task for the first "
        "fine-tune.",
    )
    parser.add_argument(
        "--single-task",
        action="store_true",
        help="stamp one constant instruction "
        f'("{SINGLE_PAWN_INSTRUCTION}") on every --funnel episode instead of the '
        "distance-specific label, so the policy learns the motor task without conditioning on "
        "language.",
    )
    parser.add_argument(
        "--tilt-jitter",
        type=float,
        default=SampleKnobs.tilt_jitter_deg,
        help="degrees of grasp lean sampled above the minimum in --funnel mode",
    )
    parser.add_argument(
        "--place-offset",
        type=float,
        default=SampleKnobs.place_offset_radius,
        help="metres the placement foot may be offset within the square in --funnel mode",
    )
    parser.add_argument(
        "--diverse",
        action="store_true",
        help="diversify the --funnel slices instead of repeating near-identical baselines: each "
        "slot keeps a random accepted candidate (not just the clean baseline) and draws its own "
        "board pose from the Step 4 envelope. Turns extra --seeds into real trajectory diversity. "
        "Board jitter magnitudes default to the full envelope; tune with --board-xy-jitter and "
        "--board-yaw-jitter.",
    )
    parser.add_argument(
        "--board-xy-jitter",
        type=float,
        default=STEP4_XY_JITTER,
        help=f"with --diverse, largest per-slot board-centre offset per axis in metres "
        f"(default {STEP4_XY_JITTER}); 0 pins the board centre",
    )
    parser.add_argument(
        "--board-yaw-jitter",
        type=float,
        default=STEP4_YAW_JITTER_DEG,
        help=f"with --diverse, largest per-slot board-yaw offset in degrees "
        f"(default {STEP4_YAW_JITTER_DEG}); 0 pins the board yaw",
    )
    parser.add_argument("--master-seed", type=int, default=0)
    parser.add_argument("--repo-id", default=None, help="LeRobot repo id (writer default if unset)")
    parser.add_argument(
        "--push",
        action="store_true",
        help="after serializing, push the dataset to the Hub repo named by --repo-id, tagged with "
        "the v3.0 codebase version so training can resolve it. Requires --repo-id and HF_TOKEN.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="with --push, create the Hub repo public instead of private (default private).",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument(
        "--single-pawn",
        action="store_true",
        help="record a lone black pawn pushed one square forward on an otherwise empty board, "
        f'labelled "{SINGLE_PAWN_INSTRUCTION}"',
    )
    parser.add_argument(
        "--no-serialize",
        action="store_true",
        help="only generate captures, do not write the dataset",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate a batch of episodes and serialize them into a LeRobot dataset.

    Args:
        argv: Command line without the program name; `sys.argv` is read when omitted.

    Returns:
        Process exit status: 0 on success.
    """
    args = parse_args(argv)
    if args.push and args.repo_id is None:
        print("--push requires --repo-id (the Hub repo to push to)", file=sys.stderr, flush=True)
        return 2
    captures_root = (
        args.captures
        if args.captures is not None
        else args.out.with_name(args.out.name + "-captures")
    )
    captures_root.mkdir(parents=True, exist_ok=True)

    if args.funnel:
        knobs = SampleKnobs(
            attempts=args.attempts,
            tilt_jitter_deg=args.tilt_jitter,
            place_offset_radius=args.place_offset,
        )
        moves = one_step_moves(EXP2_TRAINED_MOVES) if args.one_step_only else EXP2_TRAINED_MOVES
        instruction = SINGLE_PAWN_INSTRUCTION if args.single_task else None
        selection = CandidateSelection.RANDOM if args.diverse else CandidateSelection.FIRST
        manifest = generate_funnel(
            moves,
            args.seeds,
            knobs,
            captures_root,
            args.master_seed,
            args.fps,
            instruction=instruction,
            selection=selection,
            board_xy_jitter=args.board_xy_jitter if args.diverse else 0.0,
            board_yaw_jitter_deg=args.board_yaw_jitter if args.diverse else 0.0,
        )
        print(
            f"funnel kept {manifest['kept']}/{manifest['slots']} slots; "
            f"reasons {manifest['reason_totals']}",
            flush=True,
        )
        if args.no_serialize:
            print("skipping serialization (--no-serialize)", flush=True)
            return 0
        if not any(captures_root.glob("ep-*")):
            print(f"no episodes were kept under {captures_root}; nothing to serialize", flush=True)
            return 1
        stats = serialize(captures_root, args.out, args.repo_id)
        print(
            f"wrote {stats.num_episodes} episodes, {stats.num_frames} frames to {args.out}",
            flush=True,
        )
        if args.push:
            assert args.repo_id is not None  # guarded in main() before any work
            push(args.out, args.repo_id, private=not args.public)
        return 0

    pieces_for: Callable[[MoveSpec], Sequence[PiecePlacement] | None] | None = None
    if args.move_pool:
        specs = build_move_pool_specs(seeds=args.seeds)

        def pieces_for(spec: MoveSpec) -> Sequence[PiecePlacement]:
            return single_pawn_position(spec.source)
    elif args.single_pawn:
        specs = build_specs(args.episodes, instruction=SINGLE_PAWN_INSTRUCTION)

        def pieces_for(spec: MoveSpec) -> Sequence[PiecePlacement]:
            return single_pawn_position(spec.source)
    else:
        specs = build_specs(args.episodes)
    episodes = len(specs)
    remaining = pending_indices(captures_root, episodes)
    print(
        f"{episodes} episodes requested, {episodes - len(remaining)} already captured, "
        f"{len(remaining)} to generate; captures under {captures_root}",
        flush=True,
    )
    generate(specs, captures_root, args.master_seed, args.fps, pieces_for)

    if args.no_serialize:
        print("skipping serialization (--no-serialize)", flush=True)
        return 0

    if not any(captures_root.glob("ep-*")):
        print(f"no episodes were captured under {captures_root}; nothing to serialize", flush=True)
        return 1
    stats = serialize(captures_root, args.out, args.repo_id)
    print(
        f"wrote {stats.num_episodes} episodes, {stats.num_frames} frames to {args.out}",
        flush=True,
    )
    if args.push:
        assert args.repo_id is not None  # guarded in main() before any work
        push(args.out, args.repo_id, private=not args.public)
    return 0


if __name__ == "__main__":
    sys.exit(main())
