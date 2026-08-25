"""Generate a batch of pick-and-place episodes into one LeRobot dataset, resumably.

Two phases. First, every episode is generated and written to its own capture directory; a run
that is interrupted and restarted skips the captures already on disk, so the expensive physics is
never repeated. Second, once the captures exist, they are serialized into a LeRobot dataset in a
single pass that can be rerun at will — it writes to a temporary directory and swaps it into place
only when complete.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final

from vla_chess.control.planner import MoveSpec, PlanInfeasible
from vla_chess.dataset.capture import load_capture, save_capture
from vla_chess.dataset.episode import EpisodeRunner, git_engine_version
from vla_chess.dataset.observe import DEFAULT_FPS
from vla_chess.sim.scene import PieceType

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

DEFAULT_EPISODES: Final = 30
DEFAULT_OUT: Final = Path("data/chess_pick_place")


def build_specs(episodes: int, moves: Sequence[tuple[str, str]] = WORKING_MOVES) -> list[MoveSpec]:
    """Lay out the moves for a batch, cycling through the working squares.

    Args:
        episodes: How many episodes the batch holds.
        moves: The (source, target) squares to cycle through, all pawns.

    Returns:
        One spec per episode, in generation order.
    """
    return [
        MoveSpec(
            source=moves[i % len(moves)][0],
            target=moves[i % len(moves)][1],
            piece_type=PieceType.PAWN,
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


def generate(specs: Sequence[MoveSpec], captures_root: Path, master_seed: int, fps: float) -> None:
    """Generate every missing episode into its capture directory.

    Episodes whose capture already exists are skipped, so an interrupted run resumes where it
    stopped. A spec that cannot be planned is logged and left uncaptured rather than aborting
    the batch.

    Args:
        specs: The moves to generate, in order; the seed of episode i is `master_seed + i`.
        captures_root: Directory the per-episode captures are written under.
        master_seed: Seed of the first episode.
        fps: Capture frame rate.
    """
    runner = EpisodeRunner(fps=fps, engine_version=git_engine_version())
    total = len(specs)
    for index, spec in enumerate(specs):
        target = capture_dir(captures_root, index)
        head = f"[{index + 1}/{total}] {spec.source}->{spec.target}"
        if target.exists():
            print(f"{head} already captured, skipping", flush=True)
            continue
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
    parser.add_argument("--master-seed", type=int, default=0)
    parser.add_argument("--repo-id", default=None, help="LeRobot repo id (writer default if unset)")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
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
    captures_root = (
        args.captures
        if args.captures is not None
        else args.out.with_name(args.out.name + "-captures")
    )
    captures_root.mkdir(parents=True, exist_ok=True)

    specs = build_specs(args.episodes)
    remaining = pending_indices(captures_root, args.episodes)
    print(
        f"{args.episodes} episodes requested, {args.episodes - len(remaining)} already captured, "
        f"{len(remaining)} to generate; captures under {captures_root}",
        flush=True,
    )
    generate(specs, captures_root, args.master_seed, args.fps)

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
