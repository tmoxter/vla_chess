"""Record expert takeovers of recorded policy rollouts into a LeRobot dataset, resumably.

The aggregation half of a DAgger round. Each rollout written by `eval_metrics --record-states` is
taken over at random frames by the scripted expert, and every recovery that places the piece is
captured as an episode. A rollout is done once its tally file exists, so an interrupted or sharded
run picks up where it stopped; serializing the captures into a dataset is a separate final pass.
Runs in the `datagen` pixi env (the serializer imports lerobot).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from vla_chess.cli.generate_dataset import _parse_shard, serialize
from vla_chess.dataset.capture import save_capture
from vla_chess.dataset.episode import git_engine_version
from vla_chess.dataset.rollouts import load_rollout
from vla_chess.dataset.takeover import TakeoverRunner, take_over_rollout
from vla_chess.sim.positions import single_pawn_position

MANIFEST_NAME = "takeover-manifest.json"


def _tally_path(captures: Path, rollout: Path) -> Path:
    """Name the file that marks one rollout as taken over."""
    return captures / f"{rollout.stem}.json"


def take_over(
    rollouts: Sequence[Path],
    captures: Path,
    *,
    wanted: int,
    skip: int,
    max_attempts: int,
    fps: float,
    seed: int,
    shard: tuple[int, int],
) -> None:
    """Take over every rollout of this shard that has no tally yet.

    Args:
        rollouts: The rollout files, in a fixed order; rollout `i` belongs to shard `i` mod `n`.
        captures: Directory the kept episodes and per-rollout tallies are written under.
        wanted: Kept takeovers to stop at per rollout.
        skip: Leading frames of a rollout never taken over from.
        max_attempts: Recoverable frames tried at most per rollout.
        fps: Capture frame rate of the kept episodes.
        seed: Base seed; rollout `i` shuffles its frames from `[seed, i]`.
        shard: `(i, n)` — take over only the rollouts whose index is `i` mod `n`.
    """
    engine_version = git_engine_version()
    for i, path in enumerate(rollouts):
        if i % shard[1] != shard[0]:
            continue
        tally_file = _tally_path(captures, path)
        if tally_file.exists():
            continue
        rollout = load_rollout(path)
        runner = TakeoverRunner(
            pieces=single_pawn_position(rollout.source), fps=fps, engine_version=engine_version
        )
        started = time.monotonic()
        tried = take_over_rollout(
            rollout,
            runner,
            wanted=wanted,
            skip=skip,
            max_attempts=max_attempts,
            rng=np.random.default_rng([seed, i]),
            base_seed=seed * 1_000_000 + i * 1_000,
        )
        frames: list[dict[str, object]] = []
        for frame, outcome in tried:
            if outcome.episode is not None:
                save_capture(captures / f"ep-{path.stem}-f{frame:04d}", outcome.episode)
            frames.append(
                {
                    "frame": frame,
                    "situation": outcome.situation.value,
                    "kept": outcome.episode is not None,
                    "attempts": outcome.attempts,
                    "rejects": dict(outcome.rejects),
                }
            )
        tally = {
            "rollout": path.name,
            "move": f"{rollout.source}->{rollout.target}",
            "rollout_placed": rollout.placed,
            "frames": frames,
        }
        tally_file.write_text(json.dumps(tally, indent=2) + "\n")
        kept = sum(outcome.episode is not None for _, outcome in tried)
        print(
            f"[{i + 1}/{len(rollouts)}] {path.name} {tally['move']}: kept {kept} of "
            f"{len(frames)} frames tried in {time.monotonic() - started:.0f} s",
            flush=True,
        )


def summarize(captures: Path) -> dict[str, object]:
    """Total the per-rollout tallies under a captures directory.

    Args:
        captures: Directory `take_over` wrote its tallies under.

    Returns:
        Rollouts covered, frames tried and kept, and the count of each situation met.
    """
    situations: Counter[str] = Counter()
    kept_by_situation: Counter[str] = Counter()
    rejects: Counter[str] = Counter()
    rollouts = tried = kept = 0
    for path in sorted(captures.glob("rollout-*.json")):
        tally = json.loads(path.read_text())
        rollouts += 1
        for frame in tally["frames"]:
            tried += 1
            situations[frame["situation"]] += 1
            rejects.update(frame["rejects"])
            if frame["kept"]:
                kept += 1
                kept_by_situation[frame["situation"]] += 1
    return {
        "rollouts": rollouts,
        "frames_tried": tried,
        "kept": kept,
        "situations": dict(situations),
        "kept_by_situation": dict(kept_by_situation),
        "rejects": dict(rejects),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the takeover generator's command line.

    Args:
        argv: Command line without the program name; `sys.argv` is read when omitted.

    Returns:
        The parsed options.
    """
    parser = argparse.ArgumentParser(prog="dagger-takeovers", description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True, help="recorded rollouts dir")
    parser.add_argument("--captures", type=Path, required=True, help="where takeovers go")
    parser.add_argument("--out", type=Path, default=None, help="dataset to serialize into")
    parser.add_argument("--repo-id", default=None, help="LeRobot repo id stamped into --out")
    parser.add_argument("--per-rollout", type=int, default=3, help="kept takeovers per rollout")
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=15,
        help="leading frames of a rollout never taken over from (half a second at 30 fps)",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=8, help="recoverable frames tried per rollout"
    )
    parser.add_argument("--fps", type=float, default=30.0, help="capture rate of kept episodes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", default="0/1", help="'i/n': only rollouts with index i mod n")
    parser.add_argument(
        "--no-serialize", action="store_true", help="only take over, do not write the dataset"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Take over recorded rollouts and serialize the kept recoveries.

    Args:
        argv: Command line without the program name; `sys.argv` is read when omitted.

    Returns:
        Process exit status: 0 on success, 1 when there is nothing to serialize.
    """
    args = parse_args(argv)
    rollouts = sorted(args.rollouts.glob("rollout-*.npz"))
    if not rollouts:
        print(f"no rollouts under {args.rollouts}", file=sys.stderr, flush=True)
        return 1
    args.captures.mkdir(parents=True, exist_ok=True)
    take_over(
        rollouts,
        args.captures,
        wanted=args.per_rollout,
        skip=args.skip_frames,
        max_attempts=args.max_attempts,
        fps=args.fps,
        seed=args.seed,
        shard=_parse_shard(args.shard),
    )
    if args.no_serialize:
        return 0
    manifest = summarize(args.captures)
    (args.captures / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"takeovers: {manifest}", flush=True)
    if args.out is None:
        return 0
    if not any(args.captures.glob("ep-*")):
        print(f"no takeovers were kept under {args.captures}", file=sys.stderr, flush=True)
        return 1
    stats = serialize(args.captures, args.out, args.repo_id)
    print(f"wrote {stats.num_episodes} episodes, {stats.num_frames} frames to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
