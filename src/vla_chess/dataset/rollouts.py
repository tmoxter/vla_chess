"""Recorded closed-loop rollouts: the simulated state a driving policy saw at every frame.

A rollout is kept as one snapshot per frame rather than as images, so any instant of it can be
restored exactly and handed to the expert; the move it was driving and how it ended ride along.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from vla_chess.control.planner import MoveSpec
from vla_chess.sim.scene import PieceType


@dataclass(frozen=True)
class RecordedRollout:
    """One policy rollout of a move, as the states it passed through."""

    source: str
    target: str
    instruction: str
    snapshots: npt.NDArray[np.float64]  # (frames, state size): the state each frame observed
    grasped: bool
    placed: bool
    final_square: str | None
    fps: float

    def spec(self) -> MoveSpec:
        """Rebuild the move this rollout was driving.

        Returns:
            A pawn move between its squares, carrying the instruction the policy was given.
        """
        return MoveSpec(
            source=self.source,
            target=self.target,
            piece_type=PieceType.PAWN,
            instruction_override=self.instruction,
        )


def rollout_path(directory: Path, index: int) -> Path:
    """Name the file one rollout index is saved under."""
    return directory / f"rollout-{index:05d}.npz"


def save_rollout(path: Path, rollout: RecordedRollout) -> None:
    """Write one rollout to a single file, atomically.

    Args:
        path: File to write; replaced if it exists.
        rollout: The rollout to persist.
    """
    meta = {
        "source": rollout.source,
        "target": rollout.target,
        "instruction": rollout.instruction,
        "grasped": rollout.grasped,
        "placed": rollout.placed,
        "final_square": rollout.final_square,
        "fps": rollout.fps,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, snapshots=rollout.snapshots, meta=np.array(json.dumps(meta)))
    tmp.replace(path)


def load_rollout(path: Path) -> RecordedRollout:
    """Read back a rollout written by `save_rollout`.

    Args:
        path: The file to read.

    Returns:
        The rollout, its snapshots exact.
    """
    with np.load(path) as arrays:
        snapshots = np.asarray(arrays["snapshots"], dtype=np.float64)
        meta = json.loads(str(arrays["meta"]))
    return RecordedRollout(
        source=meta["source"],
        target=meta["target"],
        instruction=meta["instruction"],
        snapshots=snapshots,
        grasped=meta["grasped"],
        placed=meta["placed"],
        final_square=meta["final_square"],
        fps=meta["fps"],
    )
