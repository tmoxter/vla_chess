from pathlib import Path

import numpy as np

from vla_chess.dataset.rollouts import (
    RecordedRollout,
    load_rollout,
    rollout_path,
    save_rollout,
)
from vla_chess.sim.scene import PieceType


def _rollout() -> RecordedRollout:
    return RecordedRollout(
        source="d7",
        target="d6",
        instruction="move the pawn one field forward",
        snapshots=np.random.default_rng(0).normal(size=(12, 32)),
        grasped=True,
        placed=False,
        final_square=None,
        fps=30.0,
    )


def test_a_saved_rollout_reads_back_exactly(tmp_path: Path) -> None:
    rollout = _rollout()
    path = rollout_path(tmp_path, 7)
    save_rollout(path, rollout)
    loaded = load_rollout(path)
    np.testing.assert_array_equal(loaded.snapshots, rollout.snapshots)
    assert loaded == RecordedRollout(**{**rollout.__dict__, "snapshots": loaded.snapshots})


def test_saving_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    save_rollout(rollout_path(tmp_path, 0), _rollout())
    assert [p.name for p in tmp_path.iterdir()] == ["rollout-00000.npz"]


def test_the_rollout_rebuilds_the_move_it_drove() -> None:
    spec = _rollout().spec()
    assert (spec.source, spec.target, spec.piece_type) == ("d7", "d6", PieceType.PAWN)
    assert spec.instruction() == "move the pawn one field forward"
