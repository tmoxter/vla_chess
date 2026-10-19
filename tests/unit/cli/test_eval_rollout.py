"""Pure-function parsing helpers for the closed-loop rollout eval CLI.

Only the argument helpers are exercised here; driving a policy needs the `eval` env and a GPU-sized
checkpoint, which the unit suite does not carry.
"""

from pathlib import Path

import pytest

from vla_chess.cli.eval_rollout import _artifact_path, parse_moves


def test_parse_moves_reads_source_target_pairs_in_order() -> None:
    assert parse_moves("g7:g6,d6:d5") == [("g7", "g6"), ("d6", "d5")]


def test_parse_moves_tolerates_whitespace_and_trailing_commas() -> None:
    assert parse_moves(" c7:c6 , d7:d6 ,") == [("c7", "c6"), ("d7", "d6")]


@pytest.mark.parametrize("bad", ["g7", "g7:", ":g6", "", "g7-g6"])
def test_parse_moves_rejects_malformed_items(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_moves(bad)


def test_artifact_path_suffixes_the_move_so_batch_moves_do_not_collide() -> None:
    out = Path("runpod/artifacts/rollout.mp4")
    assert _artifact_path(out, "g7", "g6") == Path("runpod/artifacts/rollout-g7g6.mp4")
    assert _artifact_path(out, "d6", "d5") == Path("runpod/artifacts/rollout-d6d5.mp4")
