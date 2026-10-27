import argparse

import pytest

from vla_chess.cli.eval_metrics import sweep_moves
from vla_chess.cli.generate_dataset import EXP2_TRAINED_MOVES, one_step_moves


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "moves": None,
        "trained_one_step": False,
        "source": None,
        "target": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_a_single_move_is_named_by_its_squares() -> None:
    assert sweep_moves(_args(source="d7", target="d6")) == [("d7", "d6")]


def test_a_move_list_is_parsed_in_order() -> None:
    assert sweep_moves(_args(moves="c7:c6,d7:d6")) == [("c7", "c6"), ("d7", "d6")]


def test_the_trained_one_step_pool_is_every_funnel_push_of_one_square() -> None:
    moves = sweep_moves(_args(trained_one_step=True))
    assert moves == one_step_moves(EXP2_TRAINED_MOVES)
    assert len(moves) == 20


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"moves": "c7:c6", "source": "d7", "target": "d6"},
        {"moves": "c7:c6", "trained_one_step": True},
        {"source": "d7"},
    ],
)
def test_the_moves_must_be_named_exactly_one_way(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        sweep_moves(_args(**overrides))
