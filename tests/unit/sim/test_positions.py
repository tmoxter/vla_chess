from vla_chess.sim.positions import single_pawn_position, starting_position
from vla_chess.sim.scene import PieceColor, PieceType


def test_starting_position_has_all_thirty_two_pieces() -> None:
    assert len(starting_position()) == 32


def test_single_pawn_position_is_one_black_pawn_on_the_named_square() -> None:
    (placement,) = single_pawn_position("d7")
    assert placement.square == "d7"
    assert placement.piece_type is PieceType.PAWN
    assert placement.color is PieceColor.BLACK


def test_single_pawn_position_honours_the_colour_argument() -> None:
    (placement,) = single_pawn_position("e2", PieceColor.WHITE)
    assert placement.color is PieceColor.WHITE
