from collections.abc import Sequence

from vla_chess.sim.scene import FILES, PieceColor, PiecePlacement, PieceType

BACK_RANK: Sequence[PieceType] = (
    PieceType.ROOK,
    PieceType.KNIGHT,
    PieceType.BISHOP,
    PieceType.QUEEN,
    PieceType.KING,
    PieceType.BISHOP,
    PieceType.KNIGHT,
    PieceType.ROOK,
)


def single_pawn_position(
    square: str, color: PieceColor = PieceColor.BLACK
) -> tuple[PiecePlacement, ...]:
    """Build a board holding one pawn and nothing else.

    The bare board the "move the pawn one field forward" demonstrations are recorded on: with
    no other pieces to name against, the task is the push itself, not a coordinate.

    Args:
        square: Algebraic square the pawn stands on, e.g. "d7".
        color: The pawn's colour; black by default, matching the opening's back ranks.

    Returns:
        A one-element layout, ready to pass as the runner's `pieces`.
    """
    return (PiecePlacement(square, PieceType.PAWN, color),)


def starting_position() -> tuple[PiecePlacement, ...]:
    """Build the initial chess position.

    Returns:
        The 32 placements of the opening position, black on ranks 7 and 8 and white on
        ranks 1 and 2, ordered file by file.
    """
    placements = []
    for index, file in enumerate(FILES):
        placements.append(PiecePlacement(f"{file}8", BACK_RANK[index], PieceColor.BLACK))
        placements.append(PiecePlacement(f"{file}7", PieceType.PAWN, PieceColor.BLACK))
        placements.append(PiecePlacement(f"{file}1", BACK_RANK[index], PieceColor.WHITE))
        placements.append(PiecePlacement(f"{file}2", PieceType.PAWN, PieceColor.WHITE))
    return tuple(placements)
