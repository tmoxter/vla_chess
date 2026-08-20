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
