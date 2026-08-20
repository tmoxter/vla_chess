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
    """The 32-piece initial chess position."""
    placements = []
    for index, file in enumerate(FILES):
        placements.append(PiecePlacement(f"{file}8", BACK_RANK[index], PieceColor.BLACK))
        placements.append(PiecePlacement(f"{file}7", PieceType.PAWN, PieceColor.BLACK))
        placements.append(PiecePlacement(f"{file}1", BACK_RANK[index], PieceColor.WHITE))
        placements.append(PiecePlacement(f"{file}2", PieceType.PAWN, PieceColor.WHITE))
    return tuple(placements)
