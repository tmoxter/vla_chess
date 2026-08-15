import numpy as np
import pytest

from vla_chess.grasp import GRASP_HEIGHT
from vla_chess.scene import (
    DEFAULT_PINCH_OFFSET,
    PIECE_GEOMS,
    PINCH_CLEARANCE,
    WIDEST_STEM_RADIUS,
    Board,
    BoardPose,
    PieceColor,
    PiecePlacement,
    PieceType,
    SceneDescription,
    denormalize_gripper,
    normalize_gripper,
    stem_radius,
)

RNG = np.random.default_rng(0)


def random_poses(n: int) -> list[BoardPose]:
    """`n` board poses with position within a plausible table region and yaw in [-pi, pi]."""
    poses = []
    for _ in range(n):
        x = float(RNG.uniform(0.05, 0.5))
        y = float(RNG.uniform(-0.3, 0.3))
        yaw = float(RNG.uniform(-np.pi, np.pi))
        pitch = float(RNG.uniform(0.02, 0.06))
        poses.append(BoardPose(x, y, yaw, pitch))
    return poses


class TestBoardPose:
    def test_rejects_nonpositive_pitch(self) -> None:
        with pytest.raises(ValueError, match="square_pitch"):
            BoardPose(0.0, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError, match="square_pitch"):
            BoardPose(0.0, 0.0, 0.0, -0.01)


class TestBoardSquares:
    def test_lists_all_64_squares(self) -> None:
        board = Board()
        squares = board.squares()
        assert len(squares) == 64
        assert len(set(squares)) == 64
        assert "a1" in squares
        assert "h8" in squares

    def test_rejects_invalid_square_name(self) -> None:
        board = Board()
        pose = BoardPose(0.3, 0.0, 0.0, 0.05)
        with pytest.raises(ValueError, match="not a valid square"):
            board.square_to_world("i1", pose)
        with pytest.raises(ValueError, match="not a valid square"):
            board.square_to_world("a9", pose)
        with pytest.raises(ValueError, match="not a valid square"):
            board.square_to_world("a", pose)


class TestRoundTrip:
    @pytest.mark.parametrize("pose", random_poses(20))
    def test_world_to_square_inverts_square_to_world(self, pose: BoardPose) -> None:
        board = Board()
        for square in board.squares():
            world = board.square_to_world(square, pose)
            assert board.world_to_square(world, pose) == square


class TestBoundaries:
    def test_small_offset_stays_on_square(self) -> None:
        board = Board()
        pose = BoardPose(0.3, 0.0, 0.0, 0.05)
        centre = board.square_to_world("d4", pose)
        offset = centre.copy()
        offset[0] += 0.45 * pose.square_pitch
        assert board.world_to_square(offset, pose) == "d4"

    def test_large_offset_moves_to_adjacent_square(self) -> None:
        board = Board()
        pose = BoardPose(0.3, 0.0, 0.0, 0.05)
        centre = board.square_to_world("d4", pose)
        offset = centre.copy()
        offset[0] += 0.55 * pose.square_pitch
        assert board.world_to_square(offset, pose) == "e4"

    def test_far_off_board_is_none(self) -> None:
        board = Board()
        pose = BoardPose(0.3, 0.0, 0.0, 0.05)
        centre = board.square_to_world("a1", pose)
        far = centre.copy()
        far[0] -= 3 * pose.square_pitch
        assert board.world_to_square(far, pose) is None


class TestSquareToWorld:
    def test_spacing_matches_configured_pitch(self) -> None:
        board = Board()
        pose = BoardPose(0.3, 0.0, 0.0, 0.05)
        a1 = board.square_to_world("a1", pose)
        b1 = board.square_to_world("b1", pose)
        a2 = board.square_to_world("a2", pose)
        assert np.linalg.norm(b1 - a1) == pytest.approx(pose.square_pitch, abs=1e-9)
        assert np.linalg.norm(a2 - a1) == pytest.approx(pose.square_pitch, abs=1e-9)

    def test_z_is_the_board_surface_height(self) -> None:
        board = Board(surface_height=0.123)
        pose = BoardPose(0.3, 0.0, 0.4, 0.05)
        for square in ("a1", "h8", "d4"):
            assert board.square_to_world(square, pose)[2] == pytest.approx(0.123)


class TestGripperNormalization:
    def test_round_trips(self) -> None:
        joint_range = (-0.1745, 1.7453)
        for value in (0.0, 12.5, 50.0, 87.0, 100.0):
            qpos = denormalize_gripper(value, joint_range)
            assert normalize_gripper(qpos, joint_range) == pytest.approx(value)

    def test_endpoints(self) -> None:
        joint_range = (-0.1745, 1.7453)
        assert normalize_gripper(joint_range[0], joint_range) == pytest.approx(0.0)
        assert normalize_gripper(joint_range[1], joint_range) == pytest.approx(100.0)


class TestSceneDescription:
    def test_rejects_two_pieces_on_the_same_square(self) -> None:
        with pytest.raises(ValueError, match="e4"):
            SceneDescription(
                pieces=(
                    PiecePlacement("e4", PieceType.PAWN, PieceColor.WHITE),
                    PiecePlacement("e4", PieceType.ROOK, PieceColor.BLACK),
                )
            )

    def test_accepts_disjoint_squares(self) -> None:
        desc = SceneDescription(
            pieces=(
                PiecePlacement("e4", PieceType.PAWN, PieceColor.WHITE),
                PiecePlacement("e5", PieceType.PAWN, PieceColor.BLACK),
            )
        )
        assert len(desc.pieces) == 2


class TestPinchOffset:
    """The pinch site has to clear every stem it will be asked to close on."""

    def stems(self) -> dict[PieceType, float]:
        """Radius of the geom each piece type presents at grasp height."""
        radii = {}
        for piece_type, geoms in PIECE_GEOMS.items():
            spanning = [
                geom
                for geom in geoms
                if geom.shape == "cylinder"
                and geom.pos[2] - geom.size[1] <= GRASP_HEIGHT <= geom.pos[2] + geom.size[1]
            ]
            assert len(spanning) == 1, f"{piece_type} is not a single cylinder at grasp height"
            radii[piece_type] = spanning[0].size[0]
        return radii

    def test_every_piece_type_presents_a_stem_at_grasp_height(self) -> None:
        assert set(self.stems()) == set(PieceType)

    def test_the_widest_stem_constant_matches_the_piece_set(self) -> None:
        assert max(self.stems().values()) == pytest.approx(WIDEST_STEM_RADIUS)

    # The default has to clear every stem, since it is what a scene gets when the caller does
    # not say which piece it intends to lift.
    def test_the_default_offset_clears_every_stem(self) -> None:
        for piece_type, radius in self.stems().items():
            assert radius < DEFAULT_PINCH_OFFSET, f"{piece_type.value} stem reaches the pinch site"

    def test_the_default_offset_leaves_the_documented_clearance(self) -> None:
        assert pytest.approx(PINCH_CLEARANCE) == DEFAULT_PINCH_OFFSET - WIDEST_STEM_RADIUS

    # What the per-piece offset buys: the shove a closing jaw gives the piece is the gap it has
    # to cross, so sizing the site to the piece caps that at the clearance. Sized once to the
    # widest stem it was 4.5 mm for a pawn, which toppled it. See scene.PINCH_CLEARANCE.
    def test_a_per_piece_offset_caps_the_shove_at_the_clearance(self) -> None:
        for piece_type, radius in self.stems().items():
            offset = stem_radius(piece_type, GRASP_HEIGHT) + PINCH_CLEARANCE
            assert offset - radius == pytest.approx(PINCH_CLEARANCE)
