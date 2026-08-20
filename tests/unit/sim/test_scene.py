import mujoco
import numpy as np
import pytest

from vla_chess.grasp.solver import GRASP_HEIGHT
from vla_chess.sim.scene import (
    APPROACH_MARGIN,
    DEFAULT_GRIP_RADIUS,
    DEFAULT_PINCH_OFFSET,
    FIXED_FACE_INSET,
    FRICTION_IMPEDANCE_RATIO,
    GRIP_SQUEEZE,
    Board,
    BoardPose,
    PieceColor,
    PiecePlacement,
    PieceType,
    SceneDescription,
    build_scene,
    denormalize_gripper,
    grip_command,
    grip_radius,
    jaw_gap,
    normalize_gripper,
    pinch_offset_for,
    ready_command,
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


class TestGripRadius:
    """What the jaws close on is the widest thing between the fingers, not the stem."""

    def radii(self) -> dict[PieceType, float]:
        return {piece_type: grip_radius(piece_type, GRASP_HEIGHT) for piece_type in PieceType}

    def test_every_piece_type_offers_the_fingers_something_at_grasp_height(self) -> None:
        assert set(self.radii()) == set(PieceType)
        assert all(radius > 0.0 for radius in self.radii().values())

    # The pawn is the case that taught this: its head is 1 mm wider than the stem the TCP sits
    # beside, stands inside the fingers' own band, and so is what the faces actually bear on.
    def test_a_pawn_is_gripped_by_its_head_not_its_stem(self) -> None:
        assert grip_radius(PieceType.PAWN, GRASP_HEIGHT) == pytest.approx(0.010)

    # The default is what a scene gets when the caller names no piece, so nothing in the set
    # may be wider than it.
    def test_the_default_grip_radius_covers_the_piece_set(self) -> None:
        assert max(self.radii().values()) <= DEFAULT_GRIP_RADIUS + 1e-9


class TestPinchOffset:
    """The pinch site is the middle of the jaws when they are closed on the piece."""

    def test_a_piece_ends_up_centred_between_the_faces(self) -> None:
        for piece_type in PieceType:
            radius = grip_radius(piece_type, GRASP_HEIGHT)
            offset = pinch_offset_for(radius)
            # The fixed face lands a squeeze inside the piece's own surface, and the jaw closes
            # to a gap of the same depth on the other side, so the two are symmetric about the
            # site rather than one of them holding the piece against the other.
            face_to_axis = offset - FIXED_FACE_INSET
            assert radius - face_to_axis == pytest.approx(GRIP_SQUEEZE / 2.0)

    def test_the_default_offset_follows_the_same_rule(self) -> None:
        assert pinch_offset_for(DEFAULT_GRIP_RADIUS) == pytest.approx(DEFAULT_PINCH_OFFSET)


class TestJawCommands:
    """What the hand is told to do with the piece between its fingers."""

    RANGE = (-0.17453297762778586, 1.7453291995659765)
    STANDOFF = 0.005

    def commands(self, piece_type: PieceType) -> tuple[float, float]:
        radius = grip_radius(piece_type, GRASP_HEIGHT)
        return (
            grip_command(radius, self.RANGE),
            ready_command(radius, self.STANDOFF, self.RANGE),
        )

    def test_every_piece_can_be_gripped_and_approached_within_the_joint(self) -> None:
        for piece_type in PieceType:
            for command in self.commands(piece_type):
                assert 0.0 < command < 100.0, piece_type.value

    def test_the_approach_opening_is_wider_than_the_grip(self) -> None:
        for piece_type in PieceType:
            grip, ready = self.commands(piece_type)
            assert ready > grip, piece_type.value

    # Opening fully swings the jaw 100 mm past the pinch, through whatever stands on the next
    # square; the approach opening has to stay far short of it. See docs/trajectory.md.
    def test_the_approach_opening_stays_well_short_of_the_stop(self) -> None:
        for piece_type in PieceType:
            assert self.commands(piece_type)[1] < 50.0, piece_type.value

    def test_the_approach_opening_admits_the_piece_and_its_standoff(self) -> None:
        for piece_type in PieceType:
            radius = grip_radius(piece_type, GRASP_HEIGHT)
            _, ready = self.commands(piece_type)
            gap = jaw_gap(denormalize_gripper(ready, self.RANGE))
            assert gap == pytest.approx(2 * radius + self.STANDOFF + APPROACH_MARGIN)

    def test_a_wider_piece_is_gripped_with_the_jaws_further_apart(self) -> None:
        pawn = grip_command(grip_radius(PieceType.PAWN, GRASP_HEIGHT), self.RANGE)
        king = grip_command(grip_radius(PieceType.KING, GRASP_HEIGHT), self.RANGE)
        assert king > pawn


class TestContactModel:
    """MuJoCo's defaults let a correctly-sized pinch creep, and the piece slides out of it."""

    def test_the_scene_resolves_friction_as_stiffly_as_the_normal_direction(self) -> None:
        model, _ = build_scene(SceneDescription())
        assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
        assert model.opt.impratio == pytest.approx(FRICTION_IMPEDANCE_RATIO)
