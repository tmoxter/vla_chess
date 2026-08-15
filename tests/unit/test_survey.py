import mujoco
import numpy as np
import pytest

from vla_chess.scene import (
    BIRDS_EYE_ARM_QPOS,
    DEFAULT_BOARD_POSE,
    RESTING_ARM_QPOS,
    SceneDescription,
    SceneIndex,
    build_scene,
    make_data,
)
from vla_chess.survey import framing, level_rolls

Scene = tuple[mujoco.MjModel, mujoco.MjData, SceneIndex]


@pytest.fixture(scope="module")
def scene() -> Scene:
    model, index = build_scene(SceneDescription(pieces=()))
    return model, make_data(model, index), index


class TestLevelRolls:
    def test_both_roots_level_the_horizon(self, scene: Scene) -> None:
        model, data, index = scene
        qpos = np.array(BIRDS_EYE_ARM_QPOS, dtype=float)
        rolls = level_rolls(model, data, index, qpos)
        assert len(rolls) == 2
        for roll in rolls:
            view = framing(
                model, data, index, index.board, DEFAULT_BOARD_POSE, np.append(qpos[:4], roll)
            )
            assert view.horizon_tilt_deg == pytest.approx(0.0, abs=1e-6)

    def test_the_two_roots_are_distinct(self, scene: Scene) -> None:
        model, data, index = scene
        rolls = level_rolls(model, data, index, np.array(BIRDS_EYE_ARM_QPOS, dtype=float))
        assert abs(rolls[0] - rolls[1]) > 0.1

    def test_the_roots_disagree_on_where_the_camera_points(self, scene: Scene) -> None:
        """The reason both have to be tried: one of them looks away from the board."""
        model, data, index = scene
        qpos = np.array(BIRDS_EYE_ARM_QPOS, dtype=float)
        pitches = [
            framing(
                model, data, index, index.board, DEFAULT_BOARD_POSE, np.append(qpos[:4], roll)
            ).pitch_deg
            for roll in level_rolls(model, data, index, qpos)
        ]
        assert max(pitches) > 0.0 > min(pitches)


class TestFraming:
    def test_the_survey_pose_frames_the_whole_board(self, scene: Scene) -> None:
        model, data, index = scene
        view = framing(
            model,
            data,
            index,
            index.board,
            DEFAULT_BOARD_POSE,
            np.array(BIRDS_EYE_ARM_QPOS, dtype=float),
        )
        assert view.frames_whole_board()
        assert view.out_of_frame == ()

    def test_the_survey_pose_keeps_the_recorded_margin_and_visibility(self, scene: Scene) -> None:
        """The figures `scene.py` records for this pose, pinned so they cannot drift."""
        model, data, index = scene
        view = framing(
            model,
            data,
            index,
            index.board,
            DEFAULT_BOARD_POSE,
            np.array(BIRDS_EYE_ARM_QPOS, dtype=float),
            rays=True,
        )
        assert view.worst_margin == pytest.approx(0.062, abs=0.002)
        assert view.unoccluded == 60
        assert view.pitch_deg == pytest.approx(83.1, abs=0.2)

    def test_the_resting_pose_keeps_its_recorded_framing(self, scene: Scene) -> None:
        model, data, index = scene
        view = framing(
            model,
            data,
            index,
            index.board,
            DEFAULT_BOARD_POSE,
            np.array(RESTING_ARM_QPOS, dtype=float),
            rays=True,
        )
        assert 64 - len(view.out_of_frame) == 46
        assert view.unoccluded == 27

    def test_occlusion_is_not_counted_for_squares_outside_the_frame(self, scene: Scene) -> None:
        """A square the camera never sees is not "unoccluded"; the two counts compose."""
        model, data, index = scene
        view = framing(
            model,
            data,
            index,
            index.board,
            DEFAULT_BOARD_POSE,
            np.array(RESTING_ARM_QPOS, dtype=float),
            rays=True,
        )
        assert view.unoccluded <= 64 - len(view.out_of_frame)

    def test_rays_off_reports_no_occlusion(self, scene: Scene) -> None:
        model, data, index = scene
        view = framing(
            model,
            data,
            index,
            index.board,
            DEFAULT_BOARD_POSE,
            np.array(BIRDS_EYE_ARM_QPOS, dtype=float),
            rays=False,
        )
        assert view.occluded == ()
