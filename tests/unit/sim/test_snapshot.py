import mujoco
import numpy as np
import pytest

from vla_chess.sim.positions import single_pawn_position
from vla_chess.sim.scene import (
    BIRDS_EYE_ARM_QPOS,
    DEFAULT_BOARD_POSE,
    SceneDescription,
    build_scene,
    make_data,
)
from vla_chess.sim.snapshot import capture_state, restore_state, state_size


def _scene() -> tuple[mujoco.MjModel, mujoco.MjData, object]:
    description = SceneDescription(
        board_pose=DEFAULT_BOARD_POSE,
        pieces=single_pawn_position("d7"),
        arm_qpos=BIRDS_EYE_ARM_QPOS,
    )
    model, index = build_scene(description)
    return model, make_data(model, index), index


def _drive(model: mujoco.MjModel, data: mujoco.MjData, steps: int) -> None:
    data.ctrl[:] = np.linspace(-0.2, 0.2, model.nu)
    for _ in range(steps):
        mujoco.mj_step(model, data)


def test_a_snapshot_has_the_size_the_model_reports() -> None:
    model, data, _ = _scene()
    assert capture_state(model, data).shape == (state_size(model),)


def test_restoring_a_snapshot_reproduces_the_state_and_its_future() -> None:
    model, data, index = _scene()
    _drive(model, data, 50)
    snapshot = capture_state(model, data)
    _drive(model, data, 100)

    restored = make_data(model, index)
    restore_state(model, restored, snapshot)
    np.testing.assert_array_equal(restored.ctrl, np.linspace(-0.2, 0.2, model.nu))
    _drive(model, restored, 100)
    np.testing.assert_array_equal(restored.qpos, data.qpos)
    np.testing.assert_array_equal(restored.qvel, data.qvel)


def test_restoring_runs_forward_kinematics() -> None:
    model, data, index = _scene()
    _drive(model, data, 50)
    snapshot = capture_state(model, data)

    restored = make_data(model, index)
    restore_state(model, restored, snapshot)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(restored.xpos, data.xpos)


def test_a_snapshot_of_another_size_is_refused() -> None:
    model, data, _ = _scene()
    with pytest.raises(ValueError, match="state"):
        restore_state(model, data, np.zeros(state_size(model) + 1))
