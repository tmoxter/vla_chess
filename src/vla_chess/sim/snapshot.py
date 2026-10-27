"""Exact save and restore of a simulated state, so a run can be resumed from any instant."""

from __future__ import annotations

from typing import Final

import mujoco
import numpy as np
import numpy.typing as npt

# Everything that determines how the simulation continues: time, positions, velocities, actuator
# activations, the solver's warmstart and the actuator commands. A state restored from this into a
# fresh `MjData` of the same model steps bit-identically to the one it was taken from.
STATE_SPEC: Final = int(mujoco.mjtState.mjSTATE_FULLPHYSICS | mujoco.mjtState.mjSTATE_CTRL)


def state_size(model: mujoco.MjModel) -> int:
    """Count the numbers one snapshot of this model holds.

    Args:
        model: Scene model the snapshots are taken in.

    Returns:
        Length of the vector `capture_state` returns.
    """
    return int(mujoco.mj_stateSize(model, STATE_SPEC))


def capture_state(model: mujoco.MjModel, data: mujoco.MjData) -> npt.NDArray[np.float64]:
    """Take a snapshot of a simulated state.

    Args:
        model: Scene model `data` belongs to.
        data: State to snapshot; left untouched.

    Returns:
        The snapshot, laid out as `STATE_SPEC` describes.
    """
    state = np.empty(state_size(model), dtype=np.float64)
    mujoco.mj_getState(model, data, state, STATE_SPEC)
    return state


def restore_state(
    model: mujoco.MjModel, data: mujoco.MjData, state: npt.NDArray[np.float64]
) -> None:
    """Overwrite a simulated state with a snapshot and bring its derived quantities up to date.

    Args:
        model: Scene model the snapshot was taken in.
        data: State to overwrite.
        state: A snapshot from `capture_state` on the same model.

    Raises:
        ValueError: If the snapshot is not the size this model's snapshots are.
    """
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (state_size(model),):
        raise ValueError(
            f"state of shape {state.shape} does not fit this model, which snapshots "
            f"{state_size(model)} numbers"
        )
    mujoco.mj_setState(model, data, state, STATE_SPEC)
    mujoco.mj_forward(model, data)
