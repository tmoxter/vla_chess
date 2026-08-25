"""Persist a recorded episode to disk as a resumable, regenerable checkpoint.

A capture holds everything an `Episode` carries — the expensive rendered frames above all — so
generation can be stopped and resumed without re-running the physics. It is written atomically:
the frames land in a sibling `.tmp` directory that is renamed into place only once complete, so a
capture directory that exists is always whole. The LeRobot dataset is a separate, cheap
serialization built from these; the captures are the durable record, not the dataset.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from vla_chess.control.executor import AbortReason, ExecutionDiagnostics, ExecutionResult
from vla_chess.control.planner import MoveSpec, Phase
from vla_chess.dataset.episode import Episode
from vla_chess.dataset.observe import Frame
from vla_chess.sim.scene import PieceType

_ARRAYS = "arrays.npz"
_META = "episode.json"


def save_capture(directory: Path, ep: Episode) -> None:
    """Write one episode to its own directory, atomically.

    Args:
        directory: Where the capture goes. It is replaced if it already exists, and a stale
            `<name>.tmp` sibling from an interrupted write is discarded first.
        ep: The episode to persist. It must carry at least one frame.

    Raises:
        ValueError: If the episode has no frames.
    """
    if not ep.frames:
        raise ValueError("cannot capture an episode with no frames")
    directory = Path(directory)
    tmp = directory.with_name(directory.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    camera_names = sorted(ep.frames[0].images)
    arrays: dict[str, Any] = {
        f"image__{name}": np.stack([frame.images[name] for frame in ep.frames])
        for name in camera_names
    }
    arrays["frame_states"] = np.stack([frame.state for frame in ep.frames])
    arrays["frame_actions"] = np.stack([frame.action for frame in ep.frames])
    arrays["frame_timestamps"] = np.array([frame.timestamp for frame in ep.frames])
    arrays["frame_indices"] = np.array([frame.frame_index for frame in ep.frames])
    arrays["frame_control_steps"] = np.array([frame.control_step for frame in ep.frames])
    arrays["frame_phases"] = np.array([frame.phase.value for frame in ep.frames])
    arrays["exec_states"] = ep.execution.states
    arrays["exec_actions"] = ep.execution.actions
    arrays["exec_phases"] = np.array([phase.value for phase in ep.execution.phases])
    np.savez_compressed(tmp / _ARRAYS, **arrays)

    result = ep.execution
    meta = {
        "source": ep.spec.source,
        "target": ep.spec.target,
        "piece_type": ep.spec.piece_type.value,
        "instruction": ep.instruction,
        "engine_version": ep.engine_version,
        "fps": ep.fps,
        "camera_names": camera_names,
        "grasped": result.grasped,
        "placed": result.placed,
        "final_square": result.final_square,
        "abort_reason": result.abort_reason.value if result.abort_reason else None,
        "phase_boundaries": {
            phase.value: list(span) for phase, span in result.phase_boundaries.items()
        },
        "diagnostics": {
            "max_tracking_error": result.diagnostics.max_tracking_error,
            "max_ik_error": result.diagnostics.max_ik_error,
            "control_steps": result.diagnostics.control_steps,
            "sim_steps": result.diagnostics.sim_steps,
            "wall_seconds": result.diagnostics.wall_seconds,
        },
    }
    (tmp / _META).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    if directory.exists():
        shutil.rmtree(directory)
    tmp.rename(directory)


def load_capture(directory: Path) -> Episode:
    """Reconstruct the episode a capture directory holds.

    Args:
        directory: A directory written by `save_capture`.

    Returns:
        The episode, its frames exact and its execution result rebuilt from the persisted
        states, actions and verdict.
    """
    directory = Path(directory)
    meta = json.loads((directory / _META).read_text())
    with np.load(directory / _ARRAYS) as arrays:
        data = {key: arrays[key] for key in arrays.files}

    camera_names: list[str] = meta["camera_names"]
    n_frames = len(data["frame_states"])
    frames = [
        Frame(
            images={name: data[f"image__{name}"][i] for name in camera_names},
            state=data["frame_states"][i],
            action=data["frame_actions"][i],
            timestamp=float(data["frame_timestamps"][i]),
            frame_index=int(data["frame_indices"][i]),
            control_step=int(data["frame_control_steps"][i]),
            phase=Phase(str(data["frame_phases"][i])),
        )
        for i in range(n_frames)
    ]

    diagnostics = ExecutionDiagnostics(
        max_tracking_error=meta["diagnostics"]["max_tracking_error"],
        max_ik_error=meta["diagnostics"]["max_ik_error"],
        control_steps=meta["diagnostics"]["control_steps"],
        sim_steps=meta["diagnostics"]["sim_steps"],
        wall_seconds=meta["diagnostics"]["wall_seconds"],
    )
    execution = ExecutionResult(
        states=data["exec_states"],
        actions=data["exec_actions"],
        phases=tuple(Phase(str(value)) for value in data["exec_phases"]),
        phase_boundaries={
            Phase(key): (span[0], span[1]) for key, span in meta["phase_boundaries"].items()
        },
        grasped=meta["grasped"],
        placed=meta["placed"],
        final_square=meta["final_square"],
        abort_reason=AbortReason(meta["abort_reason"]) if meta["abort_reason"] else None,
        diagnostics=diagnostics,
    )
    spec = MoveSpec(
        source=meta["source"], target=meta["target"], piece_type=PieceType(meta["piece_type"])
    )
    return Episode(
        spec=spec,
        instruction=meta["instruction"],
        frames=frames,
        execution=execution,
        engine_version=meta["engine_version"],
        fps=meta["fps"],
    )
