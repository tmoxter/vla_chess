from pathlib import Path

import numpy as np
import pytest

from vla_chess.cli.generate_dataset import build_specs, capture_dir, pending_indices
from vla_chess.control.executor import AbortReason, ExecutionDiagnostics, ExecutionResult
from vla_chess.control.planner import MoveSpec, Phase
from vla_chess.dataset.capture import load_capture, save_capture
from vla_chess.dataset.episode import Episode
from vla_chess.dataset.observe import Frame
from vla_chess.sim.scene import PieceType


def make_episode(
    instruction: str = "move the pawn from d7 to d6",
    n_frames: int = 4,
    size: int = 16,
    seed: int = 0,
    abort: AbortReason | None = None,
) -> Episode:
    rng = np.random.default_rng(seed)
    steps = n_frames * 2
    exec_states = rng.normal(size=(steps, 6))
    exec_actions = rng.normal(size=(steps, 6))
    frames = [
        Frame(
            images={"wrist": np.full((size, size, 3), (i * 17) % 255, dtype=np.uint8)},
            state=exec_states[2 * i],
            action=exec_actions[2 * i],
            timestamp=i / 10.0,
            frame_index=i,
            control_step=2 * i,
            phase=Phase.TRANSPORT,
        )
        for i in range(n_frames)
    ]
    execution = ExecutionResult(
        states=exec_states,
        actions=exec_actions,
        phases=tuple(Phase.TRANSPORT for _ in range(steps)),
        phase_boundaries={Phase.TRANSPORT: (0, max(0, steps - 1))},
        grasped=True,
        placed=abort is None,
        final_square="d6" if abort is None else None,
        abort_reason=abort,
        diagnostics=ExecutionDiagnostics(0.1, 0.002, steps, steps * 10, 42.0),
    )
    spec = MoveSpec(source="d7", target="d6", piece_type=PieceType.PAWN)
    return Episode(spec, instruction, frames, execution, "sha123", fps=10.0)


def test_capture_round_trips(tmp_path: Path) -> None:
    ep = make_episode()
    save_capture(tmp_path / "ep-0000", ep)
    back = load_capture(tmp_path / "ep-0000")

    assert back.instruction == ep.instruction
    assert back.fps == ep.fps
    assert back.engine_version == "sha123"
    assert (back.spec.source, back.spec.target, back.spec.piece_type) == (
        "d7",
        "d6",
        PieceType.PAWN,
    )
    for original, restored in zip(ep.frames, back.frames, strict=True):
        np.testing.assert_array_equal(original.images["wrist"], restored.images["wrist"])
        np.testing.assert_allclose(original.state, restored.state)
        np.testing.assert_allclose(original.action, restored.action)
        assert restored.phase == original.phase
        assert restored.control_step == original.control_step
        assert restored.timestamp == original.timestamp

    result = back.execution
    assert (result.grasped, result.placed, result.final_square) == (True, True, "d6")
    assert result.abort_reason is None
    np.testing.assert_allclose(result.states, ep.execution.states)
    np.testing.assert_allclose(result.actions, ep.execution.actions)
    assert result.phase_boundaries == ep.execution.phase_boundaries
    assert result.diagnostics == ep.execution.diagnostics


def test_capture_round_trips_an_aborted_run(tmp_path: Path) -> None:
    ep = make_episode(abort=AbortReason.TIMEOUT)
    save_capture(tmp_path / "ep-0000", ep)
    back = load_capture(tmp_path / "ep-0000")
    assert back.execution.abort_reason is AbortReason.TIMEOUT
    assert back.execution.placed is False
    assert back.execution.final_square is None


def test_capture_rejects_an_empty_episode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no frames"):
        save_capture(tmp_path / "ep-0000", make_episode(n_frames=0))


def test_capture_discards_a_stale_tmp_and_commits_atomically(tmp_path: Path) -> None:
    target = tmp_path / "ep-0000"
    stale = tmp_path / "ep-0000.tmp"
    stale.mkdir()
    (stale / "junk").write_text("leftover from an interrupted write")

    save_capture(target, make_episode())

    assert target.exists()
    assert not stale.exists()
    assert load_capture(target).instruction == "move the pawn from d7 to d6"


def test_build_specs_cycles_the_working_squares() -> None:
    specs = build_specs(6)
    assert [(s.source, s.target) for s in specs] == [
        ("c7", "c6"),
        ("d7", "d6"),
        ("e7", "e6"),
        ("f7", "f6"),
        ("c7", "c6"),
        ("d7", "d6"),
    ]
    assert all(s.piece_type is PieceType.PAWN for s in specs)


def test_pending_indices_skips_existing_captures(tmp_path: Path) -> None:
    capture_dir(tmp_path, 0).mkdir()
    capture_dir(tmp_path, 2).mkdir()
    assert pending_indices(tmp_path, 4) == [1, 3]
