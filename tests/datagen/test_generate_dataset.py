from pathlib import Path

import numpy as np
import pytest

from vla_chess.cli.generate_dataset import serialize
from vla_chess.control.executor import ExecutionDiagnostics, ExecutionResult
from vla_chess.control.planner import MoveSpec, Phase
from vla_chess.dataset.capture import save_capture
from vla_chess.dataset.episode import Episode
from vla_chess.dataset.observe import Frame
from vla_chess.sim.scene import PieceType

# SVT-AV1 raises a floating-point exception on very small frames, so keep the test frames well
# above its minimum; real episodes render at 512.
FRAME_SIZE = 64


def make_episode(instruction: str, n_frames: int, seed: int) -> Episode:
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(n_frames, 6))
    actions = rng.normal(size=(n_frames, 6))
    frames = [
        Frame(
            images={"wrist": np.full((FRAME_SIZE, FRAME_SIZE, 3), (i * 17) % 255, dtype=np.uint8)},
            state=states[i],
            action=actions[i],
            timestamp=i / 10.0,
            frame_index=i,
            control_step=i,
            phase=Phase.TRANSPORT,
        )
        for i in range(n_frames)
    ]
    execution = ExecutionResult(
        states=states,
        actions=actions,
        phases=tuple(Phase.TRANSPORT for _ in range(n_frames)),
        phase_boundaries={Phase.TRANSPORT: (0, n_frames - 1)},
        grasped=True,
        placed=True,
        final_square="d6",
        abort_reason=None,
        diagnostics=ExecutionDiagnostics(0.0, 0.0, n_frames, n_frames, 0.0),
    )
    spec = MoveSpec(source="d7", target="d6", piece_type=PieceType.PAWN)
    return Episode(spec, instruction, frames, execution, "sha", fps=10.0)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")


def stage_captures(captures: Path, texts: list[str]) -> None:
    captures.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(texts):
        save_capture(captures / f"ep-{i:04d}", make_episode(text, n_frames=3 + i, seed=i))


def test_serialize_from_captures_round_trips(tmp_path: Path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    captures = tmp_path / "caps"
    texts = [
        "move the pawn from b7 to b6",
        "move the pawn from c7 to c6",
        "move the pawn from d7 to d6",
    ]
    stage_captures(captures, texts)

    stats = serialize(captures, tmp_path / "ds", repo_id="vla_chess/test")

    assert stats.num_episodes == 3
    assert stats.num_frames == 3 + 4 + 5
    back = LeRobotDataset("vla_chess/test", root=tmp_path / "ds")
    assert back.num_episodes == 3
    assert back[0]["task"] == "move the pawn from b7 to b6"


def test_serialize_replaces_an_existing_dataset(tmp_path: Path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    captures = tmp_path / "caps"
    stage_captures(captures, ["move the pawn from b7 to b6", "move the pawn from c7 to c6"])
    out = tmp_path / "ds"

    serialize(captures, out, repo_id="vla_chess/test")
    # Rerunning the (cheap, derived) serialization over the same output must not corrupt it.
    stats = serialize(captures, out, repo_id="vla_chess/test")

    assert stats.num_episodes == 2
    assert LeRobotDataset("vla_chess/test", root=out).num_episodes == 2


def test_serialize_without_captures_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no captures"):
        serialize(tmp_path / "empty", tmp_path / "ds")
