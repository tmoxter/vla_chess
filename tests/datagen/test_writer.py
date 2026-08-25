from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from vla_chess.control.executor import ExecutionDiagnostics, ExecutionResult
from vla_chess.control.planner import MoveSpec, Phase
from vla_chess.dataset.episode import Episode
from vla_chess.dataset.observe import Frame
from vla_chess.dataset.writer import (
    STATE_FEATURE_NAMES,
    DatasetStats,
    LeRobotWriter,
    episode_features,
)
from vla_chess.sim.scene import PieceType

FRAME_SIZE = 48


def make_episode(
    instruction: str, n_frames: int, seed: int, fps: float = 10.0
) -> tuple[Episode, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """An episode of synthetic frames, plus the state and action arrays that went into it."""
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(n_frames, 6))
    actions = rng.normal(size=(n_frames, 6))
    frames = [
        Frame(
            images={"wrist": np.full((FRAME_SIZE, FRAME_SIZE, 3), (i * 30) % 255, dtype=np.uint8)},
            state=states[i],
            action=actions[i],
            timestamp=i / fps,
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
    return Episode(spec, instruction, frames, execution, "test-sha", fps=fps), states, actions


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force every dataset load to stay local so a Hub fall-back becomes a hard error."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")


def test_features_are_derived_from_the_first_frame() -> None:
    episode, _, _ = make_episode("move the pawn from d7 to d6", n_frames=3, seed=0)
    features = episode_features(episode)
    assert features["observation.images.wrist"]["shape"] == (FRAME_SIZE, FRAME_SIZE, 3)
    assert features["observation.images.wrist"]["dtype"] == "video"
    assert features["observation.state"]["shape"] == (6,)
    assert features["action"]["names"] == list(STATE_FEATURE_NAMES)


def test_empty_episode_is_rejected(tmp_path: Path) -> None:
    empty, _, _ = make_episode("move the pawn from d7 to d6", n_frames=0, seed=0)
    writer = LeRobotWriter(tmp_path / "ds")
    with pytest.raises(ValueError, match="no frames"):
        writer.add_episode(empty)


def test_write_finalize_and_offline_read_round_trips(tmp_path: Path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "chess"
    writer = LeRobotWriter(root, repo_id="vla_chess/test")
    specs = [
        ("move the pawn from d7 to d6", 4),
        ("move the pawn from e7 to e6", 5),
        ("move the pawn from c7 to c6", 3),
    ]
    episodes = [make_episode(text, n, seed=i) for i, (text, n) in enumerate(specs)]

    indices = [writer.add_episode(ep) for ep, _, _ in episodes]
    stats = writer.finalize()

    assert indices == [0, 1, 2]
    assert stats == DatasetStats(num_episodes=3, num_frames=4 + 5 + 3)

    back = LeRobotDataset("vla_chess/test", root=root)
    assert back.num_episodes == 3
    assert back.num_frames == 12

    # First frame of the second episode sits at global index 4 (the first episode has 4 frames).
    _, states, actions = episodes[1]
    sample = back[4]
    np.testing.assert_allclose(
        np.asarray(sample["observation.state"], dtype=np.float32),
        states[0].astype(np.float32),
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(sample["action"], dtype=np.float32),
        actions[0].astype(np.float32),
        rtol=0,
        atol=1e-5,
    )
    assert sample["task"] == "move the pawn from e7 to e6"
