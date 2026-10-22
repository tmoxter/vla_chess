import subprocess
from typing import Any

import numpy as np
import pytest

from vla_chess.control.executor import ExecutionDiagnostics, ExecutionResult
from vla_chess.control.planner import MoveSpec, Phase
from vla_chess.dataset import episode as episode_module
from vla_chess.dataset.episode import (
    ENGINE_VERSION_UNKNOWN,
    CandidateSelection,
    Episode,
    EpisodeRunner,
    git_engine_version,
    move_scene_description,
    select_kept,
)
from vla_chess.grasp.solver import GRASP_HEIGHT
from vla_chess.sim.positions import starting_position
from vla_chess.sim.scene import (
    BIRDS_EYE_ARM_QPOS,
    DEFAULT_BOARD_POSE,
    PieceType,
    grip_radius,
    pinch_offset_for,
)


def pawn_move(source: str = "d7", target: str = "d6") -> MoveSpec:
    return MoveSpec(source=source, target=target, piece_type=PieceType.PAWN)


def stub_result(succeeded: bool) -> ExecutionResult:
    states = np.zeros((1, 6), dtype=np.float64)
    return ExecutionResult(
        states=states,
        actions=states.copy(),
        phases=(Phase.TRANSPORT,),
        phase_boundaries={Phase.TRANSPORT: (0, 0)},
        grasped=succeeded,
        placed=succeeded,
        final_square="d6" if succeeded else "d7",
        abort_reason=None,
        diagnostics=ExecutionDiagnostics(0.0, 0.0, 1, 1, 0.0),
    )


def test_scene_description_sizes_the_pinch_site_for_the_moved_pawn() -> None:
    spec = pawn_move()
    description = move_scene_description(
        spec, starting_position(), DEFAULT_BOARD_POSE, BIRDS_EYE_ARM_QPOS
    )
    expected = pinch_offset_for(grip_radius(PieceType.PAWN, GRASP_HEIGHT))
    assert description.pinch_offset == pytest.approx(expected)
    assert description.board_pose == DEFAULT_BOARD_POSE
    assert description.arm_qpos == BIRDS_EYE_ARM_QPOS
    assert description.pieces == starting_position()


def test_scene_description_rejects_an_empty_source_square() -> None:
    spec = MoveSpec(source="a4", target="a5", piece_type=PieceType.PAWN)
    with pytest.raises(ValueError, match="no piece on a4"):
        move_scene_description(spec, starting_position(), DEFAULT_BOARD_POSE, BIRDS_EYE_ARM_QPOS)


def test_scene_description_rejects_a_piece_type_mismatch() -> None:
    spec = MoveSpec(source="d7", target="d6", piece_type=PieceType.ROOK)
    with pytest.raises(ValueError, match="carries a pawn"):
        move_scene_description(spec, starting_position(), DEFAULT_BOARD_POSE, BIRDS_EYE_ARM_QPOS)


def test_episode_succeeded_reads_the_rollouts_verdict() -> None:
    spec = pawn_move()
    won = Episode(spec, spec.instruction(), [], stub_result(True), "abc")
    lost = Episode(spec, spec.instruction(), [], stub_result(False), "abc")
    assert won.succeeded
    assert not lost.succeeded


def test_run_all_yields_one_episode_per_move_with_offset_seeds() -> None:
    runner = EpisodeRunner(engine_version="abc")
    specs = [pawn_move("d7", "d6"), pawn_move("e7", "e6"), pawn_move("c7", "c6")]
    seen: list[tuple[MoveSpec, int]] = []

    def fake_run(spec: MoveSpec, *, seed: int = 0) -> Episode:
        seen.append((spec, seed))
        return Episode(spec, spec.instruction(), [], stub_result(True), "abc")

    runner.run = fake_run  # type: ignore[method-assign]
    episodes = list(runner.run_all(specs, base_seed=10))

    assert [ep.spec for ep in episodes] == specs
    assert seen == [(specs[0], 10), (specs[1], 11), (specs[2], 12)]


def test_select_kept_first_takes_the_baseline() -> None:
    accepted = ["baseline", "jittered-1", "jittered-2"]
    kept = select_kept(accepted, CandidateSelection.FIRST, np.random.default_rng(0))
    assert kept == "baseline"


def test_select_kept_random_is_reproducible_from_its_seed() -> None:
    accepted = ["baseline", "jittered-1", "jittered-2"]
    first = select_kept(accepted, CandidateSelection.RANDOM, np.random.default_rng(7))
    second = select_kept(accepted, CandidateSelection.RANDOM, np.random.default_rng(7))
    assert first == second
    assert first in accepted


def test_select_kept_random_reaches_past_the_baseline() -> None:
    accepted = ["baseline", "jittered-1", "jittered-2"]
    drawn = {
        select_kept(accepted, CandidateSelection.RANDOM, np.random.default_rng(seed))
        for seed in range(50)
    }
    # The whole point of random selection is that a non-baseline candidate can be kept.
    assert drawn - {"baseline"}
    assert drawn <= set(accepted)


def test_select_kept_rejects_an_empty_set() -> None:
    with pytest.raises(ValueError, match="empty"):
        select_kept([], CandidateSelection.RANDOM, np.random.default_rng(0))


def test_engine_version_reports_the_short_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        out = "deadbee\n" if cmd[1] == "rev-parse" else ""
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(episode_module.subprocess, "run", fake_run)
    assert git_engine_version() == "deadbee"


def test_engine_version_marks_a_dirty_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        out = "deadbee\n" if cmd[1] == "rev-parse" else " M src/vla_chess/foo.py\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(episode_module.subprocess, "run", fake_run)
    assert git_engine_version() == "deadbee-dirty"


def test_engine_version_falls_back_when_git_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="not a repo")

    monkeypatch.setattr(episode_module.subprocess, "run", fake_run)
    assert git_engine_version() == ENGINE_VERSION_UNKNOWN


def test_engine_version_falls_back_when_git_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("git")

    monkeypatch.setattr(episode_module.subprocess, "run", boom)
    assert git_engine_version("none") == "none"
