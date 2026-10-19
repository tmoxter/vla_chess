from pathlib import Path

import numpy as np
import pytest

from vla_chess.cli.generate_dataset import (
    EXP2_HELD_OUT,
    EXP2_TRAINED_MOVES,
    ONE_STEP_INSTRUCTION,
    SINGLE_PAWN_INSTRUCTION,
    TWO_STEP_INSTRUCTION,
    WORKING_MOVES,
    build_move_pool_specs,
    build_specs,
    forward_instruction,
    main,
    one_step_moves,
    push,
    serialize,
)
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


def test_build_specs_defaults_to_the_coordinate_label() -> None:
    (spec,) = build_specs(1)
    assert spec.instruction() == "move the pawn from c7 to c6"


def test_single_pawn_specs_carry_the_forward_label_and_cycle_the_files() -> None:
    specs = build_specs(len(WORKING_MOVES), instruction=SINGLE_PAWN_INSTRUCTION)
    assert [s.instruction() for s in specs] == [SINGLE_PAWN_INSTRUCTION] * len(WORKING_MOVES)
    assert [s.source for s in specs] == [move[0] for move in WORKING_MOVES]


def test_forward_instruction_reads_distance_from_the_rank() -> None:
    assert forward_instruction("d7", "d6") == ONE_STEP_INSTRUCTION
    assert forward_instruction("d7", "d5") == TWO_STEP_INSTRUCTION


def test_forward_instruction_rejects_off_file_or_out_of_range_pushes() -> None:
    with pytest.raises(ValueError, match="same-file"):
        forward_instruction("d7", "e6")
    with pytest.raises(ValueError, match="one- nor two"):
        forward_instruction("d7", "d4")


def test_exp2_pool_holds_out_the_generalization_moves() -> None:
    # Every held-out move is eval-only and must never be generated.
    for move in EXP2_HELD_OUT:
        assert move not in EXP2_TRAINED_MOVES
    # The expanded pool trains the full file span a-h (the funnel opened up the outer files), so no
    # file is reserved any more; generalization is measured on held-out squares and depth instead.
    assert {source[0] for source, _ in EXP2_TRAINED_MOVES} == set("abcdefgh")
    # Every move is a same-file forward push of one or two ranks, so it carries a distance label.
    for source, target in EXP2_TRAINED_MOVES:
        assert source[0] == target[0]
        assert abs(int(source[1]) - int(target[1])) in (1, 2)
    # No move is both trained and held out, and the pool has no duplicates.
    assert len(set(EXP2_TRAINED_MOVES)) == len(EXP2_TRAINED_MOVES)


def test_build_move_pool_repeats_every_move_per_seed_with_distance_labels() -> None:
    seeds = 6
    specs = build_move_pool_specs(seeds=seeds)
    assert len(specs) == len(EXP2_TRAINED_MOVES) * seeds
    # Each move appears exactly `seeds` times.
    seen = [(s.source, s.target) for s in specs]
    for move in EXP2_TRAINED_MOVES:
        assert seen.count(move) == seeds
    # Two-step pushes carry the two-field label, one-step the one-field label.
    for spec in specs:
        expected = (
            TWO_STEP_INSTRUCTION
            if abs(int(spec.source[1]) - int(spec.target[1])) == 2
            else ONE_STEP_INSTRUCTION
        )
        assert spec.instruction() == expected


def test_one_step_moves_drops_the_two_step_pushes() -> None:
    kept = one_step_moves(EXP2_TRAINED_MOVES)
    # Every kept move is a single-rank forward push; no two-steppers survive.
    for source, target in kept:
        assert abs(int(source[1]) - int(target[1])) == 1
    # The one-step subset is exactly the pool minus its two-step pushes, order preserved.
    expected = [m for m in EXP2_TRAINED_MOVES if abs(int(m[0][1]) - int(m[1][1])) == 1]
    assert kept == expected
    # There are two-step pushes in the pool, so the filter actually removes something.
    assert len(kept) < len(EXP2_TRAINED_MOVES)


def test_one_step_only_first_finetune_pool_still_spans_every_file() -> None:
    # Dropping the two-step moves must not cost a whole file: the first fine-tune still sees a
    # one-square push on each of a-h, so a single constant label stays positionally diverse.
    kept = one_step_moves(EXP2_TRAINED_MOVES)
    assert {source[0] for source, _ in kept} == set("abcdefgh")


def test_push_without_repo_id_is_rejected_before_any_work() -> None:
    # --push needs a Hub target; main refuses it up front rather than generating then failing.
    assert main(["--push", "--no-serialize"]) == 2


def test_push_delegates_to_the_writer_with_the_repo_and_privacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}

    def fake_push_dataset(
        root: Path, repo_id: str, *, private: bool = True, tags: list[str] | None = None
    ) -> None:
        calls.update(root=root, repo_id=repo_id, private=private)

    monkeypatch.setattr("vla_chess.dataset.writer.push_dataset", fake_push_dataset)
    out = tmp_path / "ds"
    push(out, "me/chess-smoke", private=True)
    assert calls == {"root": out, "repo_id": "me/chess-smoke", "private": True}
