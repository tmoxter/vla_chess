import pytest

from vla_chess.analysis.tracking import (
    MAX_PARAM_LENGTH,
    TrainPoint,
    expert_baseline_metrics,
    flatten_params,
    funnel_metrics,
    parse_train_log,
    policy_sweep_metrics,
    takeover_metrics,
)

LOG = (
    "INFO 2026-10-27 08:41:40 ot_train.py:272 {'batch_size': 32,\n"
    "Training:   1%|          | 199/20000 [01:10<1:43:48,  3.18step/s]\r"
    "Training:   1%|          | 200/20000 [01:10<1:43:33,  3.19step/s]INFO 2026-10-27 08:43:06 "
    "ot_train.py:641 step:200 smpl:6K ep:13 epch:0.11 loss:0.305 grdn:3.635 lr:1.5e-05 "
    "updt_s:0.314 data_s:0.038 smp/s:91\n"
    "Training:  91%|█████████ | 18200/20000 [1:37:03<09:31,  3.15step/s]INFO 2026-10-27 10:19:00 "
    "ot_train.py:641 step:18K smpl:582K ep:1K epch:9.94 loss:0.016 grdn:0.472 lr:4.7e-06 "
    "updt_s:0.312 data_s:0.005 smp/s:101\r"
    "Training:  91%|█████████ | 18201/20000 [1:37:03<09:31,  3.15step/s]"
)


def test_the_training_log_yields_one_point_per_loss_line() -> None:
    assert parse_train_log(LOG) == [
        TrainPoint(200, 0.305, 3.635, 1.5e-05, 0.314, 0.038),
        TrainPoint(18200, 0.016, 0.472, 4.7e-06, 0.312, 0.005),
    ]


def test_an_abbreviated_step_is_read_off_the_progress_bar() -> None:
    assert [p.step for p in parse_train_log(LOG)] == [200, 18200]


def test_a_loss_line_without_a_progress_bar_is_refused() -> None:
    bare = "INFO 2026 ot_train.py:641 step:18K loss:0.016 grdn:0.4 lr:1e-5 updt_s:0.3 data_s:0.0"
    with pytest.raises(ValueError, match="progress-bar"):
        parse_train_log(bare)


def test_nested_configs_flatten_to_dotted_names() -> None:
    config = {"batch_size": 32, "policy": {"chunk_size": 50, "device": None}, "tags": [1, 2]}
    assert flatten_params(config) == {
        "batch_size": "32",
        "policy.chunk_size": "50",
        "tags": "[1, 2]",
    }


def test_long_parameter_values_are_cut_to_the_tracker_limit() -> None:
    assert len(flatten_params({"x": "a" * 2000})["x"]) == MAX_PARAM_LENGTH


def _rate(p: float) -> dict[str, object]:
    return {"p": p, "ci95": [p - 0.01, p + 0.01]}


def test_a_sweep_report_yields_its_rates_and_bounds() -> None:
    report = {
        "rollouts": 1000,
        "grasp_rate": _rate(0.5),
        "lift_rate": _rate(0.4),
        "placement_rate": _rate(0.1),
        "terminal_to_target_m": {"mean": 0.1, "median": 0.05},
        "per_move": {"d7->d6": {"rollouts": 1000, "grasped": 500, "lifted": 400, "placed": 100}},
    }
    metrics = policy_sweep_metrics(report, "eval")
    assert metrics["eval/placement_rate"] == 0.1
    assert metrics["eval/grasp_rate_ci95_lo"] == pytest.approx(0.49)
    assert metrics["eval/terminal_to_target_m_median"] == 0.05
    assert not any("d7_d6" in name for name in metrics)


def test_a_multi_move_sweep_also_yields_per_move_rates() -> None:
    report = {
        "rollouts": 4,
        "grasp_rate": _rate(0.5),
        "lift_rate": _rate(0.5),
        "placement_rate": _rate(0.25),
        "terminal_to_target_m": {},
        "per_move": {
            "c7->c6": {"rollouts": 2, "grasped": 1, "lifted": 1, "placed": 1},
            "d7->d6": {"rollouts": 2, "grasped": 1, "lifted": 1, "placed": 0},
        },
    }
    metrics = policy_sweep_metrics(report, "sweep")
    assert metrics["sweep/c7_c6/placed_rate"] == 0.5
    assert metrics["sweep/d7_d6/placed_rate"] == 0.0


def test_a_sweep_that_scored_at_done_yields_those_rates_too() -> None:
    report = {
        "rollouts": 4,
        "grasp_rate": _rate(0.5),
        "lift_rate": _rate(0.5),
        "placement_rate": _rate(0.25),
        "done_rate": _rate(0.5),
        "placement_at_done_rate": _rate(0.2),
        "terminal_to_target_m": {},
        "per_move": {
            "c7->c6": {"rollouts": 2, "placed": 1, "placed_at_done": 1},
            "d7->d6": {"rollouts": 2, "placed": 0, "placed_at_done": 0},
        },
    }
    metrics = policy_sweep_metrics(report, "sweep")
    assert metrics["sweep/done_rate"] == 0.5
    assert metrics["sweep/placement_at_done_rate_ci95_hi"] == pytest.approx(0.21)
    assert metrics["sweep/c7_c6/placed_at_done_rate"] == 0.5


def test_a_sweep_that_tracked_mid_air_releases_yields_their_rates() -> None:
    report = {
        "rollouts": 4,
        "grasp_rate": _rate(0.5),
        "lift_rate": _rate(0.5),
        "placement_rate": _rate(0.25),
        "opened_mid_air_rate": _rate(0.25),
        "slipped_mid_air_rate": _rate(0.0),
        "terminal_to_target_m": {},
        "per_move": {
            "c7->c6": {"rollouts": 2, "grasped": 1, "lifted": 1, "placed": 1, "opened_mid_air": 1},
            "d7->d6": {"rollouts": 2, "grasped": 1, "lifted": 1, "placed": 0, "opened_mid_air": 0},
        },
    }
    metrics = policy_sweep_metrics(report, "sweep")
    assert metrics["sweep/opened_mid_air_rate"] == 0.25
    assert metrics["sweep/slipped_mid_air_rate_ci95_hi"] == pytest.approx(0.01)
    assert metrics["sweep/c7_c6/opened_mid_air_rate"] == 0.5


def test_an_expert_baseline_yields_its_rates() -> None:
    report = {"reps": 10, "grasp_count": 10, "placement_count": 9}
    assert expert_baseline_metrics(report, "expert") == {
        "expert/reps": 10.0,
        "expert/grasp_rate": 1.0,
        "expert/placement_rate": 0.9,
    }


def test_shard_manifests_total_to_one_yield() -> None:
    shards = [
        {"slots": 120, "kept": 15, "reason_totals": {}},
        {"slots": 120, "kept": 14, "reason_totals": {"wrong_square": 2}},
        {"slots": 120, "kept": 15, "reason_totals": {"wrong_square": 1, "ik_diverged": 1}},
    ]
    assert funnel_metrics(shards) == {
        "datagen/slots": 120.0,
        "datagen/kept": 44.0,
        "datagen/kept_fraction": 44 / 120,
        "datagen/rejects/wrong_square": 3.0,
        "datagen/rejects/ik_diverged": 1.0,
    }


def test_manifests_from_different_pools_are_refused() -> None:
    with pytest.raises(ValueError, match="different pools"):
        funnel_metrics([{"slots": 120, "kept": 1}, {"slots": 60, "kept": 1}])


def test_a_takeover_manifest_yields_its_tallies() -> None:
    manifest = {
        "rollouts": 2,
        "frames_tried": 5,
        "kept": 2,
        "situations": {"unrecoverable": 3, "to_grasp": 2},
        "kept_by_situation": {"to_grasp": 2},
        "rejects": {"direct_not_near": 2},
    }
    assert takeover_metrics(manifest) == {
        "takeovers/rollouts": 2.0,
        "takeovers/frames_tried": 5.0,
        "takeovers/kept": 2.0,
        "takeovers/tried/unrecoverable": 3.0,
        "takeovers/tried/to_grasp": 2.0,
        "takeovers/kept/to_grasp": 2.0,
        "takeovers/rejects/direct_not_near": 2.0,
    }


def test_a_takeover_manifest_with_release_tallies_yields_them() -> None:
    manifest = {
        "rollouts": 3,
        "frames_tried": 2,
        "kept": 1,
        "situations": {"in_hand": 2},
        "kept_by_situation": {"in_hand": 1},
        "rejects": {},
        "rollouts_released_mid_air": {"opened": 2, "slipped": 1},
    }
    metrics = takeover_metrics(manifest)
    assert metrics["takeovers/rollouts_released_mid_air/opened"] == 2.0
    assert metrics["takeovers/rollouts_released_mid_air/slipped"] == 1.0
