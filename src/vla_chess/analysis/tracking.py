"""Turn a run's on-disk records — training log, configs, eval reports — into tracker entries.

Pure parsing and flattening, so it runs and is tested without the tracking backend installed; the
tracker itself is driven by `cli.log_experiment`.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

# lerobot-train prints a tqdm bar and, every log interval, an INFO line straight after it on the
# same carriage-return-separated segment. The INFO line abbreviates the step ("step:18K" is both
# 18000 and 18200), so the exact step is read off the bar's counter in front of it.
_BAR_STEP: Final = re.compile(r"\|\s*(\d+)/\d+ \[")
_LOG_FIELD: Final = re.compile(r"\b(loss|grdn|lr|updt_s|data_s):([-+0-9.eE]+)")

# MLflow rejects parameter values longer than this.
MAX_PARAM_LENGTH: Final = 500


@dataclass(frozen=True)
class TrainPoint:
    """One logged point of a training run."""

    step: int
    loss: float
    grad_norm: float
    lr: float
    update_s: float
    data_s: float


def parse_train_log(text: str) -> list[TrainPoint]:
    """Read the logged training points out of a lerobot-train console log.

    Args:
        text: The log as written, carriage returns and all.

    Returns:
        Every logged point, in the order logged.

    Raises:
        ValueError: If a loss line carries no progress-bar step to pin it to.
    """
    points = []
    for segment in re.split(r"[\r\n]", text):
        if "ot_train.py" not in segment or " loss:" not in segment:
            continue
        head = segment[: segment.index("INFO")] if "INFO" in segment else ""
        bars = _BAR_STEP.findall(head)
        if not bars:
            raise ValueError(f"loss line without a progress-bar step: {segment[:120]!r}")
        fields = {name: float(value) for name, value in _LOG_FIELD.findall(segment)}
        points.append(
            TrainPoint(
                step=int(bars[-1]),
                loss=fields["loss"],
                grad_norm=fields["grdn"],
                lr=fields["lr"],
                update_s=fields["updt_s"],
                data_s=fields["data_s"],
            )
        )
    return points


def flatten_params(config: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten a nested config into dotted parameter names with string values.

    Args:
        config: The config, as loaded from JSON.
        prefix: Name prefix for every key, without the trailing dot.

    Returns:
        One entry per leaf, values cut to `MAX_PARAM_LENGTH` characters; None leaves are dropped.
    """
    flat: dict[str, str] = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_params(value, name))
        elif value is not None:
            flat[name] = str(value)[:MAX_PARAM_LENGTH]
    return flat


def policy_sweep_metrics(report: Mapping[str, Any], prefix: str) -> dict[str, float]:
    """Pull the headline numbers out of an `eval_metrics` policy-sweep report.

    Args:
        report: The report as loaded from its JSON file.
        prefix: Metric name prefix, e.g. "eval_d7d6".

    Returns:
        Grasp, lift and placement rates with their 95% bounds, the rollout count, the terminal
        piece distance summary, and per-move rates when the sweep drove more than one move.
    """
    metrics: dict[str, float] = {f"{prefix}/rollouts": float(report["rollouts"])}
    for rate in ("grasp_rate", "lift_rate", "placement_rate"):
        metrics[f"{prefix}/{rate}"] = float(report[rate]["p"])
        metrics[f"{prefix}/{rate}_ci95_lo"] = float(report[rate]["ci95"][0])
        metrics[f"{prefix}/{rate}_ci95_hi"] = float(report[rate]["ci95"][1])
    for stat, value in report["terminal_to_target_m"].items():
        metrics[f"{prefix}/terminal_to_target_m_{stat}"] = float(value)
    per_move = report.get("per_move") or {}
    if len(per_move) > 1:
        for move, tally in per_move.items():
            name = move.replace("->", "_")
            for outcome in ("grasped", "lifted", "placed"):
                metrics[f"{prefix}/{name}/{outcome}_rate"] = tally[outcome] / tally["rollouts"]
    return metrics


def expert_baseline_metrics(report: Mapping[str, Any], prefix: str) -> dict[str, float]:
    """Pull the counts out of an `eval_metrics` expert-baseline report.

    Args:
        report: The report as loaded from its JSON file.
        prefix: Metric name prefix, e.g. "expert_d7d6".

    Returns:
        The repetitions, and the grasp and placement rates over them.
    """
    reps = int(report["reps"])
    return {
        f"{prefix}/reps": float(reps),
        f"{prefix}/grasp_rate": report["grasp_count"] / reps,
        f"{prefix}/placement_rate": report["placement_count"] / reps,
    }


def funnel_metrics(
    manifests: Sequence[Mapping[str, Any]], prefix: str = "datagen"
) -> dict[str, float]:
    """Total the yield of a funnel generation over the manifests its shards wrote.

    Every shard's manifest reports the whole pool's slot count but only its own kept episodes and
    rejects, so the slots are read once and the rest is summed.

    Args:
        manifests: One manifest per shard of a single generation, as loaded from JSON.
        prefix: Metric name prefix.

    Returns:
        Slots, kept episodes and their ratio, and the count of each reject reason.

    Raises:
        ValueError: If no manifest is given, or they disagree on the pool's slot count.
    """
    if not manifests:
        raise ValueError("no funnel manifests to total")
    slots = {int(m["slots"]) for m in manifests}
    if len(slots) != 1:
        raise ValueError(f"the manifests come from different pools: slot counts {sorted(slots)}")
    total = slots.pop()
    kept = sum(int(m["kept"]) for m in manifests)
    rejects: Counter[str] = Counter()
    for manifest in manifests:
        rejects.update(manifest.get("reason_totals", {}))
    metrics = {
        f"{prefix}/slots": float(total),
        f"{prefix}/kept": float(kept),
        f"{prefix}/kept_fraction": kept / total,
    }
    for reason, count in rejects.items():
        metrics[f"{prefix}/rejects/{reason}"] = float(count)
    return metrics


def takeover_metrics(manifest: Mapping[str, Any], prefix: str = "takeovers") -> dict[str, float]:
    """Pull the tallies out of a DAgger round's takeover manifest.

    Args:
        manifest: The manifest as loaded from its JSON file.
        prefix: Metric name prefix.

    Returns:
        Rollouts, frames tried and takeovers kept, then the frames tried and the takeovers kept
        per situation, and the count of each reject reason.
    """
    metrics = {
        f"{prefix}/{key}": float(manifest[key]) for key in ("rollouts", "frames_tried", "kept")
    }
    for group, name in (
        ("situations", "tried"),
        ("kept_by_situation", "kept"),
        ("rejects", "rejects"),
    ):
        for key, count in manifest[group].items():
            metrics[f"{prefix}/{name}/{key}"] = float(count)
    return metrics
