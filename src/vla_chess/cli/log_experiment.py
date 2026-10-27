"""Record one experiment run in the local MLflow store from the files it left on disk.

Runs in the `track` pixi env. Every source is optional, so a run can be logged from whatever it
produced; the source files are attached as artifacts too. Example:

    pixi run -e track python -m vla_chess.cli.log_experiment \
        --experiment vla-chess --run-name dart-v2 \
        --train-log outputs/run_logs/train.log \
        --train-config outputs/run/checkpoints/020000/pretrained_model/train_config.json \
        --sweep eval_d7d6=outputs/run_logs/eval/metrics-d7d6.json \
        --expert expert_d7d6=outputs/run_logs/eval/expert-d7d6.json \
        --funnel data/dataset-captures/funnel-manifest-shard*.json \
        --param dataset=hugginggeorge/chess-pawns-funnel-dart-v2 --tag stage=dart
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vla_chess.analysis.tracking import (
    expert_baseline_metrics,
    flatten_params,
    funnel_metrics,
    parse_train_log,
    policy_sweep_metrics,
    takeover_metrics,
)

DEFAULT_TRACKING_URI = "sqlite:///mlflow/mlflow.db"
DEFAULT_ARTIFACT_ROOT = Path("mlflow/artifacts")


def _named(spec: str) -> tuple[str, Path]:
    """Split a 'name=path' argument."""
    name, sep, path = spec.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"{spec!r} is not a name=path pair")
    return name, Path(path)


def _pair(spec: str) -> tuple[str, str]:
    """Split a 'key=value' argument."""
    key, sep, value = spec.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"{spec!r} is not a key=value pair")
    return key, value


def _load(path: Path) -> dict[str, Any]:
    """Read one JSON object from a file."""
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return loaded


def main(argv: Sequence[str] | None = None) -> int:
    """Log one run's parameters, metrics and source files to MLflow.

    Args:
        argv: Command line without the program name; `sys.argv` is read when omitted.

    Returns:
        Process exit status: 0 on success.
    """
    parser = argparse.ArgumentParser(prog="log-experiment", description=__doc__)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="where a new experiment keeps its artifacts",
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--note", default=None, help="markdown description shown on the run")
    parser.add_argument("--train-log", type=Path, default=None, help="lerobot-train console log")
    parser.add_argument("--train-config", type=Path, default=None, help="train_config.json")
    parser.add_argument(
        "--sweep", type=_named, action="append", default=[], help="name=policy-sweep report"
    )
    parser.add_argument(
        "--expert", type=_named, action="append", default=[], help="name=expert-baseline report"
    )
    parser.add_argument(
        "--funnel", type=Path, nargs="+", default=[], help="funnel manifests of one generation"
    )
    parser.add_argument(
        "--takeovers", type=Path, default=None, help="takeover manifest of a DAgger round"
    )
    parser.add_argument("--param", type=_pair, action="append", default=[], help="key=value")
    parser.add_argument("--tag", type=_pair, action="append", default=[], help="key=value")
    parser.add_argument(
        "--artifact", type=Path, action="append", default=[], help="extra file to attach"
    )
    args = parser.parse_args(argv)

    import mlflow

    mlflow.set_tracking_uri(args.tracking_uri)
    if mlflow.get_experiment_by_name(args.experiment) is None:
        # Artifacts go where the `mlflow-server` task serves them from, not MLflow's ./mlruns.
        location = (args.artifact_root / args.experiment).resolve().as_uri()
        mlflow.create_experiment(args.experiment, artifact_location=location)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name) as run:
        if args.note is not None:
            mlflow.set_tag("mlflow.note.content", args.note)
        mlflow.set_tags(dict(args.tag))
        mlflow.log_params(dict(args.param))
        attachments: list[Path] = list(args.artifact)

        if args.train_config is not None:
            mlflow.log_params(flatten_params(_load(args.train_config), "train"))
            attachments.append(args.train_config)
        if args.train_log is not None:
            points = parse_train_log(args.train_log.read_text(errors="replace"))
            for point in points:
                mlflow.log_metrics(
                    {
                        "train/loss": point.loss,
                        "train/grad_norm": point.grad_norm,
                        "train/lr": point.lr,
                        "train/update_s": point.update_s,
                        "train/data_s": point.data_s,
                    },
                    step=point.step,
                )
            if points:
                mlflow.log_metric("train/final_loss", points[-1].loss, step=points[-1].step)
            attachments.append(args.train_log)
        for name, path in args.sweep:
            mlflow.log_metrics(policy_sweep_metrics(_load(path), name))
            attachments.append(path)
        for name, path in args.expert:
            mlflow.log_metrics(expert_baseline_metrics(_load(path), name))
            attachments.append(path)
        if args.funnel:
            mlflow.log_metrics(funnel_metrics([_load(path) for path in args.funnel]))
            attachments.extend(args.funnel)
        if args.takeovers is not None:
            mlflow.log_metrics(takeover_metrics(_load(args.takeovers)))
            attachments.append(args.takeovers)

        for path in attachments:
            mlflow.log_artifact(str(path))
        print(f"logged run {run.info.run_id} ({args.run_name}) to {args.experiment}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
