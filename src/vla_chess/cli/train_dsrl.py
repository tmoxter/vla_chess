"""Train a DSRL steering agent over a frozen SmolVLA checkpoint on one pawn move.

Writes into --out: the run's settings (config.json), one line per warm-up and training episode
(episodes.jsonl), one line per evaluation (evals.jsonl), and the agent after every evaluation
(agent-<episodes>.pt, plus agent-last.pt). Evaluations score every episode as the closed-loop
eval scores a rollout: no early stop, judged on where the piece stands after --max-frames.

Runs in the `eval` pixi env. Example:

    pixi run -e eval python -m vla_chess.cli.train_dsrl \
        --checkpoint /workspace/policies/smolvla_chess_dagger_v3_r1_010000 \
        --source d7 --target d6 --device cuda --out /workspace/dsrl/v1

    # evaluate a saved agent, and the frozen policy, on the run's evaluation starts:
    pixi run -e eval python -m vla_chess.cli.train_dsrl --eval-only \
        --agent /workspace/dsrl/v1/agent-last.pt --checkpoint ... --out /workspace/dsrl/v1-eval
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vla_chess.cli.eval_metrics import wilson_interval
from vla_chess.cli.eval_rollout import _spec_for
from vla_chess.dataset.observe import DEFAULT_FPS
from vla_chess.rl.dsrl import (
    EpisodeReport,
    Schedule,
    Steering,
    SteeringConfig,
    eval_seeds,
    evaluate,
    train,
)
from vla_chess.rl.env import EnvConfig, MoveEnv
from vla_chess.rl.noise import Fill


def summarise(reports: list[EpisodeReport]) -> dict[str, Any]:
    """Rates with Wilson 95% intervals over a set of episodes.

    Args:
        reports: The episodes to summarise.

    Returns:
        Grasp, lift, placement, mid-air-opening and loss rates as `{"p", "ci95"}`, the mean
        return and chunk count, and the episode count.
    """
    n = len(reports)

    def rate(count: int) -> dict[str, Any]:
        p, lo, hi = wilson_interval(count, n)
        return {"p": p, "ci95": [lo, hi]}

    outcomes = [r.outcome for r in reports]
    return {
        "episodes": n,
        "grasp_rate": rate(sum(o.grasped for o in outcomes)),
        "lift_rate": rate(sum(o.lifted for o in outcomes)),
        "placement_rate": rate(sum(o.placed for o in outcomes)),
        "opened_mid_air_rate": rate(sum(o.opened_mid_air for o in outcomes)),
        "lost_rate": rate(sum(o.lost for o in outcomes)),
        "mean_return": sum(r.episode_return for r in reports) / n if n else 0.0,
        "mean_chunks": sum(r.chunks for r in reports) / n if n else 0.0,
    }


def _record(report: EpisodeReport) -> dict[str, Any]:
    """Flatten an episode report into one JSON-serialisable line."""
    record = asdict(report)
    record["steering"] = report.steering.value
    return record


def _append(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON line to a file."""
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="frozen SmolVLA pretrained_model dir")
    parser.add_argument("--source", default="d7")
    parser.add_argument("--target", default="d6")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--max-frames", type=int, default=750)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument(
        "--piece-jitter-mm",
        type=float,
        default=3.0,
        help="half-width of the window each episode's start is drawn from around the square centre",
    )
    parser.add_argument("--steered-steps", type=int, default=10, help="noise rows the agent sets")
    parser.add_argument("--magnitude", type=float, default=1.5, help="noise of a latent entry of 1")
    parser.add_argument("--fill", choices=[f.value for f in Fill], default=Fill.ZERO.value)
    parser.add_argument("--warmup-episodes", type=int, default=20)
    parser.add_argument("--train-episodes", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--learn-after", type=int, default=256)
    parser.add_argument("--updates-per-chunk", type=int, default=1)
    parser.add_argument("--hidden", default="512,512,512", help="actor and critic widths")
    parser.add_argument("--num-critics", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--discount", type=float, default=0.99, help="per chunk")
    parser.add_argument("--lr", type=float, default=3e-4, help="actor, critic and temperature")
    parser.add_argument("--temperature-init", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0, help="seeds the networks' initialisation")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="evaluate --agent and the frozen policy on the evaluation starts instead of training",
    )
    parser.add_argument("--agent", type=Path, default=None, help="saved agent for --eval-only")
    args = parser.parse_args()
    if args.eval_only and args.agent is None:
        parser.error("--eval-only needs --agent")

    import torch

    from vla_chess.rl.agent import AgentConfig, SteeringAgent
    from vla_chess.rl.steer import SteerablePolicy

    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2
        )
    )
    spec = _spec_for(0, args.source, args.target)
    jitter = args.piece_jitter_mm / 1000.0
    train_env = MoveEnv(
        spec, EnvConfig(fps=args.fps, max_frames=args.max_frames, piece_jitter_m=jitter)
    )
    eval_env = MoveEnv(
        spec,
        EnvConfig(
            fps=args.fps,
            max_frames=args.max_frames,
            stop_on_placed=False,
            stop_on_lost=False,
            piece_jitter_m=jitter,
        ),
    )
    policy = SteerablePolicy(args.checkpoint, args.device, args.n_action_steps)
    config = SteeringConfig(
        layout=policy.layout(args.steered_steps), magnitude=args.magnitude, fill=Fill(args.fill)
    )
    schedule = Schedule(
        warmup_episodes=args.warmup_episodes,
        train_episodes=args.train_episodes,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        learn_after=args.learn_after,
        updates_per_chunk=args.updates_per_chunk,
    )
    if args.eval_only:
        agent = SteeringAgent.load(args.agent, args.device)
    else:
        agent = SteeringAgent(
            policy.feature_dim,
            config.layout.latent_dim,
            AgentConfig(
                hidden_dims=tuple(int(w) for w in args.hidden.split(",")),
                num_critics=args.num_critics,
                batch_size=args.batch_size,
                discount=args.discount,
                actor_lr=args.lr,
                critic_lr=args.lr,
                temperature_lr=args.lr,
                temperature_init=args.temperature_init,
            ),
            args.device,
        )
    started = time.monotonic()

    def on_episode(kind: str, report: EpisodeReport) -> None:
        record = {"kind": kind, **_record(report), "replay": len(agent)}
        _append(args.out / "episodes.jsonl", record)
        o = report.outcome
        print(
            f"[{int(time.monotonic() - started)}s] {kind} seed {report.seed}: "
            f"{'PLACED' if o.placed else 'lost' if o.lost else 'fail'} "
            f"grasp={o.grasped} chunks={report.chunks} updates={report.updates} "
            f"{report.wall_seconds:.1f}s",
            flush=True,
        )

    def on_eval(done: int, steering: Steering, reports: list[EpisodeReport]) -> None:
        summary = summarise(reports)
        _append(
            args.out / "evals.jsonl",
            {
                "train_episodes": done,
                "steering": steering.value,
                **summary,
                "records": [_record(r) for r in reports],
            },
        )
        if steering is Steering.GREEDY and not args.eval_only:
            agent.save(args.out / f"agent-{done:05d}.pt")
            agent.save(args.out / "agent-last.pt")
        g, p = summary["grasp_rate"], summary["placement_rate"]
        print(
            f"=== eval after {done} train episodes, {steering.value}: grasp {g['p']:.1%} "
            f"placement {p['p']:.1%} (95% CI {p['ci95'][0]:.1%}-{p['ci95'][1]:.1%}) "
            f"over {summary['episodes']} ===",
            flush=True,
        )

    try:
        if args.eval_only:
            seeds = eval_seeds(schedule)
            on_eval(
                0, Steering.BASE, evaluate(eval_env, policy, None, Steering.BASE, config, seeds)
            )
            on_eval(
                0,
                Steering.GREEDY,
                evaluate(eval_env, policy, agent, Steering.GREEDY, config, seeds),
            )
        else:
            train(train_env, eval_env, policy, agent, config, schedule, on_episode, on_eval)
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
