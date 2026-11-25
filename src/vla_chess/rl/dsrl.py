"""Diffusion steering (DSRL): reinforcement learning over a frozen policy's initial noise.

The frozen policy reads each chunk's observation, a steering agent picks the noise the policy's
action expert integrates from, and the resulting chunk is executed. The agent is trained by soft
actor-critic on the chunk-level transitions, so the frozen policy's weights never change and
everything the agent can do is something the policy's own prior can produce.

Episodes run in one of four steering modes, and a training run is a schedule of them: the frozen
policy as trained, as a baseline; latents drawn at random to seed the replay; the actor's sampled
latents while learning; and the actor's mode for evaluation.

Under `Steer.OFFSET` every episode of a seed draws the same Gaussian noise, chunk for chunk, and
the latent only shifts it, so a steered episode and the frozen policy's episode on the same seed
are paired draws.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TypeVar

import numpy as np
import numpy.typing as npt

from vla_chess.rl.env import Observation, Outcome, StepResult
from vla_chess.rl.noise import Fill, NoiseLayout, Steer


class Steering(Enum):
    """Where an episode's noise comes from."""

    BASE = "base"  # every entry a fresh Gaussian draw: the frozen policy as it was trained
    # Random latents, stored for the replay: under REPLACE the Gaussian draw of the steered block,
    # under OFFSET a uniform draw over the latent box.
    WARMUP = "warmup"
    EXPLORE = "explore"  # the actor's sampled latent, learned from
    GREEDY = "greedy"  # the actor's mode, not learned from


class Encoded(Protocol):
    """An observation read by the frozen policy."""

    @property
    def features(self) -> npt.NDArray[np.float32]: ...


E = TypeVar("E", bound=Encoded)


class FrozenPolicy(Protocol[E]):
    """A policy whose chunks are decoded from a chosen initial noise."""

    def encode(self, observation: Observation) -> E: ...

    def decode(self, encoding: E, noise: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]: ...


class Agent(Protocol):
    """A learner that picks latents from observation features and trains on transitions."""

    def __len__(self) -> int: ...

    def act(self, features: npt.NDArray[np.floating], explore: bool) -> npt.NDArray[np.float32]: ...

    def remember(
        self,
        features: npt.NDArray[np.floating],
        latent: npt.NDArray[np.floating],
        reward: float,
        next_features: npt.NDArray[np.floating],
        terminated: bool,
        truncated: bool,
    ) -> None: ...

    def update(self) -> dict[str, float]: ...


class Env(Protocol):
    """An environment stepped one chunk of actions at a time."""

    def reset(self, rng: np.random.Generator | None = None) -> Observation: ...

    def step(self, actions: npt.NDArray[np.floating]) -> StepResult: ...


@dataclass(frozen=True)
class SteeringConfig:
    """How a latent becomes the frozen policy's initial noise."""

    layout: NoiseLayout
    # REPLACE: noise value a latent entry of 1 stands for; OFFSET: shift it stands for, in
    # standard deviations of the noise.
    magnitude: float = 1.5
    fill: Fill = Fill.ZERO  # entries outside the steered block, under REPLACE
    steer: Steer = Steer.REPLACE


@dataclass(frozen=True)
class EpisodeReport:
    """What one episode did and how it ended."""

    steering: Steering
    seed: int
    outcome: Outcome
    episode_return: float
    chunks: int
    updates: int
    wall_seconds: float
    losses: dict[str, float] = field(default_factory=dict)  # means over the episode's updates


def run_episode(
    env: Env,
    policy: FrozenPolicy[Any],
    agent: Agent | None,
    steering: Steering,
    config: SteeringConfig,
    seed: int,
    learn_after: int | None = None,
    updates_per_chunk: int = 1,
) -> EpisodeReport:
    """Drive one episode, storing and learning from its transitions when the mode calls for it.

    Warm-up and explore episodes store every chunk's transition. Once the replay holds
    `learn_after` transitions, each stored chunk is followed by `updates_per_chunk` updates.

    Args:
        env: The environment; reset from `seed`, so a seed always gives the same start.
        policy: The frozen policy.
        agent: The steering agent; needed by every mode but `Steering.BASE`.
        steering: Where the episode's noise comes from.
        config: How a latent becomes noise.
        seed: Seeds the start state and every Gaussian draw of the episode.
        learn_after: Replay size updates start at; None never updates.
        updates_per_chunk: Updates after each stored chunk, once learning has started.

    Returns:
        The episode's outcome, return, chunk and update counts, and mean update losses.

    Raises:
        ValueError: If a mode that needs the agent is run without one.
    """
    if agent is None and steering in (Steering.EXPLORE, Steering.GREEDY):
        raise ValueError(f"{steering.value} episodes need an agent")
    learning = agent is not None and steering in (Steering.WARMUP, Steering.EXPLORE)
    started = time.monotonic()
    start_rng, noise_rng, latent_rng = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
    )
    layout = config.layout
    encoding = policy.encode(env.reset(start_rng))
    episode_return = 0.0
    chunks = updates = 0
    totals: dict[str, float] = {}
    while True:
        latent: npt.NDArray[np.float32] | None = None
        noise: npt.NDArray[np.floating]
        if steering is Steering.BASE:
            noise = noise_rng.standard_normal((layout.chunk_size, layout.max_action_dim))
        elif config.steer is Steer.OFFSET:
            if steering is Steering.WARMUP:
                latent = latent_rng.uniform(-1.0, 1.0, layout.latent_dim).astype(np.float32)
            else:
                assert agent is not None
                latent = agent.act(encoding.features, explore=steering is Steering.EXPLORE)
            noise = layout.offset(latent, config.magnitude, noise_rng)
        else:
            if steering is Steering.WARMUP:
                draw = noise_rng.standard_normal((layout.chunk_size, layout.max_action_dim))
                latent = layout.latent_of(draw, config.magnitude)
            else:
                assert agent is not None
                latent = agent.act(encoding.features, explore=steering is Steering.EXPLORE)
            noise = layout.noise(latent, config.magnitude, config.fill, noise_rng)
        result = env.step(policy.decode(encoding, noise))
        chunks += 1
        episode_return += result.reward
        ended = result.terminated or result.truncated
        next_encoding = None
        if not ended or (learning and result.truncated):
            next_encoding = policy.encode(result.observation)
        if learning:
            assert agent is not None and latent is not None
            following = encoding if next_encoding is None else next_encoding
            agent.remember(
                encoding.features,
                latent,
                result.reward,
                following.features,
                result.terminated,
                result.truncated,
            )
            if learn_after is not None and len(agent) >= learn_after:
                for _ in range(updates_per_chunk):
                    for key, value in agent.update().items():
                        totals[key] = totals.get(key, 0.0) + value
                    updates += 1
        if ended:
            assert result.outcome is not None
            return EpisodeReport(
                steering=steering,
                seed=seed,
                outcome=result.outcome,
                episode_return=episode_return,
                chunks=chunks,
                updates=updates,
                wall_seconds=time.monotonic() - started,
                losses={key: value / updates for key, value in totals.items()} if updates else {},
            )
        assert next_encoding is not None
        encoding = next_encoding


@dataclass(frozen=True)
class Schedule:
    """The episodes a training run drives, in order."""

    warmup_episodes: int = 20
    train_episodes: int = 1000
    eval_every: int = 100  # train episodes between evaluations
    eval_episodes: int = 50
    learn_after: int = 256  # replay size updates start at
    updates_per_chunk: int = 1
    eval_seed_offset: int = 1_000_000  # evaluation seeds start here, clear of training seeds


def eval_seeds(schedule: Schedule) -> list[int]:
    """Seeds every evaluation of a run drives, so each one sees the same starts."""
    return [schedule.eval_seed_offset + i for i in range(schedule.eval_episodes)]


def evaluate(
    env: Env,
    policy: FrozenPolicy[Any],
    agent: Agent | None,
    steering: Steering,
    config: SteeringConfig,
    seeds: Sequence[int],
) -> list[EpisodeReport]:
    """Drive one episode per seed without learning.

    Args:
        env: The environment, usually scored as the closed-loop eval scores a rollout.
        policy: The frozen policy.
        agent: The steering agent; None only for `Steering.BASE`.
        steering: `Steering.BASE` for the frozen policy, `Steering.GREEDY` for the actor's mode.
        config: How a latent becomes noise.
        seeds: One episode per seed.

    Returns:
        The episodes' reports, in seed order.
    """
    return [run_episode(env, policy, agent, steering, config, seed) for seed in seeds]


def train(
    train_env: Env,
    eval_env: Env,
    policy: FrozenPolicy[Any],
    agent: Agent,
    config: SteeringConfig,
    schedule: Schedule,
    on_episode: Callable[[str, EpisodeReport], None],
    on_eval: Callable[[int, Steering, list[EpisodeReport]], None],
) -> None:
    """Run a training schedule: baselines, warm-up, then learning with periodic evaluation.

    Training episodes are seeded 0, 1, 2, ... in the order they run. Every evaluation drives the
    same seeds, first the frozen policy and the untrained actor, then the actor every
    `eval_every` training episodes and once more at the end.

    Args:
        train_env: The environment learned in.
        eval_env: The environment evaluations are scored in.
        policy: The frozen policy.
        agent: The steering agent, trained in place.
        config: How a latent becomes noise.
        schedule: The episodes to run.
        on_episode: Called with "warmup" or "train" and the report after every such episode.
        on_eval: Called with the training episodes run so far, the mode and the reports after
            every evaluation.
    """
    seeds = eval_seeds(schedule)
    on_eval(0, Steering.BASE, evaluate(eval_env, policy, None, Steering.BASE, config, seeds))
    on_eval(0, Steering.GREEDY, evaluate(eval_env, policy, agent, Steering.GREEDY, config, seeds))
    seed = 0
    for _ in range(schedule.warmup_episodes):
        report = run_episode(
            train_env,
            policy,
            agent,
            Steering.WARMUP,
            config,
            seed,
            learn_after=schedule.learn_after,
            updates_per_chunk=schedule.updates_per_chunk,
        )
        on_episode("warmup", report)
        seed += 1
    for done in range(1, schedule.train_episodes + 1):
        report = run_episode(
            train_env,
            policy,
            agent,
            Steering.EXPLORE,
            config,
            seed,
            learn_after=schedule.learn_after,
            updates_per_chunk=schedule.updates_per_chunk,
        )
        on_episode("train", report)
        seed += 1
        if done % schedule.eval_every == 0 or done == schedule.train_episodes:
            reports = evaluate(eval_env, policy, agent, Steering.GREEDY, config, seeds)
            on_eval(done, Steering.GREEDY, reports)
