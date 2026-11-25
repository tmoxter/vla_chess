from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pytest

from vla_chess.rl.dsrl import (
    EpisodeReport,
    Schedule,
    Steering,
    SteeringConfig,
    eval_seeds,
    run_episode,
    train,
)
from vla_chess.rl.env import Observation, Outcome, StepResult
from vla_chess.rl.noise import Fill, NoiseLayout, Steer

LAYOUT = NoiseLayout(chunk_size=4, max_action_dim=3, action_dim=2, steered_steps=2)
CONFIG = SteeringConfig(layout=LAYOUT, magnitude=1.5, fill=Fill.ZERO)


def _outcome(placed: bool, frames: int) -> Outcome:
    return Outcome(
        grasped=placed,
        lifted=placed,
        placed=placed,
        settled=True,
        final_square="d6" if placed else "d7",
        opened_mid_air=False,
        slipped_mid_air=False,
        lost=False,
        frames=frames,
        done_frame=None,
        placed_at_done=False,
    )


class FakeEnv:
    """Episodes of `length` chunks that succeed, or are truncated, on the last one."""

    def __init__(self, length: int, succeed: bool) -> None:
        self.length = length
        self.succeed = succeed
        self.starts: list[float] = []
        self.actions: list[np.ndarray] = []
        self._chunk = 0

    def _observation(self) -> Observation:
        return Observation(
            frame=np.zeros((2, 2, 3), dtype=np.uint8),
            state=np.full(6, float(self._chunk)),
            instruction="move",
        )

    def reset(self, rng: np.random.Generator | None = None) -> Observation:
        self._chunk = 0
        self.starts.append(float(rng.uniform()) if rng is not None else float("nan"))
        return self._observation()

    def step(self, actions: npt.NDArray[np.floating]) -> StepResult:
        self.actions.append(np.asarray(actions))
        self._chunk += 1
        if self._chunk < self.length:
            return StepResult(self._observation(), 0.0, False, False, None)
        outcome = _outcome(self.succeed, self._chunk)
        if self.succeed:
            return StepResult(self._observation(), 1.0, True, False, outcome)
        return StepResult(self._observation(), 0.0, False, True, outcome)


@dataclass
class FakeEncoding:
    features: npt.NDArray[np.float32]


class FakePolicy:
    """Encodes an observation as its state and decodes a chunk as the noise itself."""

    def __init__(self) -> None:
        self.encoded = 0

    def encode(self, observation: Observation) -> FakeEncoding:
        self.encoded += 1
        return FakeEncoding(observation.state.astype(np.float32))

    def decode(self, encoding: FakeEncoding, noise: npt.NDArray[np.floating]) -> np.ndarray:
        return np.asarray(noise, dtype=np.float64)


@dataclass
class FakeAgent:
    latent: float = 0.5
    stored: list[tuple] = field(default_factory=list)
    updates: int = 0
    explored: list[bool] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.stored)

    def act(self, features: npt.NDArray[np.floating], explore: bool) -> npt.NDArray[np.float32]:
        self.explored.append(explore)
        return np.full(LAYOUT.latent_dim, self.latent, dtype=np.float32)

    def remember(
        self,
        features: npt.NDArray[np.floating],
        latent: npt.NDArray[np.floating],
        reward: float,
        next_features: npt.NDArray[np.floating],
        terminated: bool,
        truncated: bool,
    ) -> None:
        self.stored.append(
            (
                features.copy(),
                np.asarray(latent).copy(),
                reward,
                next_features.copy(),
                terminated,
                truncated,
            )
        )

    def update(self) -> dict[str, float]:
        self.updates += 1
        return {"loss": float(self.updates)}


def test_a_base_episode_draws_every_entry_and_stores_nothing() -> None:
    env = FakeEnv(length=3, succeed=True)
    report = run_episode(env, FakePolicy(), None, Steering.BASE, CONFIG, seed=0)
    assert report.chunks == 3 and report.episode_return == 1.0
    assert report.outcome.placed
    assert all(np.count_nonzero(a) == a.size for a in env.actions)


@pytest.mark.parametrize("steering", [Steering.EXPLORE, Steering.GREEDY])
def test_agent_modes_need_an_agent(steering: Steering) -> None:
    with pytest.raises(ValueError, match="need an agent"):
        run_episode(FakeEnv(2, True), FakePolicy(), None, steering, CONFIG, seed=0)


def test_an_explore_episode_stores_one_transition_per_chunk() -> None:
    agent = FakeAgent()
    run_episode(FakeEnv(3, True), FakePolicy(), agent, Steering.EXPLORE, CONFIG, seed=0)
    assert len(agent.stored) == 3
    assert agent.explored == [True, True, True]
    features = [s[0][0] for s in agent.stored]
    following = [s[3][0] for s in agent.stored]
    assert features == [0.0, 1.0, 2.0]
    assert following[:2] == [1.0, 2.0]
    assert [s[2] for s in agent.stored] == [0.0, 0.0, 1.0]
    assert [s[4] for s in agent.stored] == [False, False, True]


def test_the_latent_sets_the_steered_block_and_zero_fills_the_rest() -> None:
    env = FakeEnv(1, True)
    run_episode(env, FakePolicy(), FakeAgent(latent=0.5), Steering.EXPLORE, CONFIG, seed=0)
    noise = env.actions[0]
    np.testing.assert_allclose(noise[:2, :2], 0.75)
    assert not noise[2:].any() and not noise[:, 2:].any()


def test_a_truncated_episode_bootstraps_from_its_last_observation() -> None:
    agent = FakeAgent()
    policy = FakePolicy()
    run_episode(FakeEnv(2, False), policy, agent, Steering.EXPLORE, CONFIG, seed=0)
    last = agent.stored[-1]
    assert (last[4], last[5]) == (False, True)
    assert last[3][0] == 2.0
    assert policy.encoded == 3


def test_a_greedy_episode_takes_the_mode_and_stores_nothing() -> None:
    agent = FakeAgent()
    policy = FakePolicy()
    run_episode(FakeEnv(2, False), policy, agent, Steering.GREEDY, CONFIG, seed=0)
    assert agent.explored == [False, False]
    assert not agent.stored
    assert policy.encoded == 2


def test_updates_start_once_the_replay_holds_enough() -> None:
    agent = FakeAgent()
    report = run_episode(
        FakeEnv(5, True),
        FakePolicy(),
        agent,
        Steering.EXPLORE,
        CONFIG,
        seed=0,
        learn_after=3,
        updates_per_chunk=2,
    )
    assert report.updates == 6 and agent.updates == 6
    assert report.losses == {"loss": pytest.approx(3.5)}


def test_warmup_latents_reproduce_the_noise_that_was_executed() -> None:
    env = FakeEnv(4, True)
    agent = FakeAgent()
    run_episode(env, FakePolicy(), agent, Steering.WARMUP, CONFIG, seed=3)
    for (_, latent, *_), noise in zip(agent.stored, env.actions, strict=True):
        np.testing.assert_allclose(noise[:2, :2].ravel(), 1.5 * latent, rtol=1e-6)
        assert np.abs(latent).max() < 1.0


def test_a_seed_fixes_the_start_and_the_noise() -> None:
    first, second, other = FakeEnv(2, True), FakeEnv(2, True), FakeEnv(2, True)
    for env, seed in ((first, 7), (second, 7), (other, 8)):
        run_episode(env, FakePolicy(), None, Steering.BASE, CONFIG, seed=seed)
    assert first.starts == second.starts != other.starts
    np.testing.assert_array_equal(np.stack(first.actions), np.stack(second.actions))
    assert not np.array_equal(np.stack(first.actions), np.stack(other.actions))


def test_train_runs_baselines_warmup_and_periodic_evaluations() -> None:
    schedule = Schedule(
        warmup_episodes=2, train_episodes=5, eval_every=2, eval_episodes=3, learn_after=4
    )
    episodes: list[tuple[str, int]] = []
    evals: list[tuple[int, Steering, list[int]]] = []

    def on_episode(kind: str, report: EpisodeReport) -> None:
        episodes.append((kind, report.seed))

    def on_eval(done: int, steering: Steering, reports: list[EpisodeReport]) -> None:
        evals.append((done, steering, [r.seed for r in reports]))

    agent = FakeAgent()
    train(
        FakeEnv(3, True),
        FakeEnv(3, True),
        FakePolicy(),
        agent,
        CONFIG,
        schedule,
        on_episode,
        on_eval,
    )
    assert episodes == [("warmup", 0), ("warmup", 1)] + [("train", s) for s in range(2, 7)]
    seeds = eval_seeds(schedule)
    assert evals == [
        (0, Steering.BASE, seeds),
        (0, Steering.GREEDY, seeds),
        (2, Steering.GREEDY, seeds),
        (4, Steering.GREEDY, seeds),
        (5, Steering.GREEDY, seeds),
    ]
    assert len(agent.stored) == 7 * 3
    assert agent.updates > 0


OFFSET = SteeringConfig(layout=LAYOUT, magnitude=0.5, steer=Steer.OFFSET)


def test_an_offset_latent_shifts_the_frozen_policys_own_draw() -> None:
    base, steered = FakeEnv(3, True), FakeEnv(3, True)
    run_episode(base, FakePolicy(), None, Steering.BASE, OFFSET, seed=5)
    run_episode(steered, FakePolicy(), FakeAgent(latent=0.4), Steering.EXPLORE, OFFSET, seed=5)
    for drawn, shifted in zip(base.actions, steered.actions, strict=True):
        np.testing.assert_allclose(shifted[:2, :2] - drawn[:2, :2], 0.2, atol=1e-6)
        np.testing.assert_allclose(shifted[2:], drawn[2:], atol=1e-6)
        np.testing.assert_allclose(shifted[:, 2:], drawn[:, 2:], atol=1e-6)


def test_offset_warmup_stores_uniform_latents_and_keeps_the_draws_paired() -> None:
    base, warm = FakeEnv(4, True), FakeEnv(4, True)
    agent = FakeAgent()
    run_episode(base, FakePolicy(), None, Steering.BASE, OFFSET, seed=2)
    run_episode(warm, FakePolicy(), agent, Steering.WARMUP, OFFSET, seed=2)
    latents = np.stack([stored[1] for stored in agent.stored])
    assert np.abs(latents).max() < 1.0 and latents.std() > 0.3
    for (_, latent, *_), drawn, shifted in zip(
        agent.stored, base.actions, warm.actions, strict=True
    ):
        np.testing.assert_allclose(
            shifted[:2, :2].ravel() - drawn[:2, :2].ravel(), 0.5 * latent, atol=1e-6
        )
