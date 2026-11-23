from pathlib import Path

import numpy as np
import pytest

from vla_chess.rl.agent import AgentConfig, SteeringAgent

CONFIG = AgentConfig(hidden_dims=(32, 32), encoder_dim=16, num_critics=4, batch_size=8)


def _agent() -> SteeringAgent:
    return SteeringAgent(feature_dim=12, latent_dim=6, config=CONFIG, device="cpu")


def _fill(agent: SteeringAgent, count: int) -> None:
    rng = np.random.default_rng(0)
    for i in range(count):
        agent.remember(
            rng.standard_normal(12),
            rng.uniform(-1, 1, 6),
            float(i % 2),
            rng.standard_normal(12),
            terminated=i % 5 == 4,
            truncated=False,
        )


def test_latents_lie_inside_the_unit_box() -> None:
    agent = _agent()
    features = np.random.default_rng(0).standard_normal(12)
    for explore in (True, False):
        latent = agent.act(features, explore=explore)
        assert latent.shape == (6,) and latent.dtype == np.float32
        assert np.abs(latent).max() < 1.0


def test_the_mode_is_deterministic_and_samples_are_not() -> None:
    agent = _agent()
    features = np.random.default_rng(0).standard_normal(12)
    np.testing.assert_array_equal(agent.act(features, False), agent.act(features, False))
    assert not np.array_equal(agent.act(features, True), agent.act(features, True))


def test_an_update_needs_a_full_batch() -> None:
    agent = _agent()
    _fill(agent, 7)
    with pytest.raises(RuntimeError, match="fewer than a batch"):
        agent.update()


def test_updates_report_finite_losses_and_move_the_actor() -> None:
    agent = _agent()
    _fill(agent, 40)
    features = np.zeros(12)
    before = agent.act(features, False)
    for _ in range(5):
        stats = agent.update()
    assert {"loss_critic", "loss_actor", "loss_temperature", "temperature"} <= set(stats)
    assert all(np.isfinite(v) for v in stats.values())
    assert not np.array_equal(before, agent.act(features, False))


def test_save_and_load_keep_the_actor(tmp_path: Path) -> None:
    agent = _agent()
    _fill(agent, 40)
    agent.update()
    agent.save(tmp_path / "agent.pt")
    loaded = SteeringAgent.load(tmp_path / "agent.pt", device="cpu")
    features = np.random.default_rng(1).standard_normal(12)
    np.testing.assert_array_equal(agent.act(features, False), loaded.act(features, False))
    assert loaded.algorithm.optimization_step == agent.algorithm.optimization_step
    assert len(loaded) == 0
