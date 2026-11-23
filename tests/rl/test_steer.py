import os
from collections.abc import Iterator

import numpy as np
import pytest

from vla_chess.control.planner import MoveSpec
from vla_chess.rl.env import EnvConfig, MoveEnv
from vla_chess.rl.noise import Fill
from vla_chess.rl.steer import SteerablePolicy
from vla_chess.sim.scene import PieceType

CHECKPOINT = os.environ.get("VLA_CHESS_SMOLVLA_CHECKPOINT")

pytestmark = pytest.mark.skipif(
    CHECKPOINT is None, reason="set VLA_CHESS_SMOLVLA_CHECKPOINT to a pretrained_model directory"
)

MOVE = MoveSpec(
    source="d7",
    target="d6",
    piece_type=PieceType.PAWN,
    instruction_override="move the pawn one field forward",
)


@pytest.fixture(scope="module")
def policy() -> SteerablePolicy:
    assert CHECKPOINT is not None
    return SteerablePolicy(CHECKPOINT, device="cpu", n_action_steps=10)


@pytest.fixture(scope="module")
def env() -> Iterator[MoveEnv]:
    env = MoveEnv(MOVE, EnvConfig())
    yield env
    env.close()


def test_decode_agrees_with_predict_action_chunk(policy: SteerablePolicy, env: MoveEnv) -> None:
    observation = env.reset()
    noise = np.random.default_rng(0).standard_normal((50, 32)).astype(np.float32)
    steered = policy.decode(policy.encode(observation), noise)
    reference = policy.reference_chunk(observation, noise)
    assert steered.shape == (10, 6)
    np.testing.assert_allclose(steered, reference, rtol=1e-4, atol=1e-4)


def test_features_have_the_advertised_size(policy: SteerablePolicy, env: MoveEnv) -> None:
    encoding = policy.encode(env.reset())
    assert encoding.features.shape == (policy.feature_dim,)
    assert np.isfinite(encoding.features).all()


def test_the_noise_steers_the_chunk(policy: SteerablePolicy, env: MoveEnv) -> None:
    encoding = policy.encode(env.reset())
    layout = policy.layout(10)
    rng = np.random.default_rng(0)
    low = policy.decode(encoding, layout.noise(-np.ones(layout.latent_dim), 1.5, Fill.ZERO, rng))
    high = policy.decode(encoding, layout.noise(np.ones(layout.latent_dim), 1.5, Fill.ZERO, rng))
    again = policy.decode(encoding, layout.noise(np.ones(layout.latent_dim), 1.5, Fill.ZERO, rng))
    np.testing.assert_allclose(high, again, rtol=1e-5, atol=1e-5)
    assert np.abs(high - low).max() > 1e-2
