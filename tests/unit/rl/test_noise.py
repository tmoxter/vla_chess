import numpy as np
import pytest

from vla_chess.rl.noise import LATENT_LIMIT, Fill, NoiseLayout

LAYOUT = NoiseLayout(chunk_size=50, max_action_dim=32, action_dim=6, steered_steps=10)


def test_latent_dim_counts_the_steered_block() -> None:
    assert LAYOUT.latent_dim == 60


def test_latent_fills_the_steered_block_row_by_row() -> None:
    latent = np.linspace(-1.0, 1.0, LAYOUT.latent_dim)
    noise = LAYOUT.noise(latent, 1.5, Fill.ZERO, np.random.default_rng(0))
    assert noise.shape == (50, 32)
    assert noise.dtype == np.float32
    np.testing.assert_allclose(noise[:10, :6], 1.5 * latent.reshape(10, 6), rtol=1e-6)
    assert noise[0, 1] == pytest.approx(1.5 * latent[1])


def test_zero_fill_leaves_everything_outside_the_block_zero() -> None:
    noise = LAYOUT.noise(np.ones(LAYOUT.latent_dim), 1.0, Fill.ZERO, np.random.default_rng(0))
    assert not noise[10:].any()
    assert not noise[:, 6:].any()


def test_gaussian_fill_draws_outside_the_block_and_keeps_the_block() -> None:
    latent = np.full(LAYOUT.latent_dim, 0.5)
    noise = LAYOUT.noise(latent, 2.0, Fill.GAUSSIAN, np.random.default_rng(0))
    np.testing.assert_allclose(noise[:10, :6], 1.0)
    rest = noise[10:].ravel()
    assert abs(rest.mean()) < 0.05
    assert rest.std() == pytest.approx(1.0, abs=0.05)


def test_latent_of_inverts_noise_inside_the_magnitude() -> None:
    latent = np.random.default_rng(1).uniform(-0.9, 0.9, LAYOUT.latent_dim)
    noise = LAYOUT.noise(latent, 1.5, Fill.GAUSSIAN, np.random.default_rng(2))
    np.testing.assert_allclose(LAYOUT.latent_of(noise, 1.5), latent, rtol=1e-5, atol=1e-6)


def test_latent_of_clips_noise_beyond_the_magnitude() -> None:
    noise = np.zeros((50, 32))
    noise[0, 0] = 10.0
    noise[0, 1] = -10.0
    latent = LAYOUT.latent_of(noise, 1.0)
    assert latent[0] == pytest.approx(LATENT_LIMIT)
    assert latent[1] == pytest.approx(-LATENT_LIMIT)


def test_noise_rejects_a_latent_of_the_wrong_size() -> None:
    with pytest.raises(ValueError, match="latent has shape"):
        LAYOUT.noise(np.zeros(59), 1.0, Fill.ZERO, np.random.default_rng(0))


@pytest.mark.parametrize(("action_dim", "steered_steps"), [(0, 10), (33, 10), (6, 0), (6, 51)])
def test_layout_rejects_blocks_outside_the_chunk(action_dim: int, steered_steps: int) -> None:
    with pytest.raises(ValueError):
        NoiseLayout(
            chunk_size=50, max_action_dim=32, action_dim=action_dim, steered_steps=steered_steps
        )
