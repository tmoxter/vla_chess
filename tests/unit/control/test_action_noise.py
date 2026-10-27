"""The DART action-noise process: its scale, its correlation, its schedule, and its recovery."""

import itertools

import numpy as np
import pytest

from vla_chess.control.executor import DART_PHASE_GAINS, ActionNoise, ProgressGate
from vla_chess.control.planner import Phase


def _noise(sigma: float = 0.05, tau: float = 0.3, seed: int = 0) -> ActionNoise:
    return ActionNoise(sigma, tau, control_hz=50.0, rng=np.random.default_rng(seed))


def test_ungated_offset_has_the_requested_stationary_std() -> None:
    noise = _noise(sigma=0.05)
    draws = np.stack([noise.sample(Phase.SURVEY_START, 0.0) for _ in range(20000)])
    assert np.allclose(draws.std(axis=0), 0.05, rtol=0.15)


def test_offset_is_correlated_from_one_step_to_the_next() -> None:
    noise = _noise(tau=0.3)
    draws = np.stack([noise.sample(Phase.SURVEY_START, 0.0)[0] for _ in range(20000)])
    lag1 = np.corrcoef(draws[:-1], draws[1:])[0, 1]
    # exp(-1 / (0.3 s * 50 Hz)) ~ 0.94: a drift that persists, not per-step jitter.
    assert lag1 == pytest.approx(np.exp(-1.0 / 15.0), abs=0.02)


@pytest.mark.parametrize("phase", list(Phase))
def test_every_phase_is_perturbed(phase: Phase) -> None:
    noise = _noise()
    assert np.any(noise.sample(phase, 0.5))


def test_adjacent_phases_meet_at_nearly_the_same_gain() -> None:
    for before, after in itertools.pairwise(Phase):
        assert DART_PHASE_GAINS[before][1] == pytest.approx(DART_PHASE_GAINS[after][0], abs=0.05)


def test_a_recovering_step_relaxes_the_offset_without_a_kick() -> None:
    noise = _noise()
    previous = noise.sample(Phase.SURVEY_START, 0.0)
    relaxed = noise.sample(Phase.SURVEY_START, 0.0, recovering=True)
    assert np.allclose(relaxed, np.exp(-1.0 / (0.1 * 50.0)) * previous)


def test_same_seed_reproduces_the_same_draws() -> None:
    a, b = _noise(seed=3), _noise(seed=3)
    for _ in range(10):
        assert np.array_equal(a.sample(Phase.TRANSPORT, 0.2), b.sample(Phase.TRANSPORT, 0.2))


def test_rejects_a_non_positive_time_constant() -> None:
    with pytest.raises(ValueError):
        ActionNoise(0.05, 0.0, control_hz=50.0, rng=np.random.default_rng(0))


def test_rejects_a_non_positive_recovery_time_constant() -> None:
    with pytest.raises(ValueError):
        ActionNoise(0.05, 0.3, 50.0, np.random.default_rng(0), recovery_tau=0.0)


def test_gate_judges_the_board_plane_and_the_vertical_separately() -> None:
    gate = ProgressGate(lateral=0.002, vertical=0.005)
    assert gate.admits(np.array([0.001, 0.001, 0.004]))
    assert not gate.admits(np.array([0.002, 0.002, 0.0]))
    assert not gate.admits(np.array([0.0, 0.0, 0.006]))
