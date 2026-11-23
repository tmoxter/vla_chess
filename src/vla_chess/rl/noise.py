"""The initial noise a flow-matching action expert integrates from, and the part of it RL steers.

A SmolVLA action chunk starts from a `(chunk_size, max_action_dim)` draw of standard-normal noise
that the expert integrates into actions. Steering the frozen expert means choosing that draw
instead of sampling it. Only the rows the arm executes and the columns that are real action
dimensions are worth choosing: the remaining rows are discarded once the chunk is cut to its
executed steps, and the remaining columns are padding the expert unpads away. The layout here
says which entries are chosen, and fills the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

# Largest latent entry `latent_of` returns: atanh stays finite, so a squashed Gaussian can score it.
LATENT_LIMIT = 0.999


class Fill(Enum):
    """What the noise entries outside the steered block are set to."""

    ZERO = "zero"  # the mode of the expert's prior; makes a chunk a function of the latent alone
    GAUSSIAN = "gaussian"  # a fresh standard-normal draw, as the expert samples them itself


@dataclass(frozen=True)
class NoiseLayout:
    """Which entries of an action chunk's initial noise a steering latent sets.

    The steered block is the first `steered_steps` rows and the first `action_dim` columns; the
    latent fills it row by row.
    """

    chunk_size: int  # rows the expert integrates, one per action in its chunk
    max_action_dim: int  # columns, the action dimension the expert pads to
    action_dim: int  # leading columns that are real action dimensions
    steered_steps: int  # leading rows the latent sets

    def __post_init__(self) -> None:
        if not 0 < self.action_dim <= self.max_action_dim:
            raise ValueError(f"action_dim {self.action_dim} must lie in [1, {self.max_action_dim}]")
        if not 0 < self.steered_steps <= self.chunk_size:
            raise ValueError(
                f"steered_steps {self.steered_steps} must lie in [1, {self.chunk_size}]"
            )

    @property
    def latent_dim(self) -> int:
        """Number of entries a steering latent sets."""
        return self.steered_steps * self.action_dim

    def noise(
        self,
        latent: npt.NDArray[np.floating],
        magnitude: float,
        fill: Fill,
        rng: np.random.Generator,
    ) -> npt.NDArray[np.float32]:
        """Expand a steering latent into the expert's full initial noise.

        Args:
            latent: The steered entries, shape `(latent_dim,)`, each in [-1, 1].
            magnitude: Noise value a latent entry of 1 stands for; the steered block spans
                [-magnitude, magnitude].
            fill: What the entries outside the steered block are set to.
            rng: Source of the Gaussian fill; not drawn from for `Fill.ZERO`.

        Returns:
            The noise, shape `(chunk_size, max_action_dim)`.

        Raises:
            ValueError: If the latent is not `(latent_dim,)`.
        """
        latent = np.asarray(latent)
        if latent.shape != (self.latent_dim,):
            raise ValueError(f"latent has shape {latent.shape}, expected ({self.latent_dim},)")
        shape = (self.chunk_size, self.max_action_dim)
        if fill is Fill.GAUSSIAN:
            noise = rng.standard_normal(shape).astype(np.float32)
        else:
            noise = np.zeros(shape, dtype=np.float32)
        block = magnitude * latent.reshape(self.steered_steps, self.action_dim)
        noise[: self.steered_steps, : self.action_dim] = block
        return noise

    def latent_of(
        self, noise: npt.NDArray[np.floating], magnitude: float
    ) -> npt.NDArray[np.float32]:
        """Read the steering latent that reproduces a noise draw's steered block.

        Entries beyond the magnitude are clipped just inside [-1, 1], where a tanh-squashed actor
        can still assign them a finite likelihood.

        Args:
            noise: An initial noise draw, shape `(chunk_size, max_action_dim)`.
            magnitude: Noise value a latent entry of 1 stands for.

        Returns:
            The latent, shape `(latent_dim,)`.
        """
        block = np.asarray(noise)[: self.steered_steps, : self.action_dim] / magnitude
        return np.clip(block, -LATENT_LIMIT, LATENT_LIMIT).reshape(-1).astype(np.float32)
