"""The steering agent: soft actor-critic over a frozen policy's noise latent, built from LeRobot.

The actor reads the frozen policy's summary of an observation and picks a latent in [-1, 1] per
steered noise entry; the critics score that choice. Actor, critics, temperature and replay come
from LeRobot's SAC (`GaussianActorPolicy`, `SACAlgorithm`, `ReplayBuffer`), with the observation
handed over as one flat state vector.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import (
    ActorNetworkConfig,
    CriticNetworkConfig,
    GaussianActorConfig,
)
from lerobot.policies.gaussian_actor.modeling_gaussian_actor import GaussianActorPolicy
from lerobot.rl.algorithms.sac import SACAlgorithmConfig
from lerobot.rl.algorithms.sac.sac_algorithm import SACAlgorithm
from lerobot.rl.buffer import ReplayBuffer
from lerobot.utils.constants import ACTION, OBS_STATE


@dataclass(frozen=True)
class AgentConfig:
    """Soft actor-critic hyperparameters for the steering agent."""

    hidden_dims: tuple[int, ...] = (512, 512, 512)  # actor and each critic
    encoder_dim: int = 256  # width the shared observation encoder projects features to
    num_critics: int = 10
    num_subsample_critics: int = 2  # critics the TD target takes its minimum over
    discount: float = 0.99  # per chunk
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temperature_lr: float = 3e-4
    temperature_init: float = 0.1
    target_entropy: float | None = None  # None: -latent_dim / 2
    batch_size: int = 256
    buffer_capacity: int = 200_000


class SteeringAgent:
    """A SAC actor and critic ensemble over a frozen policy's noise latent, with its replay."""

    def __init__(self, feature_dim: int, latent_dim: int, config: AgentConfig, device: str) -> None:
        """Build the actor, critics, optimizers and an empty replay buffer.

        Args:
            feature_dim: Size of the frozen policy's observation summary.
            latent_dim: Size of the latent the actor picks.
            config: Hyperparameters.
            device: Torch device the networks train on; the replay stays on the CPU.
        """
        self.config = config
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.device = torch.device(device)
        hidden = list(config.hidden_dims)
        actor_config = GaussianActorConfig(
            input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(feature_dim,))},
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(latent_dim,))},
            dataset_stats=None,
            device=device,
            latent_dim=config.encoder_dim,
            actor_network_kwargs=ActorNetworkConfig(hidden_dims=hidden),
        )
        self.policy = GaussianActorPolicy(actor_config)
        self.algorithm = SACAlgorithm(
            self.policy,
            SACAlgorithmConfig(
                policy_config=actor_config,
                actor_lr=config.actor_lr,
                critic_lr=config.critic_lr,
                temperature_lr=config.temperature_lr,
                discount=config.discount,
                num_critics=config.num_critics,
                num_subsample_critics=config.num_subsample_critics,
                critic_network_kwargs=CriticNetworkConfig(hidden_dims=hidden),
                temperature_init=config.temperature_init,
                target_entropy=config.target_entropy,
            ),
        )
        self.algorithm.make_optimizers_and_scheduler()
        self.buffer = ReplayBuffer(
            config.buffer_capacity,
            device=device,
            state_keys=[OBS_STATE],
            image_augmentation_function=_identity,
            use_drq=False,
            storage_device="cpu",
        )
        self._batches: Iterator[Any] | None = None

    def __len__(self) -> int:
        return len(self.buffer)

    def act(self, features: npt.NDArray[np.floating], explore: bool) -> npt.NDArray[np.float32]:
        """Pick a latent for one observation.

        Args:
            features: The frozen policy's summary of the observation, shape `(feature_dim,)`.
            explore: Sample from the actor's squashed Gaussian; otherwise take its mode.

        Returns:
            The latent, shape `(latent_dim,)`, in (-1, 1).
        """
        with torch.no_grad():
            observation = {OBS_STATE: self._tensor(features).unsqueeze(0)}
            sampled, _, means = self.policy.actor(observation, None)
            latent = sampled if explore else torch.tanh(means)
            picked: npt.NDArray[np.float32] = latent.squeeze(0).cpu().numpy().astype(np.float32)
        return picked

    def remember(
        self,
        features: npt.NDArray[np.floating],
        latent: npt.NDArray[np.floating],
        reward: float,
        next_features: npt.NDArray[np.floating],
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Store one chunk's transition.

        Args:
            features: Observation summary where the chunk started.
            latent: Latent the chunk was decoded from.
            reward: Reward the chunk earned.
            next_features: Observation summary where the next chunk would start.
            terminated: The episode ended on success or loss; the TD target does not bootstrap.
            truncated: The frame budget ran out; the TD target still bootstraps.
        """
        self.buffer.add(
            state={OBS_STATE: _cpu(features).unsqueeze(0)},
            action=_cpu(latent).unsqueeze(0),
            reward=float(reward),
            next_state={OBS_STATE: _cpu(next_features).unsqueeze(0)},
            done=bool(terminated),
            truncated=bool(truncated),
        )

    def update(self) -> dict[str, float]:
        """Run one SAC step on a replay batch.

        Returns:
            Losses, gradient norms and the temperature, flattened into one dict.

        Raises:
            RuntimeError: If the replay holds fewer transitions than one batch.
        """
        if len(self.buffer) < self.config.batch_size:
            raise RuntimeError(
                f"replay holds {len(self.buffer)} transitions, fewer than a batch of "
                f"{self.config.batch_size}"
            )
        if self._batches is None:
            self._batches = self.buffer.get_iterator(
                batch_size=self.config.batch_size, async_prefetch=False, queue_size=1
            )
        stats = self.algorithm.update(self._batches)
        flat = {key: float(value) for key, value in stats.losses.items()}
        flat.update({f"grad_{key}": float(value) for key, value in stats.grad_norms.items()})
        flat.update({key: float(value) for key, value in stats.extra.items()})
        return flat

    def save(self, path: Path) -> None:
        """Write the actor, critics, temperature and optimizer states to one file.

        Args:
            path: File to write; its directory is created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "feature_dim": self.feature_dim,
                "latent_dim": self.latent_dim,
                "config": asdict(self.config),
                "policy": self.policy.state_dict(),
                "algorithm": self.algorithm.state_dict(),
                "optimizers": {k: o.state_dict() for k, o in self.algorithm.optimizers.items()},
                "optimization_step": self.algorithm.optimization_step,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str) -> SteeringAgent:
        """Rebuild an agent from a file `save` wrote, with an empty replay.

        Args:
            path: File to read.
            device: Torch device to place the networks on.

        Returns:
            The agent, ready to act or keep training.
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        config = state["config"]
        config["hidden_dims"] = tuple(config["hidden_dims"])
        agent = cls(state["feature_dim"], state["latent_dim"], AgentConfig(**config), device)
        agent.policy.load_state_dict(state["policy"])
        agent.algorithm.load_state_dict(state["algorithm"], device=device)
        for key, optimizer in agent.algorithm.optimizers.items():
            optimizer.load_state_dict(state["optimizers"][key])
        agent.algorithm.optimization_step = state["optimization_step"]
        return agent

    def _tensor(self, values: npt.NDArray[np.floating]) -> torch.Tensor:
        """Put an array on the agent's device as float32."""
        return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=self.device)


def _cpu(values: npt.NDArray[np.floating]) -> torch.Tensor:
    """Copy an array into a float32 CPU tensor."""
    return torch.as_tensor(np.asarray(values), dtype=torch.float32)


def _identity(images: torch.Tensor) -> torch.Tensor:
    """Leave a batch of images unchanged."""
    return images
