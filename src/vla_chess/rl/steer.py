"""A frozen SmolVLA policy whose action chunks are integrated from noise the caller chooses.

SmolVLA computes an action chunk in two parts: the vision-language model reads the observation
into a key/value prefix, then the action expert integrates an initial noise draw into actions
against that prefix. `SteerablePolicy` runs the two parts separately, so a steering agent can see
a summary of the observation before it picks the noise, without the prefix being computed twice.
The summary is the prefix's final hidden states averaged over its valid tokens, followed by the
normalised joint state.

This reaches into the LeRobot 0.6.1 SmolVLA internals (`embed_prefix`, `denoise_step` and the
VLM-with-expert forward); `predict_action_chunk` is the reference it must agree with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from lerobot.policies import make_pre_post_processors
from lerobot.policies.common.flow_matching import euler_integrate
from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from vla_chess.rl.env import Observation
from vla_chess.rl.noise import NoiseLayout

# The wrist view was renamed to `camera1` at training time; SmolVLA pads the views it is not given.
CAMERA_KEY = "observation.images.camera1"


@dataclass
class Encoding:
    """One observation read into the policy's prefix, ready to be decoded from any noise."""

    features: npt.NDArray[np.float32]  # the steering agent's view of the observation
    prefix_pad_masks: torch.Tensor
    past_key_values: Any


class SteerablePolicy:
    """A frozen SmolVLA checkpoint, split into reading an observation and decoding a chunk."""

    def __init__(self, checkpoint: str, device: str, n_action_steps: int) -> None:
        """Load a checkpoint and its processors, frozen and in eval mode.

        Args:
            checkpoint: Path to a `pretrained_model` directory.
            device: Torch device to run on.
            n_action_steps: Leading actions of each decoded chunk that are returned for execution.

        Raises:
            ValueError: If the checkpoint executes more steps per chunk than it predicts, or uses
                the pi-ALOHA action space.
        """
        config = SmolVLAConfig.from_pretrained(checkpoint)
        config.device = device
        config.use_amp = False
        if config.adapt_to_pi_aloha:
            raise ValueError("pi-ALOHA action adaptation is not supported")
        if not 0 < n_action_steps <= config.chunk_size:
            raise ValueError(f"n_action_steps {n_action_steps} exceeds chunk {config.chunk_size}")
        config.n_action_steps = n_action_steps
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config)
        self.policy.eval()
        self.policy.to(device)
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)
        self._pre, self._post = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.device = torch.device(device)
        self.config = config
        self.action_dim = int(config.action_feature.shape[0])
        self.n_action_steps = n_action_steps
        self.feature_dim = int(self.policy.model.vlm_with_expert.config.text_config.hidden_size)
        self.feature_dim += int(config.input_features[OBS_STATE].shape[0])

    def layout(self, steered_steps: int) -> NoiseLayout:
        """Describe this policy's initial noise with its first `steered_steps` rows steered."""
        return NoiseLayout(
            chunk_size=self.config.chunk_size,
            max_action_dim=self.config.max_action_dim,
            action_dim=self.action_dim,
            steered_steps=steered_steps,
        )

    def batch(self, observation: Observation) -> dict[str, Any]:
        """Turn an observation into the preprocessed batch the policy reads.

        Args:
            observation: Wrist frame, joint state and instruction.

        Returns:
            The tokenised, normalised batch of one, on the policy's device.
        """
        image = torch.from_numpy(observation.frame).to(torch.float32).permute(2, 0, 1) / 255.0
        raw = {
            OBS_STATE: torch.from_numpy(observation.state).to(torch.float32).unsqueeze(0),
            CAMERA_KEY: image.unsqueeze(0),
            "task": [observation.instruction],
        }
        return dict(self._pre(raw))

    def encode(self, observation: Observation) -> Encoding:
        """Read an observation into the policy's prefix and summarise it.

        Args:
            observation: Where the chunk starts.

        Returns:
            The prefix cache to decode against, and the features a steering agent sees.
        """
        with torch.no_grad():
            batch = self.policy._prepare_batch(self.batch(observation))
            images, img_masks = self.policy.prepare_images(batch)
            state = self.policy.prepare_state(batch)
            model = self.policy.model
            embs, pad_masks, att_masks = model.embed_prefix(
                images,
                img_masks,
                batch[OBS_LANGUAGE_TOKENS],
                batch[OBS_LANGUAGE_ATTENTION_MASK],
                state=state,
            )
            att_2d = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            outputs, past_key_values = model.vlm_with_expert.forward(
                attention_mask=att_2d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[embs, None],
                use_cache=self.config.use_cache,
            )
            hidden = outputs[0].to(torch.float32)
            weights = pad_masks.to(torch.float32).unsqueeze(-1)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
            features = torch.cat([pooled, batch[OBS_STATE].to(torch.float32)], dim=-1)
            return Encoding(
                features=features.squeeze(0).cpu().numpy(),
                prefix_pad_masks=pad_masks,
                past_key_values=past_key_values,
            )

    def decode(
        self, encoding: Encoding, noise: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.float64]:
        """Integrate an initial noise draw into the actions the arm executes.

        Args:
            encoding: The observation's prefix, from `encode`.
            noise: Initial noise, shape `(chunk_size, max_action_dim)`.

        Returns:
            The first `n_action_steps` actions in the dataset's units, shape
            `(n_action_steps, action_dim)`.
        """
        with torch.no_grad():
            x = torch.as_tensor(
                np.asarray(noise), dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            model = self.policy.model
            chunk = euler_integrate(
                lambda x_t, t: model.denoise_step(
                    x_t=x_t,
                    prefix_pad_masks=encoding.prefix_pad_masks,
                    past_key_values=encoding.past_key_values,
                    timestep=t,
                ),
                x,
                self.config.num_steps,
            )
            chunk = chunk[:, : self.n_action_steps, : self.action_dim]
            steps = [self._post(chunk[:, i]).to(torch.float32) for i in range(self.n_action_steps)]
            actions: npt.NDArray[np.float64] = (
                torch.cat(steps, dim=0).cpu().numpy().astype(np.float64)
            )
        return actions

    def reference_chunk(
        self, observation: Observation, noise: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.float64]:
        """Compute the executed actions through SmolVLA's own `predict_action_chunk`.

        Args:
            observation: Where the chunk starts.
            noise: Initial noise, shape `(chunk_size, max_action_dim)`.

        Returns:
            The first `n_action_steps` actions in the dataset's units, as `decode` returns them.
        """
        with torch.no_grad():
            self.policy.reset()
            x = torch.as_tensor(
                np.asarray(noise), dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            chunk = self.policy.predict_action_chunk(self.batch(observation), noise=x)
            chunk = chunk[:, : self.n_action_steps]
            steps = [self._post(chunk[:, i]).to(torch.float32) for i in range(self.n_action_steps)]
            actions: npt.NDArray[np.float64] = (
                torch.cat(steps, dim=0).cpu().numpy().astype(np.float64)
            )
        return actions
