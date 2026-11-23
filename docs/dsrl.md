# DSRL: steering a frozen SmolVLA through its initial noise

DSRL (Wagenmaker et al., *Steering Your Diffusion Policy with Latent Space Reinforcement
Learning*, CoRL 2025, arXiv 2506.15799) leaves a trained flow-matching policy untouched and
learns which initial noise to hand its action expert. Everything the steered policy does is
something the frozen policy's prior can already produce; RL only chooses among it.

## Pieces

| Module | Role |
|---|---|
| `rl/env.py` | `MoveEnv`: the lone-pawn scene of the closed-loop eval, stepped one action chunk at a time. Renders the wrist camera only where a chunk starts. |
| `rl/noise.py` | `NoiseLayout`: which entries of the `(chunk_size, max_action_dim)` noise the agent sets, and what fills the rest. |
| `rl/steer.py` | `SteerablePolicy`: SmolVLA split into `encode` (VLM prefix + observation summary) and `decode` (integrate a given noise against that prefix). Tested to agree with `predict_action_chunk`. |
| `rl/agent.py` | `SteeringAgent`: LeRobot's SAC (`GaussianActorPolicy`, `SACAlgorithm`, `ReplayBuffer`) with the observation as one flat vector. |
| `rl/dsrl.py` | Episode loop in four steering modes, and the training schedule. |
| `cli/train_dsrl.py` | Wiring and logs; `runpod/dsrl.sh` runs it on a pod (`LAUNCH=dsrl`). |

## Choices

- **Action space.** The agent sets the noise rows of the executed steps (`--steered-steps`,
  default 10, the `n_action_steps` the policy runs at) and only the real action columns (6 of 32):
  a 60-dimensional latent in (-1, 1), scaled by `--magnitude` (1.5). The other rows are cut off
  after decoding and the other columns are unpadded away, so setting them would only add
  dimensions the critic has to learn to ignore. They are filled with zeros (`--fill zero`), which
  makes a chunk a function of the latent alone; `--fill gaussian` restores the expert's own draws.
- **Observation.** The VLM prefix's final hidden states averaged over valid tokens (960) plus the
  normalised joint state (6). Computing the prefix is needed for decoding anyway, so the agent's
  view of the observation costs nothing extra.
- **Reward.** Sparse: 1 when the piece stands on its target and the hand has let go, 0 otherwise,
  discounted 0.99 per chunk. Training episodes stop early on success and when the piece lies
  toppled (more than 45 degrees) or off the board, unheld.
- **Start states.** The scene is the same at every reset and the simulation is deterministic, so
  without variation a deterministic agent drives one trajectory. Each seeded episode draws the
  pawn's start within `--piece-jitter-mm` (3) of its square's centre.
- **Evaluation.** As the closed-loop eval scores a rollout: no early stop, judged on where the
  piece stands after 750 frames. Every evaluation drives the same seeded starts, first with the
  frozen policy (every noise entry a fresh Gaussian draw) and the untrained actor, then with the
  actor's mode.
- **Replay warm-up.** Warm-up episodes draw the steered block from the policy's own Gaussian prior
  and store the matching latents, so the replay starts with the frozen policy's behaviour.
