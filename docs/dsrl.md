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

- **Action space.** The agent picks a latent in (-1, 1) for the noise rows of the executed steps
  (`--steered-steps`, default 10, the `n_action_steps` the policy runs at) and the real action
  columns (6 of 32): 60 dimensions. The other entries still shape the executed actions, since the
  expert attends across the whole chunk, but less strongly.
- **Offset steering** (`--steer offset`, the default). The expert's noise is drawn as it would draw
  it, and the latent shifts the steered block by at most `--magnitude` (0.5) standard deviations.
  The actor's mean starts at zero, so the untrained agent's mode is the frozen policy, and its
  exploration stays inside the noise the policy was trained on. `--steer replace` sets the block to
  `--magnitude` times the latent outright, as in the paper, with `--fill` for the rest.
- **Observation.** The VLM prefix's final hidden states averaged over valid tokens (960) plus the
  normalised joint state (6). Computing the prefix is needed for decoding anyway, so the agent's
  view of the observation costs nothing extra.
- **Reward.** Sparse, discounted 0.99 per chunk. With `--success done` (the default) an episode
  ends where the move is done (`control.done`: the hand let go and the arm stood at rest for
  half a second) and is rewarded 1 if the pawn stood on its target then; a move never done within
  `--max-frames` earns nothing. `--success placed` rewards the pawn standing on its target,
  released, at any chunk boundary. Training episodes also end once the piece lies toppled or off
  the board, unheld. `--grasp-bonus`, `--lift-bonus` and `--place-bonus` (all 0 by default) pay a
  training episode once each, on the chunk where the move first reaches that stage; evaluations
  never pay them.
- **Start states.** The scene is the same at every reset and the simulation is deterministic, so
  without variation a deterministic agent drives one trajectory. Each seeded episode draws the
  pawn's start within `--piece-jitter-mm` (3) of its square's centre.
- **Evaluation.** Every evaluation drives the same seeded starts, the frozen policy first. Under
  offset steering the noise of a seed is drawn identically in every mode, so the frozen policy's
  and the agent's episodes are paired. Scored at done, an evaluation stops once the move is done;
  scored as placed, it runs to `--max-frames` like the closed-loop eval.
- **Replay warm-up.** Under offset steering, warm-up episodes draw latents uniformly from the
  latent box; under replace steering, from the policy's own Gaussian prior.

## What replace steering did (v1)

`runpod/dsrl_v1.env` steered the round 1 policy on d7->d6 with `--steer replace`, magnitude 1.5,
zero fill, rewarded on placement. The frozen policy placed 30% of 50 jittered starts. The untrained
actor's mode placed 10%, the actor after 100 training episodes 6%, and after 200 episodes 0%,
with the arm never grasping, and the run was stopped there.

The latents were not extreme: on real observations the actor's mode had |a| about 0.3 per entry
and none near 1. What changed was the noise's randomness. A replaced block is the same latent at
every chunk wherever the observation is similar (the mode varied by 0.05-0.17 per entry across an
episode's observations), and the frozen policy depends on drawing fresh noise: decoded from all-zero
noise, it placed 10% against 30% from Gaussian draws. Each chunk moves the joints by no more than
about 0.07 rad under any of these noise choices, against 0.09 rad between two Gaussian draws, but
a bias that repeats chunk after chunk adds up over an episode: after 200 episodes the arm stopped
its descent 3-5 mm higher than the frozen policy's and missed the grasp. Exploration had the same
problem from the start (training placement fell from 40% in warm-up to about 20%), and with a
sparse reward and a few dozen successes the critics could not tell the actor which way back was.
Offset steering keeps the draw's randomness and starts from the frozen policy instead.

## What offset steering did (v2)

A run of 20 warm-up and 300 training episodes, evaluated every 100, steered the same policy with
`--steer offset`, magnitude 0.5, rewarded on the pawn standing on d6 at done. On 50 paired starts the frozen policy succeeded 30% and the untrained agent 28%, the same outcome on 49 of 50. After 100 and 200 training episodes the agent succeeded 20%, and after 300 episodes 28%.
Training episodes, with exploration, succeeded 15-25%.

The policy stayed in distribution: success never collapsed as under replace steering. Nor did it
rise. Against the frozen policy the agent after 100 and after 200 episodes lost 11 successes and
gained 6 on the same starts: a small shift of the noise changes which episodes succeed rather than
how many. With one reward per success, earned in 4 to 8 of every 25 training episodes and only at
the end of a 50-75 chunk episode, the critics have little to tell good shifts from bad ones, while
most failures (the grasp missed, or the pawn knocked over) are decided in the first 20-30 chunks.
The staged reward below pays for those early stages.

## What the staged reward did (v3)

A run on the same seeds as v2 added a 0.1 bonus for the first grasp, 0.2 for the first lift and 0.3 for the first placement. Over its first 100 training
episodes it succeeded 17 times against v2's 16. After 100 episodes the agent grasped and lifted
as often as the frozen policy on the same 50 starts (27 and 26 against 28 and 26), but placed 9
against 18 and succeeded at done 10% against 30%: it kept what the bonuses paid for and lost the
put-down. The run was stopped there.

Neither run is long enough to tell steering that helps from steering that does not: a few hundred
episodes, with a success in a quarter of them, leave the critics too little to learn from.
