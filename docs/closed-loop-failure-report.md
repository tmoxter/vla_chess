# Closed-loop failure: findings from the funnel + diverse-funnel fine-tunes

**Date:** 2026-10-22
**Task:** single pawn, one-square forward push, single constant instruction, SmolVLA
fine-tune, evaluated by closed-loop rollout in the MuJoCo sim.
**Headline:** two successive SmolVLA fine-tunes both learn the open-loop map almost
perfectly (train loss ~0.02) yet **cannot reliably execute the move closed-loop**. The
cause is covariate shift at the grasp, and it is **not** fixed by more data diversity,
longer/again training, frame rate, or action-chunk / smoothing knobs at inference.

---

## Experiment 1 — funnel pool (seeds + squares, minimal diversity)

- **Data:** `chess-pawns-funnel-v1` — the sample-then-filter funnel over the one-step
  move pool (all files, 7→6 / 6→5 / rank-4 where reachable), `--seeds` jittered repeats
  per move, single constant task label, 30 fps.
- **Diversity present:** only the per-slot *seed* (IK-restart reseeding). Slots of the
  same move are near-identical trajectories — same centred placement, minimum-tilt grasp.
- **Training:** SmolVLA fine-tune, batch 32, 20k steps, **final train loss ~0.017**.
- **Open-loop check:** predicted-vs-oracle action error is tiny — **~0.025 rad (~1.4°)**
  arm-joint L2, uniform across approach/transport/place. The state→action map is learned.
- **Closed-loop result: fails.** Checkpoint sweep (8k/12k/16k/20k) all fail on in-sample
  `c7→c6`; held-out `d6→d5` also fails. Not late-overfitting — no checkpoint rolls out
  better.
- **Frame rate ruled out.** The eval harness is fps-aware (30 fps → 17 sim substeps per
  action; the old hard-coded 50 was for the 10 fps overfit and would overshoot 3×).
  Failure persists at the data's native rate, so it is not an fps/substep mismatch.
- **Harness validated.** `oracle_replay.py` pushes the oracle's own trajectory through the
  identical eval stepping and it places `c7→c6` cleanly — so the failure is the policy,
  not the tooling.
- **Diagnosis:** classic **covariate shift / compounding error** — ~1–3°/step errors
  accumulate over ~480 autonomous steps and drift off the demo manifold; the grasp is the
  sensitive point.

## Experiment 2 — diverse funnel (real randomization, retrain, inference sweeps)

**Hypothesis:** Exp-1's data was too uniform — seed repeats alone are *not* diversity.
Add genuine coverage and retrain.

- **What changed in the data (`--diverse`):** instead of repeating the centred baseline,
  each slot (a) draws a **random board pose** in the Step-4 envelope (xy ±3 cm, yaw ±15°)
  and (b) keeps a **random *successful* candidate** (sampled grasp lean + placement
  offset) rather than the clean baseline. → `chess-pawns-funnel-diverse-v1`, 100 episodes
  / 48.3k frames (one-step pool, single task, 30 fps, seeds 6; 20/120 slots yielded no
  keeper — the honest tilted-grasp yield).
- **Finding on "more seeds":** confirmed that simply adding seeds is a weak diversity
  axis — the real spread had to come from board-pose + grasp/placement randomization.
- **Training:** SmolVLA fine-tune, batch 16 (16 GB A4000), 20k steps to convergence,
  **final train loss 0.022** (0.040 at the 10k midpoint). Clean convergence.
- **Closed-loop result: still fails.**
  - Rollouts are **stochastic** (flow-matching action expert samples fresh noise each
    query), so single draws are noisy — the model *occasionally* lands the move.
  - **Multi-seed success-rate sweep** (72 rollouts: `n_action_steps` ∈ {10, 25, 50} ×
    moves {d7d6, c7c6, e6e5, g7g6} × 6 reps):

    | n_action_steps | success rate |
    |---|---|
    | 10 | 1/24 (4%) |
    | 25 | 0/24 (0%) |
    | 50 | 1/24 (4%) |

    Only `d7→d6` ever succeeded; the other three were 0/6 at every setting. ~0–4% floor.
- **Action-length / smoothing ruled out.** Sweeping `n_action_steps` (inference-only, no
  retrain) changes nothing for success (4/0/4% indistinguishable). Shrinking it + EMA
  **crushed motion jitter ~9×** (peak step-delta 0.156 → 0.017 rad) but did *not* improve
  placement — the smoothest variants all failed, and the lone success was the *jittery*
  default. Jitter magnitude does not predict success; the piece ends on its **start
  square** because the grasp never closes. Smoothness and success are decoupled.
- **Per-square difficulty is noise here.** `d7→d6` (the only occasional success) had the
  *least* training data (3/6 kept) while `e6→e5` had the most (6/6) and never succeeded —
  so coverage doesn't explain the per-square pattern; with 6 samples/square it's noise on
  the ~0–4% floor. Geometrically the hard zone in this rig is the near ranks + outer files
  (extreme wrist tilt), not the far home rank.

---

## Conclusion & what's exhausted

Both fine-tunes prove the data and training pipeline are sound (loss collapses, open-loop
error ~1°) but the policy **does not survive its own closed-loop rollout**. The following
levers have now been tried and ruled out as the fix:

- ❌ More data volume / more seeds (Exp-1 → uniform; weak axis)
- ❌ More *open-loop* diversity — board-pose + grasp/placement randomization (Exp-2)
- ❌ Longer/again training (converges, same failure)
- ❌ Frame rate / substep cadence (harness fps-correct; oracle replay places)
- ❌ Action-chunk horizon `n_action_steps` and motion smoothing (inference knobs)

**The wall is covariate shift.** The remaining remedies are **training-side, on the error
distribution** rather than more clean demos:

1. **On-policy / DAgger-style correction** — roll the policy out, have the oracle label the
   states it actually visits (especially the drifted, near-grasp states), add those to the
   training set. Directly attacks the distribution the policy fails on.
2. **Grasp/recovery-state coverage** — demos that start from perturbed, off-manifold poses
   around the grasp so the policy learns to recover instead of compounding error.
3. (Optional, cosmetic only) **temporal ensembling** for smooth motion in demos — does not
   address success.

Artifacts: final model `runpod/artifacts/smolvla-chess-exp2-step20000/`; clips + sweep log
+ eval scripts under `runpod/artifacts/`; dataset `hugginggeorge/chess-pawns-funnel-diverse-v1`.
