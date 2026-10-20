"""Open-loop action-error probe: at each ORACLE state, how far is the policy's predicted joint
target from the oracle's action? This measures the teacher-forced-style error WITHOUT closed-loop
compounding, so it separates "the model never learned the mapping" from "small per-step errors
compound in rollout".

For each sampled oracle frame we reset the policy (so every prediction is a fresh inference from
that exact state, not a queued action), feed the oracle's wrist image + 6-D state + the task, and
compare the first predicted action to the oracle's recorded action. Reported per arm-joint and for
the gripper, split into approach / transport / place thirds of the trajectory.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

from vla_chess.cli.eval_rollout import _load_policy, _observation
from vla_chess.cli.generate_dataset import SINGLE_PAWN_INSTRUCTION
from vla_chess.control.planner import MoveSpec
from vla_chess.dataset.episode import WRIST_CAMERA, EpisodeRunner
from vla_chess.sim.positions import single_pawn_position
from vla_chess.sim.scene import PieceType


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="c7")
    ap.add_argument("--target", default="c6")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--repo-id", default="hugginggeorge/smolvla-chess-exp2")
    ap.add_argument("--revision", default="020000")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stride", type=int, default=5, help="probe every Nth oracle frame")
    args = ap.parse_args()

    import torch

    spec = MoveSpec(
        source=args.source,
        target=args.target,
        piece_type=PieceType.PAWN,
        instruction_override=SINGLE_PAWN_INSTRUCTION,
    )
    pieces = single_pawn_position(args.source)
    ep = EpisodeRunner(fps=args.fps, pieces=pieces).run(spec, seed=0)
    frames = ep.frames
    print(
        f"[probe] {args.source}->{args.target} rev={args.revision}: oracle {len(frames)} frames, "
        f"succeeded={ep.succeeded}",
        flush=True,
    )

    policy, pre, post = _load_policy(args.repo_id, args.revision, args.device)
    instr = SINGLE_PAWN_INSTRUCTION

    idxs, arm_l2, grip_err, perjoint = [], [], [], []
    for i in range(0, len(frames), args.stride):
        f = frames[i]
        img = np.asarray(f.images[WRIST_CAMERA.name])
        state = np.asarray(f.state, dtype=np.float32)
        obs = _observation(img, state, instr)
        policy.reset()  # fresh inference from THIS state, not a queued chunk action
        with torch.inference_mode():
            pred = post(policy.select_action(pre(obs))).squeeze(0).to(torch.float32).cpu().numpy()
        oracle = np.asarray(f.action, dtype=np.float32)
        idxs.append(i)
        arm_l2.append(float(np.linalg.norm(pred[:5] - oracle[:5])))
        grip_err.append(float(abs(pred[5] - oracle[5])))
        perjoint.append(np.abs(pred[:5] - oracle[:5]))

    arm = np.array(arm_l2)
    grip = np.array(grip_err)
    pj = np.stack(perjoint)
    n = len(arm)
    thirds = {
        "approach": arm[: n // 3],
        "transport": arm[n // 3 : 2 * n // 3],
        "place": arm[2 * n // 3 :],
    }
    print(f"\n=== OPEN-LOOP ACTION ERROR ({n} probed states, stride {args.stride}) ===", flush=True)
    print(
        f"arm-joint L2 (rad):  mean={arm.mean():.3f}  median={np.median(arm):.3f}  "
        f"p95={np.percentile(arm, 95):.3f}  max={arm.max():.3f}",
        flush=True,
    )
    print(f"gripper abs err:     mean={grip.mean():.3f}  max={grip.max():.3f}", flush=True)
    print(
        "by phase (mean arm L2, rad): "
        + "  ".join(f"{k}={v.mean():.3f}" for k, v in thirds.items()),
        flush=True,
    )
    print(
        "per-joint mean abs err (rad): "
        + "  ".join(f"j{j}={pj[:, j].mean():.3f}" for j in range(5)),
        flush=True,
    )
    worst = int(np.argmax(arm))
    print(f"worst state: frame {idxs[worst]} arm-L2={arm[worst]:.3f}", flush=True)


if __name__ == "__main__":
    main()
