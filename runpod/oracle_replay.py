"""Harness-vs-model isolation: does the ORACLE's own trajectory place the piece when pushed through
the eval harness's manual stepping (`_write_ctrl` + `substeps_per_frame` substeps/frame)?

1. Run the oracle (planner/Executor) for the move at `--fps`, record its per-frame actions and its
   own native success verdict.
2. Rebuild the identical scene and replay those exact actions through the eval-harness step loop,
   then score placement the same way.

If the oracle is native-SUCCESS but eval-replay FAILS, the eval harness's stepping (e.g. the fps
substep count) is the problem, not the policy. If both succeed, the harness is faithful and the
policy's closed-loop failure is real.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np

from vla_chess.cli.eval_rollout import substeps_per_frame, write_mp4
from vla_chess.cli.generate_dataset import SINGLE_PAWN_INSTRUCTION
from vla_chess.control.executor import Executor
from vla_chess.control.planner import MoveSpec
from vla_chess.dataset.episode import (
    WRIST_CAMERA,
    EpisodeRunner,
    move_scene_description,
)
from vla_chess.sensor.render import Renderer
from vla_chess.sim.positions import single_pawn_position
from vla_chess.sim.scene import (
    BIRDS_EYE_ARM_QPOS,
    DEFAULT_BOARD_POSE,
    PieceType,
    build_scene,
    make_data,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="c7")
    ap.add_argument("--target", default="c6")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", type=Path, default=Path("/root/rollouts/oracle-replay-c7c6.mp4"))
    args = ap.parse_args()

    spec = MoveSpec(
        source=args.source,
        target=args.target,
        piece_type=PieceType.PAWN,
        instruction_override=SINGLE_PAWN_INSTRUCTION,
    )
    pieces = single_pawn_position(args.source)

    # 1) Oracle native run.
    runner = EpisodeRunner(fps=args.fps, pieces=pieces)
    ep = runner.run(spec, seed=0)
    actions = [np.asarray(f.action, dtype=np.float32) for f in ep.frames]
    print(
        f"[oracle-native] {args.source}->{args.target} fps={args.fps}: "
        f"{len(actions)} frames, succeeded={ep.succeeded}",
        flush=True,
    )

    # 2) Replay those actions through the eval-harness stepping.
    description = move_scene_description(spec, pieces, DEFAULT_BOARD_POSE, BIRDS_EYE_ARM_QPOS)
    model, index = build_scene(description)
    data = make_data(model, index)
    renderer = Renderer(model, [WRIST_CAMERA])
    executor = Executor(model, index)
    fsub = substeps_per_frame(args.fps)

    frames = []
    for act in actions:
        frames.append(np.array(renderer.render(data)[WRIST_CAMERA.name], copy=True))
        executor._write_ctrl(data, act[:5], float(act[5]))
        for _ in range(fsub):
            mujoco.mj_step(model, data)
    check = executor.placement(data, args.source, args.target)
    print(
        f"[eval-replay] fsub={fsub}/frame: ended {check.final_square} "
        f"settled={check.settled} placed={check.placed}",
        flush=True,
    )

    if write_mp4(frames, args.out, args.fps):
        print(f"wrote {args.out} ({len(frames)} frames @ {args.fps} fps)", flush=True)

    verdict = (
        "HARNESS OK (oracle places via eval stepping)"
        if (ep.succeeded and check.placed)
        else "HARNESS SUSPECT (oracle native success but eval-replay fails)"
        if ep.succeeded and not check.placed
        else "ORACLE itself did not place — inconclusive for harness"
    )
    print(f"=== {verdict} ===", flush=True)


if __name__ == "__main__":
    main()
