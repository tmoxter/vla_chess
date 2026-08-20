import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np

from vla_chess.control.executor import CONTROL_HZ, Executor
from vla_chess.control.planner import MoveSpec, plan_move, total_duration
from vla_chess.dataset.observe import DEFAULT_FPS, Frame, ObservationRecorder, image_is_sane
from vla_chess.grasp.solver import GRASP_HEIGHT, VerticalGraspSolver
from vla_chess.sensor.render import CameraSpec, FreeCameraPose, Renderer
from vla_chess.sim.positions import starting_position
from vla_chess.sim.scene import (
    BIRDS_EYE_ARM_QPOS,
    DEFAULT_BOARD_POSE,
    RENDER_OFFHEIGHT,
    RENDER_OFFWIDTH,
    WRIST_CAMERA_NAME,
    SceneDescription,
    build_scene,
    grip_radius,
    make_data,
    pinch_offset_for,
)
from vla_chess.viz.imaging import contact_sheet, downscale, write_png

DEFAULT_OUTPUT_DIR: Final = Path("docs/episode-views")
SHEET_COLUMNS: Final = 4
SHEET_PANELS: Final = 12
SHEET_DOWNSCALE: Final = 2

# A second camera, for people rather than for the policy. The wrist view is the observation and
# the dataset carries nothing else, but it is also the one view that cannot answer the question
# a pick-and-place episode is judged on — whether the piece came with the hand — because the
# hand fills it. This one watches from across the table and is written alongside.
WITNESS_CAMERA: Final = "witness"
WITNESS_EYE: Final = (0.10, -0.55, 0.30)
WITNESS_LOOKAT: Final = (0.30, 0.0, 0.10)


def sheet(frames: Sequence[Frame], panels: int, columns: int, camera: str) -> np.ndarray:
    """Lay out a sample of an episode's frames as one contact sheet.

    Args:
        frames: The episode's captured frames, in order.
        panels: How many to show, spread evenly across the episode. A shorter episode
            contributes every frame it has.
        columns: How many panels per row.
        camera: Output name of the camera to show.

    Returns:
        The sheet, each panel downscaled and captioned with its phase and timestamp.
    """
    picks = np.unique(np.linspace(0, len(frames) - 1, num=min(panels, len(frames))).round()).astype(
        int
    )
    tiles = [
        (
            f"{frames[i].phase.value} t={frames[i].timestamp:.2f}",
            downscale(frames[i].images[camera], SHEET_DOWNSCALE),
        )
        for i in picks
    ]
    return contact_sheet(tiles, columns)


def write_video(path: Path, frames: Sequence[Frame], fps: float, camera: str) -> bool:
    """Encode one camera's frames to an mp4 with ffmpeg.

    Args:
        path: File to write.
        frames: The episode's captured frames, in order; at least one is required.
        fps: Playback rate to stamp on the video.
        camera: Output name of the camera to encode.

    Returns:
        True once written. False if ffmpeg is not installed or exited non-zero, in which case
        its stderr is printed — an absent encoder is a missing convenience, not a failed
        episode.
    """
    first = frames[0].images[camera]
    height, width, _ = first.shape
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    try:
        process = subprocess.run(
            command,
            input=b"".join(frame.images[camera].tobytes() for frame in frames),
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    if process.returncode != 0:
        print(process.stderr.decode(errors="replace")[-2000:], file=sys.stderr)
        return False
    return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the single-episode generator's command line.

    Args:
        argv: Command line to parse, without the program name. `sys.argv` is read when
            omitted.

    Returns:
        The parsed options: which piece to move where, the capture and control rates, the
        seed, where the output goes, and which of the video, the onlooker's view and the
        per-frame PNGs to skip.
    """
    parser = argparse.ArgumentParser(
        prog="generate-episode",
        description="Plan, execute and record one pick-and-place episode, and write the wrist "
        "camera's view of it out for inspection.",
    )
    parser.add_argument("--source", default="d7")
    parser.add_argument("--target", default="d6")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--control-hz", type=float, default=CONTROL_HZ)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--no-witness",
        action="store_true",
        help="render only the wrist camera the dataset carries, not the onlooker's view",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="also write every captured frame as its own PNG, for frame-by-frame inspection",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Plan, execute and record one episode, and write out what it produced.

    Args:
        argv: Command line to parse, without the program name. `sys.argv` is read when
            omitted.

    Returns:
        Process exit status: 0 if the episode both gripped the piece and left it standing on
        its destination, 1 otherwise. The views and metadata are written either way, since a
        failed episode is the one worth looking at.
    """
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pieces = starting_position()
    occupant = {placement.square: placement.piece_type for placement in pieces}
    if args.source not in occupant:
        print(f"no piece on {args.source} to move", file=sys.stderr)
        return 1
    piece_type = occupant[args.source]

    # The pinch site is built for the piece this episode lifts: it is the middle of the jaws
    # when they are closed on that piece, so the piece ends up centred between the faces
    # instead of shoved against one of them.
    description = SceneDescription(
        board_pose=DEFAULT_BOARD_POSE,
        pieces=pieces,
        arm_qpos=BIRDS_EYE_ARM_QPOS,
        pinch_offset=pinch_offset_for(grip_radius(piece_type, GRASP_HEIGHT)),
    )
    model, index = build_scene(description)

    spec = MoveSpec(source=args.source, target=args.target, piece_type=piece_type)
    plan_data = make_data(model, index)
    solver = VerticalGraspSolver(model, plan_data, index)
    print(f"solving anchor poses for {spec.source} -> {spec.target}", flush=True)
    plan = plan_move(spec, model, plan_data, index, solver, np.random.default_rng(args.seed))
    print(
        f"planned {len(plan.waypoints)} waypoints, "
        f"{total_duration(plan.waypoints):.1f} s: {' '.join(p.value for p in plan.phases())}",
        flush=True,
    )

    data = make_data(model, index)
    executor = Executor(model, index, control_hz=args.control_hz)
    cameras = [CameraSpec(WRIST_CAMERA_NAME, RENDER_OFFWIDTH, RENDER_OFFHEIGHT)]
    if not args.no_witness:
        cameras.append(
            CameraSpec(
                WITNESS_CAMERA,
                RENDER_OFFWIDTH,
                RENDER_OFFHEIGHT,
                fovy=45,
                pose=FreeCameraPose(WITNESS_EYE, WITNESS_LOOKAT),
            )
        )
    with Renderer(model, cameras) as renderer:
        recorder = ObservationRecorder(renderer, fps=args.fps, control_hz=args.control_hz)
        print("executing", flush=True)
        result = executor.run(data, plan, on_control_step=recorder.on_control_step)
        frames = recorder.frames(result)

    print(
        f"{result.diagnostics.control_steps} control steps in "
        f"{result.diagnostics.wall_seconds:.1f} s, {len(frames)} frames captured",
        flush=True,
    )
    print(
        f"grasped={result.grasped} placed={result.placed} final_square={result.final_square} "
        f"abort={result.abort_reason.value if result.abort_reason else None}",
        flush=True,
    )
    print(
        f"max tracking error {result.diagnostics.max_tracking_error:.4f} rad, "
        f"max IK error {result.diagnostics.max_ik_error * 1000:.2f} mm",
        flush=True,
    )

    insane = [
        frame.frame_index for frame in frames if not image_is_sane(frame.images[WRIST_CAMERA_NAME])
    ]
    if insane:
        print(f"frames failing the image sanity check: {insane}", flush=True)

    if frames:
        stem = f"episode-{spec.source}-{spec.target}"
        for camera in (camera.name for camera in cameras):
            suffix = "" if camera == WRIST_CAMERA_NAME else f"-{camera}"
            sheet_path = args.out_dir / f"{stem}{suffix}.png"
            write_png(sheet_path, sheet(frames, SHEET_PANELS, SHEET_COLUMNS, camera))
            print(f"wrote {sheet_path}", flush=True)

            if not args.no_video:
                video_path = args.out_dir / f"{stem}{suffix}.mp4"
                if write_video(video_path, frames, args.fps, camera):
                    print(f"wrote {video_path}", flush=True)
                else:
                    print("ffmpeg unavailable or failed; skipped the video", flush=True)

        if args.frames_dir is not None:
            args.frames_dir.mkdir(parents=True, exist_ok=True)
            for frame in frames:
                write_png(
                    args.frames_dir / f"{frame.frame_index:04d}.png",
                    frame.images[WRIST_CAMERA_NAME],
                )
            print(f"wrote {len(frames)} frames to {args.frames_dir}", flush=True)

    metadata = {
        "instruction": plan.instruction,
        "source": spec.source,
        "target": spec.target,
        "board_pose": {"x": DEFAULT_BOARD_POSE.x, "yaw": DEFAULT_BOARD_POSE.yaw},
        "hover_height": spec.hover_height,
        "grasp_height": spec.grasp_height,
        "control_hz": args.control_hz,
        "fps": args.fps,
        "seed": args.seed,
        "frames": len(frames),
        "grasped": result.grasped,
        "placed": result.placed,
        "final_square": result.final_square,
        "abort_reason": result.abort_reason.value if result.abort_reason else None,
        "phase_boundaries": {
            phase.value: list(span) for phase, span in result.phase_boundaries.items()
        },
        "diagnostics": {
            "max_tracking_error": result.diagnostics.max_tracking_error,
            "max_ik_error": result.diagnostics.max_ik_error,
            "control_steps": result.diagnostics.control_steps,
            "sim_steps": result.diagnostics.sim_steps,
            "wall_seconds": result.diagnostics.wall_seconds,
        },
    }
    metadata_path = args.out_dir / f"episode-{spec.source}-{spec.target}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"wrote {metadata_path}", flush=True)
    return 0 if result.succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
