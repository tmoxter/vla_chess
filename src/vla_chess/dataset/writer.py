"""Append recorded episodes to a LeRobotDataset v3.0 on disk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from vla_chess.dataset.episode import Episode
from vla_chess.sim.scene import ARM_JOINT_NAMES

DEFAULT_REPO_ID: Final = "vla_chess/chess_pick_place"
DEFAULT_ROBOT_TYPE: Final = "so101"

# The six numbers a state and an action carry, in the order the executor writes them: the five
# arm joints, then the jaw. These name the state/action feature columns in the dataset.
STATE_FEATURE_NAMES: Final = (*ARM_JOINT_NAMES, "gripper")


@dataclass(frozen=True)
class DatasetStats:
    """What a finished dataset holds."""

    num_episodes: int
    num_frames: int


def episode_features(ep: Episode) -> dict[str, dict[str, Any]]:
    """Derive the LeRobot feature schema from an episode's first frame.

    Every camera the frames carry becomes a video feature under `observation.images.<name>`,
    sized to the images actually captured; the proprioceptive state and the action become
    float32 vectors named by their joints.

    Args:
        ep: The episode to read the schema off; its frames set the image resolution and the
            state and action widths.

    Returns:
        A features mapping in the shape `LeRobotDataset.create` expects.

    Raises:
        ValueError: If the episode has no frames, or its state and action width does not match
            the joint names.
    """
    if not ep.frames:
        raise ValueError("cannot derive a schema from an episode with no frames")
    frame = ep.frames[0]
    features: dict[str, dict[str, Any]] = {}
    for name, image in frame.images.items():
        height, width, channels = image.shape
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": (int(height), int(width), int(channels)),
            "names": ["height", "width", "channels"],
        }
    for key, vector in (("observation.state", frame.state), ("action", frame.action)):
        width = int(vector.shape[0])
        if width != len(STATE_FEATURE_NAMES):
            raise ValueError(
                f"{key} has width {width}, but {len(STATE_FEATURE_NAMES)} joint names are "
                f"defined: {STATE_FEATURE_NAMES}"
            )
        features[key] = {
            "dtype": "float32",
            "shape": (width,),
            "names": list(STATE_FEATURE_NAMES),
        }
    return features


class LeRobotWriter:
    """Writes episodes into one LeRobotDataset, creating it from the first episode's schema.

    The dataset is created lazily on the first `add_episode` so the schema and frame rate come
    from the episodes themselves rather than being declared twice. Every episode after the first
    must match that schema — same cameras, same resolution, same state and action widths.
    """

    def __init__(
        self,
        root: Path,
        *,
        repo_id: str = DEFAULT_REPO_ID,
        robot_type: str = DEFAULT_ROBOT_TYPE,
    ) -> None:
        """Bind the writer to an output directory, without touching it yet.

        Args:
            root: Directory the dataset is written into. It must not already hold one — this
                writer creates, it does not resume.
            repo_id: LeRobot repository id stamped into the dataset's metadata.
            robot_type: Robot identifier recorded in the dataset's metadata.
        """
        self._root = Path(root)
        self._repo_id = repo_id
        self._robot_type = robot_type
        self._dataset: Any = None

    def add_episode(self, ep: Episode) -> int:
        """Append one episode's frames and commit it as a new episode.

        The episode's instruction is attached to every frame as its task, and the per-frame
        state, action and camera images are written straight through.

        Args:
            ep: The episode to write. It must carry at least one frame.

        Returns:
            The index the episode was written at, counting from zero.

        Raises:
            ValueError: If the episode has no frames.
        """
        if not ep.frames:
            raise ValueError("cannot write an episode with no frames")
        if self._dataset is None:
            # LeRobot encodes the video at an integer frame rate (it hands fps straight to the
            # muxer as a stream rate), so a fractional capture rate has no home in the dataset.
            if not ep.fps.is_integer():
                raise ValueError(f"the dataset frame rate must be a whole number, got {ep.fps}")
            self._dataset = LeRobotDataset.create(
                repo_id=self._repo_id,
                fps=int(ep.fps),
                features=episode_features(ep),
                root=self._root,
                robot_type=self._robot_type,
                use_videos=True,
            )
        dataset = self._dataset
        for frame in ep.frames:
            payload: dict[str, Any] = {
                f"observation.images.{name}": image for name, image in frame.images.items()
            }
            payload["observation.state"] = np.asarray(frame.state, dtype=np.float32)
            payload["action"] = np.asarray(frame.action, dtype=np.float32)
            payload["task"] = ep.instruction
            dataset.add_frame(payload)
        dataset.save_episode()
        return int(dataset.num_episodes) - 1

    def finalize(self) -> DatasetStats:
        """Flush the dataset's metadata and report what it holds.

        `save_episode` leaves per-episode metadata in an in-memory buffer; without this flush a
        small run writes no episode metadata and the dataset looks written but is invalid.

        Returns:
            The episode and frame counts. Both are zero if no episode was ever added.
        """
        if self._dataset is None:
            return DatasetStats(0, 0)
        self._dataset.finalize()
        return DatasetStats(int(self._dataset.num_episodes), int(self._dataset.num_frames))
