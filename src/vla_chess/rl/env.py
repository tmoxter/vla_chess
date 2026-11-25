"""One pawn move as an environment a policy drives a chunk of actions at a time.

The scene, the cadence and the scoring are those of the closed-loop policy eval: the lone pawn on
its source square, one action per dataset frame stepped at the frame's share of simulated time,
and success read as the piece standing on its target. What this adds is the episode structure an
RL loop needs: the wrist frame is only rendered where a chunk starts, a reward is handed out, and
the episode can end early once the move has succeeded or can no longer succeed. Success is read
either as soon as the piece stands on its target released, or where the move is done by
`control.done`: the arm let go and came back to rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import mujoco
import numpy as np
import numpy.typing as npt

from vla_chess.control.done import DoneDetector
from vla_chess.control.executor import Executor
from vla_chess.control.planner import MoveSpec
from vla_chess.dataset.episode import WRIST_CAMERA, move_scene_description
from vla_chess.dataset.takeover import HIGH_RELEASE, GripTrace, mid_air_releases
from vla_chess.sensor.render import Renderer
from vla_chess.sim.positions import single_pawn_position
from vla_chess.sim.scene import BIRDS_EYE_ARM_QPOS, DEFAULT_BOARD_POSE, build_scene, make_data

# How far above the board surface the piece must rise, while both jaws bear on it, to count the
# grasp as a *lift* rather than a jaws-closed-but-piece-still-grounded touch. One centimetre clears
# the settle tolerance so a piece merely nudged on its square is not read as lifted.
LIFT_CLEARANCE_M = 0.01

# Tilt past which a piece nobody holds is lying down rather than rocking on its base.
TOPPLED_TILT_DEG = 45.0


class Success(Enum):
    """When an episode counts as a success."""

    PLACED = "placed"  # the piece stands on its target and the hand has let go of it
    DONE = "done"  # the move is done by `control.done` with the piece standing on its target


@dataclass(frozen=True)
class EnvConfig:
    """How an episode is paced and when it ends.

    With both early stops off, an episode is scored exactly as a closed-loop eval rollout: on
    where the piece stands after `max_frames` frames.
    """

    fps: float = 30.0  # dataset frames per simulated second; one action spans one frame
    max_frames: int = 750
    success: Success = Success.PLACED
    stop_on_success: bool = True  # end once the move succeeded, or under DONE once it is done
    stop_on_lost: bool = True  # end once the piece lies toppled or off the board, unheld
    # Half-width of the square window, in metres, the piece's start is drawn from around its
    # square's centre at every seeded reset. Without it every episode starts from the same state.
    piece_jitter_m: float = 0.0
    # Reward paid once, on the chunk where the episode first reaches each milestone, on top of
    # the success reward.
    grasp_bonus: float = 0.0
    lift_bonus: float = 0.0
    place_bonus: float = 0.0  # the piece stands on its target, released


@dataclass(frozen=True)
class Milestones:
    """The stages of a move an episode has reached so far."""

    grasped: bool = False
    lifted: bool = False
    placed: bool = False  # the piece stood on its target, released, at a chunk boundary


def milestone_bonus(before: Milestones, after: Milestones, config: EnvConfig) -> float:
    """Sum the bonuses of the milestones first reached between two chunk boundaries.

    Args:
        before: Milestones reached by the start of the chunk.
        after: Milestones reached by its end.
        config: The bonus each milestone pays.

    Returns:
        The bonus the chunk earns.
    """
    bonus = 0.0
    if after.grasped and not before.grasped:
        bonus += config.grasp_bonus
    if after.lifted and not before.lifted:
        bonus += config.lift_bonus
    if after.placed and not before.placed:
        bonus += config.place_bonus
    return bonus


@dataclass(frozen=True)
class Observation:
    """What the policy sees where a chunk starts."""

    frame: npt.NDArray[np.uint8]  # wrist camera, (H, W, 3)
    state: npt.NDArray[np.float64]  # five arm joints in radians, then the jaw on [0, 100]
    instruction: str


@dataclass(frozen=True)
class Outcome:
    """How an episode went, scored as the closed-loop eval scores a rollout."""

    grasped: bool  # both jaws ever bore on the piece
    lifted: bool  # ... while the piece was clear of the board
    placed: bool  # the piece stands upright on the target square
    settled: bool
    final_square: str | None
    opened_mid_air: bool  # the jaws were opened on the piece higher than HIGH_RELEASE
    slipped_mid_air: bool  # the piece left jaws still commanded closed, higher than HIGH_RELEASE
    lost: bool  # the piece lay toppled or off the board, unheld, when the episode ended
    frames: int
    done_frame: int | None  # frame the move became done on by `control.done`, None if never
    placed_at_done: bool  # the piece stood on its target on that frame


@dataclass(frozen=True)
class StepResult:
    """What one chunk of actions led to."""

    observation: Observation  # where the next chunk would start
    reward: float
    terminated: bool  # the move succeeded or can no longer succeed
    truncated: bool  # the frame budget ran out first
    outcome: Outcome | None  # set once the episode has ended


def end_of_chunk(
    frames: int,
    placed_and_released: bool,
    lost: bool,
    config: EnvConfig,
    done: bool = False,
    placed_at_done: bool = False,
) -> tuple[float, bool, bool]:
    """Decide the reward and whether the episode ends, at a chunk boundary.

    Args:
        frames: Frames driven so far.
        placed_and_released: The piece stands on its target and neither jaw bears on it.
        lost: The piece lies toppled or off the board, and nothing holds it.
        config: The episode's pacing and stopping rules.
        done: The move has become done by `control.done`, at this chunk or earlier.
        placed_at_done: The piece stood on its target on the frame the move became done.

    Returns:
        The reward, then whether the episode terminated and whether it was truncated. A success
        is rewarded 1 and everything else 0, whether it ends the episode early or is read off
        the last frame. Under `Success.DONE` a move that never became done is no success.
    """
    if config.success is Success.DONE:
        reward = 1.0 if done and placed_at_done else 0.0
        if config.stop_on_success and done:
            return reward, True, False
        if config.stop_on_lost and lost:
            return 0.0, True, False
        if frames >= config.max_frames:
            return reward, False, True
        return 0.0, False, False
    if config.stop_on_success and placed_and_released:
        return 1.0, True, False
    if config.stop_on_lost and lost:
        return 0.0, True, False
    if frames >= config.max_frames:
        return (1.0 if placed_and_released else 0.0), False, True
    return 0.0, False, False


class MoveEnv:
    """The lone-pawn scene for one move, driven a chunk of actions at a time.

    The model and renderer are built once; every reset puts the scene back at rest.
    """

    def __init__(self, spec: MoveSpec, config: EnvConfig) -> None:
        """Build the move's scene and open the wrist camera on it.

        Args:
            spec: The move every episode drives.
            config: Pacing and stopping rules.
        """
        self.spec = spec
        self.config = config
        description = move_scene_description(
            spec, single_pawn_position(spec.source), DEFAULT_BOARD_POSE, BIRDS_EYE_ARM_QPOS
        )
        self._model, self._index = build_scene(description)
        self._executor = Executor(self._model, self._index)
        self._renderer = Renderer(self._model, [WRIST_CAMERA])
        self._piece = self._index.piece_body_id[spec.source]
        self._surface = self._index.board.surface_height
        self.frame_substeps = max(1, round(1.0 / (config.fps * self._model.opt.timestep)))
        self._data = make_data(self._model, self._index)
        self._frames = 0
        self._ended = False
        self._grasped = False
        self._lifted = False
        self._held: list[bool] = []
        self._heights: list[float] = []
        self._jaw: list[float] = []
        self._done = DoneDetector()
        self._placed_at_done = False
        self._milestones = Milestones()

    def close(self) -> None:
        """Release the renderer's GL contexts."""
        self._renderer.close()

    @property
    def model(self) -> mujoco.MjModel:
        """The scene model."""
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        """The live simulated state."""
        return self._data

    def reset(self, rng: np.random.Generator | None = None) -> Observation:
        """Put the scene back at rest and observe where the first chunk starts.

        Args:
            rng: Source of the piece's start offset within its square; the piece starts on the
                square's centre when None or when the config allows no jitter.

        Returns:
            The wrist view and joint state the first chunk starts from.
        """
        self._data = make_data(self._model, self._index)
        if rng is not None and self.config.piece_jitter_m > 0.0:
            jitter = self.config.piece_jitter_m
            adr = int(self._model.jnt_qposadr[self._model.body_jntadr[self._piece]])
            self._data.qpos[adr : adr + 2] += rng.uniform(-jitter, jitter, size=2)
            mujoco.mj_forward(self._model, self._data)
        self._reset_trace()
        return self._observe()

    def piece_position(self) -> npt.NDArray[np.float64]:
        """World position of the piece being moved, in metres."""
        return np.array(self._data.xpos[self._piece], copy=True)

    def step(self, actions: npt.NDArray[np.floating]) -> StepResult:
        """Drive one chunk of actions, one per frame, and score where it leaves the move.

        A chunk that would run past the frame budget is cut at it.

        Args:
            actions: Commands in execution order, shape (k, 6): the five arm joints in radians,
                then the jaw on its [0, 100] closed-to-open scale.

        Returns:
            The observation the next chunk starts from, the reward, and how the episode stands.

        Raises:
            RuntimeError: If the episode has already ended.
        """
        if self._ended:
            raise RuntimeError("the episode has ended; reset before stepping again")
        for action in np.asarray(actions, dtype=np.float64):
            if self._frames >= self.config.max_frames:
                break
            self._drive_frame(action)
        holding = self._executor._holding(self._data, self._piece)
        check = self._executor.placement(self._data, self.spec.source, self.spec.target)
        lost = not holding and self._lost()
        done = self._done.done_frame is not None
        reward, terminated, truncated = end_of_chunk(
            self._frames,
            check.placed and not holding,
            lost,
            self.config,
            done=done,
            placed_at_done=self._placed_at_done,
        )
        reached = Milestones(
            grasped=self._grasped,
            lifted=self._lifted,
            placed=self._milestones.placed or (check.placed and not holding),
        )
        reward += milestone_bonus(self._milestones, reached, self.config)
        self._milestones = reached
        outcome = None
        if terminated or truncated:
            self._ended = True
            outcome = self._outcome(check.placed, check.settled, check.final_square, lost)
        return StepResult(self._observe(), reward, terminated, truncated, outcome)

    def _drive_frame(self, action: npt.NDArray[np.float64]) -> None:
        """Command one action, advance one frame and record the grip on the piece."""
        self._executor._write_ctrl(self._data, action[:5], float(action[5]))
        for _ in range(self.frame_substeps):
            mujoco.mj_step(self._model, self._data)
        holding = self._executor._holding(self._data, self._piece)
        height = float(self._data.xpos[self._piece][2]) - self._surface
        self._held.append(holding)
        self._heights.append(height)
        self._jaw.append(float(self._data.ctrl[self._index.gripper_actuator_id]))
        if holding:
            self._grasped = True
            self._lifted = self._lifted or height > LIFT_CLEARANCE_M
        if self._done.done_frame is None and self._done.update(
            self._executor._arm_qpos(self._data), holding
        ):
            check = self._executor.placement(self._data, self.spec.source, self.spec.target)
            self._placed_at_done = bool(check.placed)
        self._frames += 1

    def _lost(self) -> bool:
        """Report whether the piece lies toppled or has left the board."""
        upright = float(self._data.xmat[self._piece].reshape(3, 3)[2, 2])
        toppled = upright < float(np.cos(np.radians(TOPPLED_TILT_DEG)))
        square = self._index.board.world_to_square(
            self._data.xpos[self._piece], self._index.board_pose
        )
        return toppled or square is None

    def _observe(self) -> Observation:
        """Render the wrist camera and read the arm's state."""
        frame = np.array(self._renderer.render(self._data)[WRIST_CAMERA.name], copy=True)
        return Observation(
            frame=frame,
            state=self._executor._state(self._data),
            instruction=self.spec.instruction(),
        )

    def _outcome(
        self, placed: bool, settled: bool, final_square: str | None, lost: bool
    ) -> Outcome:
        """Summarise the episode that just ended."""
        releases = mid_air_releases(
            GripTrace(
                gripped=np.array(self._held, dtype=bool),
                height=np.array(self._heights),
                jaw_command=np.array(self._jaw),
            ),
            HIGH_RELEASE,
        )
        return Outcome(
            grasped=self._grasped,
            lifted=self._lifted,
            placed=placed,
            settled=settled,
            final_square=final_square,
            opened_mid_air=any(r.opened for r in releases),
            slipped_mid_air=any(not r.opened for r in releases),
            lost=lost,
            frames=self._frames,
            done_frame=self._done.done_frame,
            placed_at_done=self._placed_at_done,
        )

    def _reset_trace(self) -> None:
        """Forget the grip trace and counters of the previous episode."""
        self._frames = 0
        self._ended = False
        self._grasped = False
        self._lifted = False
        self._held = []
        self._heights = []
        self._jaw = []
        self._done.reset()
        self._placed_at_done = False
        self._milestones = Milestones()
