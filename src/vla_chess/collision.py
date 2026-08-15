"""Contact queries for a static arm configuration, split into robot-vs-world and self."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import mujoco
import numpy as np
import numpy.typing as npt

from vla_chess.scene import ARM_JOINT_NAMES, GRIPPER_BODY_NAME, SceneIndex

MOVING_JAW_BODY_NAME: Final = "moving_jaw_so101_v1"

# Contacts shallower than this are treated as touching rather than penetrating: the piece
# meshes are convex primitive stacks and the arm's collision proxies are coarse, so sub-tenth
# -millimetre overlaps are quantisation, not interference.
PENETRATION_TOLERANCE: Final = 1e-4

_BASELINE_SAMPLES: Final = 60


@dataclass(frozen=True)
class Contact:
    """One penetrating contact, named by the bodies involved."""

    robot_body: str
    other_body: str
    depth: float


@dataclass(frozen=True)
class PoseContacts:
    """Contacts at one arm configuration, separated by what the robot is touching."""

    world: tuple[Contact, ...]
    self_collisions: tuple[Contact, ...]

    def is_clear(self) -> bool:
        """Whether the pose is free of both world contacts and self-collisions."""
        return not self.world and not self.self_collisions


def robot_body_ids(model: mujoco.MjModel, index: SceneIndex) -> frozenset[int]:
    """Every body belonging to the arm, including the fixed base the shoulder mounts on.

    The base matters: rooting this at the first joint's body leaves base-vs-anything looking
    like a robot-vs-world contact, which mislabels the model's own base/shoulder overlap as
    an obstacle collision.
    """
    root = int(model.jnt_bodyid[index.arm_joint_id[ARM_JOINT_NAMES[0]]])
    bodies = {root, int(model.body_parentid[root])}
    for body in range(model.nbody):
        parent = body
        while parent != 0:
            if parent == root:
                bodies.add(body)
                break
            parent = int(model.body_parentid[parent])
    bodies.discard(0)
    return frozenset(bodies)


def hand_body_ids(model: mujoco.MjModel) -> frozenset[int]:
    """The gripper bodies allowed to touch the piece being grasped."""
    ids = [
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        for name in (GRIPPER_BODY_NAME, MOVING_JAW_BODY_NAME)
    ]
    return frozenset(i for i in ids if i >= 0)


# A rejection from this module is not proof of a collision. MuJoCo collides meshes as their
# convex hulls, and the arm's links are C-shaped printed parts whose hulls fill the cavity —
# measured, the shoulder bracket's hull is 3.9x its mesh volume and the gripper's 2.6x. On a
# folded pose that has been seen to report 12.7 mm of penetration where the true triangle meshes
# clear each other by ~0.1 mm. Treat a self-collision verdict as "look closer" rather than
# "impossible", and check the exact meshes before recording a square as ungraspable; see
# docs/grasp-poses.md. Contacts far from the arm's own body do not have this problem.
def baseline_self_pairs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    index: SceneIndex,
    rng: np.random.Generator,
    samples: int = _BASELINE_SAMPLES,
) -> frozenset[tuple[int, int]]:
    """Robot geom pairs that overlap in every sampled configuration.

    The vendored SO-101 asset declares no exclude pairs and its collision proxies overlap
    where the real links do not, so a self-contact only means something if it is not one of
    these. `data` is mutated in place.
    """
    robot = robot_body_ids(model, index)
    robot_geoms = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in robot}
    qadr = [model.jnt_qposadr[index.arm_joint_id[n]] for n in ARM_JOINT_NAMES]
    ids = [index.arm_joint_id[n] for n in ARM_JOINT_NAMES]
    lo = np.array([model.jnt_range[j][0] for j in ids])
    hi = np.array([model.jnt_range[j][1] for j in ids])

    counts: dict[tuple[int, int], int] = {}
    for i in range(samples):
        qpos = np.array(index.arm_qpos) if i == 0 else rng.uniform(lo, hi)
        data.qpos[qadr] = qpos
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        seen = set()
        for c in range(data.ncon):
            contact = data.contact[c]
            if contact.dist >= 0:
                continue
            pair = (int(contact.geom1), int(contact.geom2))
            if pair[0] in robot_geoms and pair[1] in robot_geoms:
                seen.add(pair)
        for pair in seen:
            counts[pair] = counts.get(pair, 0) + 1
    return frozenset(pair for pair, count in counts.items() if count == samples)


def contacts_at_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    index: SceneIndex,
    qpos: npt.NDArray[np.float64],
    grasped_square: str | None = None,
    baseline: frozenset[tuple[int, int]] = frozenset(),
    gripper_qpos: float | None = None,
) -> PoseContacts:
    """Penetrating contacts with the arm at `qpos`, ignoring the grasped piece and `baseline`.

    The hand touching the piece named by `grasped_square` is the point of a grasp, so those
    contacts are excluded; the same piece touched by any other link is not. `data` is mutated
    in place.
    """
    qadr = [model.jnt_qposadr[index.arm_joint_id[n]] for n in ARM_JOINT_NAMES]
    data.qpos[qadr] = qpos
    if gripper_qpos is not None:
        data.qpos[model.jnt_qposadr[index.gripper_joint_id]] = gripper_qpos
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)

    robot = robot_body_ids(model, index)
    hands = hand_body_ids(model)
    target_body = index.piece_body_id.get(grasped_square) if grasped_square else None

    def body_name(geom: int) -> str:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom]))
        return name if name else f"body{int(model.geom_bodyid[geom])}"

    world: list[Contact] = []
    self_hits: list[Contact] = []
    for c in range(data.ncon):
        contact = data.contact[c]
        depth = -float(contact.dist)
        if depth <= PENETRATION_TOLERANCE:
            continue
        g1, g2 = int(contact.geom1), int(contact.geom2)
        in1 = int(model.geom_bodyid[g1]) in robot
        in2 = int(model.geom_bodyid[g2]) in robot
        if not in1 and not in2:
            continue
        if in1 and in2:
            if (g1, g2) in baseline or (g2, g1) in baseline:
                continue
            self_hits.append(Contact(body_name(g1), body_name(g2), depth))
            continue
        mine, other = (g1, g2) if in1 else (g2, g1)
        grasping = target_body is not None and int(model.geom_bodyid[other]) == target_body
        if grasping and int(model.geom_bodyid[mine]) in hands:
            continue
        world.append(Contact(body_name(mine), body_name(other), depth))
    return PoseContacts(tuple(world), tuple(self_hits))
