import mujoco
import pytest

from vla_chess.sim.collision import GripContacts, grip_contacts

# Two static jaw spheres either side of a free-floating piece sphere. Radii sum to 20 mm, so a
# jaw placed nearer than that to the piece is in contact and one placed further away is not;
# MuJoCo only generates the pair because the piece carries a freejoint.
_JAW_TOUCHING = 0.018
_JAW_CLEAR = 0.05

_MODEL = """
<mujoco>
  <worldbody>
    <body name="piece" pos="0 0 0.5">
      <freejoint/>
      <geom name="piece_g" type="sphere" size="0.01"/>
    </body>
    <body name="gripper" pos="{fixed} 0 0.5">
      <geom name="fixed_g" type="sphere" size="0.01"/>
    </body>
    <body name="moving_jaw_so101_v1" pos="{jaw} 0 0.5">
      <geom name="jaw_g" type="sphere" size="0.01"/>
    </body>
  </worldbody>
</mujoco>
"""


def hand(fixed: float, jaw: float) -> GripContacts:
    """Grip verdict for a piece with the fixed finger at `-fixed` and the moving jaw at `jaw`."""
    model = mujoco.MjModel.from_xml_string(_MODEL.format(fixed=-fixed, jaw=jaw))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    piece = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "piece"))
    return grip_contacts(model, data, piece)


class TestGripContacts:
    def test_both_jaws_on_the_piece_is_a_grip(self) -> None:
        assert hand(_JAW_TOUCHING, _JAW_TOUCHING).is_gripped()

    def test_neither_jaw_on_the_piece_is_not_a_grip(self) -> None:
        contacts = hand(_JAW_CLEAR, _JAW_CLEAR)
        assert contacts == GripContacts(fixed_finger=False, moving_jaw=False)
        assert not contacts.is_gripped()

    # The failure this predicate exists for: the fixed finger rides on the `gripper` body and
    # was buried in the pawn while the moving jaw closed on nothing, and the old either-jaw
    # test called that a hold. See docs/trajectory.md.
    def test_fixed_finger_alone_is_not_a_grip(self) -> None:
        contacts = hand(_JAW_TOUCHING, _JAW_CLEAR)
        assert contacts == GripContacts(fixed_finger=True, moving_jaw=False)
        assert not contacts.is_gripped()

    def test_moving_jaw_alone_is_not_a_grip(self) -> None:
        contacts = hand(_JAW_CLEAR, _JAW_TOUCHING)
        assert contacts == GripContacts(fixed_finger=False, moving_jaw=True)
        assert not contacts.is_gripped()


class TestGripContactsVerdict:
    @pytest.mark.parametrize(
        ("fixed_finger", "moving_jaw"), [(False, False), (True, False), (False, True)]
    )
    def test_one_jaw_or_none_is_not_gripped(self, fixed_finger: bool, moving_jaw: bool) -> None:
        assert not GripContacts(fixed_finger, moving_jaw).is_gripped()

    def test_both_jaws_is_gripped(self) -> None:
        assert GripContacts(fixed_finger=True, moving_jaw=True).is_gripped()
