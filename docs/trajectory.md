# Single-episode trajectory

`pixi run generate-episode` plans, executes and records one pick-and-place episode and writes it
to `docs/episode-views/`: the wrist camera, which is the observation the dataset carries, and a
witness view from across the table, which is the only one that can show whether the piece came
with the hand. Every number below comes from that task on this host, or from the probes
described alongside it; none are estimates.

**The episode picks the piece up and puts it down.** `d7 -> d6`, a black centre pawn one square
forward: grasped, lifted, carried, placed and released, ending upright on `d6` with the arm back
at the survey pose. What it took is the second half of this document; the first half is the four
failures that came before it, kept because each one was invisible until the thing after it broke.

Black, not white: the squares with a collision-free fingers-down pose are all on ranks 5-8
(`assets/reference_grasps.json`), so a white pawn's `d2 -> d3` has no grasp pose at either end.
The plan's Step 7 acceptance criterion still names `d2 -> d4`; that test cannot pass as written.

## Shape of the plan

Ten legs: `survey_start approach descend grasp lift transport place release retreat survey_end`,
16.1 s at 50 Hz control, captured at 10 fps.

The descent stops beside the piece rather than over it, and the grasp leg slides in; the release
backs out the same way before the retreat lifts. Both are `planner.GRASP_STANDOFF`, 5 mm along
the jaws' own closing direction, and both exist because only one finger moves — see *Only one
finger moves* below.

The episode opens and closes at `BIRDS_EYE_ARM_QPOS` rather than at the resting pose, because
the wrist camera is the only observation and at rest the arm is blind to ranks 7-8 — its own
home ranks, and the ones this move happens on. The survey pose frames all 64 squares, so frame 0
carries the board state the instruction refers to and the closing frames carry the result. It
costs two legs and ~4 s of episode.

The two survey transits are interpolated in joint space, the rest in Cartesian space with
warm-started fingers-down IK per control step. Interpolating the transits in Cartesian space
instead would impose the fingers-down constraint from the first control step, which turns the
survey pose's wrist orientation into a step change rather than a motion.

## Hover height is capped at ~70 mm, not by clearance

Fingers-down poses over the arm's home ranks exist only close to the surface. Solved with the
collision screening the reference grasps use, best attempt per height:

| Height above surface | d7 | d6 |
|---|---|---|
| 25 mm | 0.000 mm | 0.001 mm |
| 50 mm | 0.000 mm | 0.000 mm |
| 70 mm | 0.000 mm | 0.001 mm |
| 90 mm | 13.5 mm | 9.8 mm |
| 110 mm | 32.7 mm | 29.3 mm |
| 130 mm | 52.0 mm | 48.8 mm |

Hover is set to 65 mm from this table. The consequence is worth stating plainly: the tallest
piece is 92 mm, so **a transport leg cannot clear an occupied square on these ranks at all**.
The plan's Step 6 safety criterion — every transport waypoint clears the tallest piece by
≥ 1 cm — is unsatisfiable near the arm, and `d7 -> d6` only works because nothing stands
between the two squares. Moves that pass over a piece need a different answer than a higher arc.

## The board's squares had to be given collision

The 64 square overlays were `contype=0, conaffinity=0`, so the surface pieces actually rested on
was the board base, 4.1 mm below `SURFACE_HEIGHT` — the constant every grasp height in this
project is measured from. Pieces started 4.1 mm in the air, dropped on the first steps and
overshot to 5.5 mm mid-bounce, which is what the first run of this episode aborted on.

Making the overlays collide puts the physical surface where the renderer draws it. Measured
after the change: maximum piece drift from placement settles at 0.09 mm, against 4.09 mm before,
and all 30 reference grasps still screen collision-free, so no solved pose moved.

## What the first complete run showed

765 control steps, no aborts, 153 frames, 457 s wall clock. Maximum IK error along the whole
Cartesian path was under 0.01 mm: the fingers-down constraint holds continuously between the
anchors, which `docs/ik.md` could not establish from endpoints alone.

The piece never moved. `final_square` was still `d7`.

## The TCP site sits on the fixed finger, not on the jaw centreline

The first reading of this failure was that `gripperframe` sits ~6 cm past the fingertips, from
the jaw's collision mesh bottoming out at world z 0.069 against a site at 0.045. That reading was
an artifact of measuring with the jaws **wide open**, where the moving jaw is retracted clear of
the workspace by design. Probing the closure directly says something different and narrower.

Sweeping the `gripper` joint with nothing between the jaws, measuring true triangle meshes (not
MuJoCo's convex hulls) in the distal region beyond the pivot, in the `gripper` body frame:

| `gripper` qpos | normalized | fingertip gap |
|---|---|---|
| -0.17453 (lower limit) | `GRIPPER_CLOSED` = 0 | 2.28 mm |
| 0.00000 | 9 | 15.80 mm |
| +0.05000 | 12 | 19.43 mm |
| +0.20000 | 20 | 30.35 mm |
| +1.74533 (upper limit) | `GRIPPER_OPEN` = 100 | jaw retracted out of the band |

**The hand model closes correctly.** The lower limit is the closed end, the jaws converge to a
2.28 mm residual gap at the fingertips, and `denormalize_gripper` maps `GRIPPER_CLOSED` to that
end — the 0/100 convention is not inverted. Plan §0.9 item 2 is not what broke this episode, and
nothing here calls for touching the vendored mechanism.

The site is not 6 cm from anything. In the `gripper` body frame it sits at
`(-0.0079, -0.0002, -0.0981)`, and the closed jaws' contact midpoint is at
`(-0.0068, 0.0024, -0.1006)` — **3.7 mm apart**. `gripperframe` is, near enough, the point where
the two fingers meet when shut.

That is the fault, and it is a lateral one. The fingers close along the body's local x: the fixed
finger's face is at x = -0.0079 and the moving jaw sweeps in from positive x. A frame at the
*closed contact point* is a frame flat against the fixed finger's face, not on the centreline
between the jaws when they are open enough to admit an object. Solving IK to put that point on a
piece's axis therefore parks the fixed finger inside the piece.

Measured at the `d7` reference grasp, jaws open: the site lands exactly on the pawn's stem axis,
and the fixed finger's true mesh passes within **0.03 mm of that same axis** — inside a stem of
9 mm radius. MuJoCo reports the `gripper` body penetrating `piece_d7` by 12.7-17.0 mm. The jaw
that is supposed to do the gripping never has the stem in front of it at all, so closing it
travels to the stop, which is what the run showed.

### Why the screening passed it

`contacts_at_pose` excludes contacts between the hand bodies and the piece named by
`grasped_square`, on the grounds that touching the target is the point of a grasp. That exclusion
covers exactly this penetration: called with `grasped_square='d7'` the pose reports zero world
contacts, and called with `grasped_square=None` it reports all three. The eye check had the same
blind spot from the other side — `back-rank-grasps.png` confirmed TCP over the square centre,
fingers down, no self-collision, and all three hold while a finger is buried in the piece.

## The repair, and what it uncovered

Four changes were made against the diagnosis above. The pick still does not work, but it now
fails on something else, and the something else is a defect in the model rather than in the plan.

1. **Screening.** `contacts_at_pose` still forgives hand-vs-target contacts, because MuJoCo
   collides the hand as a convex hull and the fixed finger is a forked part whose hull fills the
   fork — the space a gripped piece has to occupy. At the d7 grasp the hull claims up to 6.7 mm
   of penetration where the true meshes clear by 4.8 mm, so hull depth carries no information
   there. What changed is that the forgiveness is now paid for: `collision.piece_clearance`
   measures the same pair against the true meshes and the screen requires it positive.
   `executor` additionally requires both jaws in contact before calling anything a hold.
2. **Pinch frame.** A second site, `pinchframe`, offset from `gripperframe` along the closing
   axis; `gripperframe` stays in the model and nothing solves against it.
3. **Grip command.** Sized to the piece's stem rather than driven to the mechanical stop.
4. **Per-piece offset.** Sized once to the widest stem, the offset left a pawn 4.5 mm of run-up
   before the second jaw reached it; the closing jaw shoved the pawn 17 mm and toppled it into
   the king on e8. The offset is now built per piece (`SceneDescription.pinch_offset`).

With all four in, 14 of the 16 black squares solve at grasp height, against the 30-of-64 whole
board the old frame reported. d8 and e8 still do not: the queen and king sit in front of the
shoulder, which is the case `DEFAULT_BOARD_X` was pinned on.

## The blocker now is the base collision box, and it always was

The episode aborts on bystander drift partway down the descent. The commanded path is not at
fault — IK error along it stays under 0.01 mm and the hand's true-mesh clearance to the pawn
never goes negative, bottoming out at 0.63 mm. The arm simply is not where it is told to be:
`shoulder_pan` runs 7.7 degrees behind its command, which at this reach is ~39 mm sideways at
the board, and 39 mm of sideways is a knocked-over pawn.

That error is not tracking lag. Slowing every leg by 2.5x moved it from 7.71 to 7.69 degrees.
Holding the arm perfectly still, commanding exactly the pose it is already in, reproduces it:

| | |
|---|---|
| `base` vs `shoulder` interpenetration | 38.36 mm |
| contact normal force | 262,907 N |
| `shoulder_pan` actuator force | -3.350 N·m, its full limit |
| resulting drift, nothing commanded to move | 0.38 deg (birds-eye), 1.48 deg (resting) |

The base's collision proxy — the box added to the vendored asset, `VENDORED.md` modification 2,
fitted to the AABB of the base's four visual parts — swallows the shoulder that turns inside it.
A quarter of a meganewton of contact force stands on `shoulder_pan` at every pose, and the
actuator is saturated against it permanently.

`baseline_self_pairs` subtracts this exact pair from screening as a known constant overlap, which
is why nothing ever reported it. Subtracting it from the *screen* does not remove it from the
*physics*. Every episode run in this project has been executed by an arm whose base joint is
pinned at its torque limit, including the first one; `docs/trajectory.md`'s original run recorded
the same 0.14 rad and it went unremarked because nothing then depended on the arm being where it
was asked to be. A grasp does.

### What it actually was, and the fix

Re-fitting the box was the wrong repair, because the box shape was not the cause. MuJoCo filters
parent-child body pairs automatically — but not when the parent is welded to the world, and
`base` is bolted down (`dofnum` 0, `weldid` 0). Verified on a two-body toy model: parent welded
to world, child fully inside it, 4 contacts; give the parent its own joint and the identical
geometry reports 0. `base` is therefore the one body in this chain whose collision geometry is
allowed to fight its own child. Every other adjacent pair — `shoulder`/`upper_arm`,
`upper_arm`/`lower_arm` — has been filtered for free all along.

Swapping the box for the true meshes does *not* fix it, measured over 300 sampled arm
configurations:

| base collision geometry | poses with base/shoulder overlap | worst overlap |
|---|---|---|
| box, AABB of the four visual parts | 300/300 | -53.8 mm |
| four mesh geoms (what upstream's visual parts are) | 300/300 | -33.9 mm |
| four mesh geoms + `contact/exclude` | 0/300 | — |
| box + `contact/exclude` | 0/300 | — |

MuJoCo collides meshes as convex hulls, and `base_so101_v2` is a C-shaped part whose hull fills
the cavity the shoulder turns in — the same hull inflation `collision.baseline_self_pairs` already
documents for the shoulder bracket. Mesh geometry makes the overlap smaller, not absent.

The fix is `<contact><exclude body1="base" body2="shoulder"/></contact>`, which restores exactly
the filtering the rest of the chain gets. The base's geometry is now the four visual parts
duplicated as collision geoms rather than a box — an independent change, made because the true
meshes' planar footprint radius is 7.2 cm against the box's 8.6 cm, which is 1.4 cm of board
clearance the AABB was giving away (`docs/reachability.md`, clearance floor 10.9 -> 9.5 cm).

Holding the arm still and commanding the pose it is already in, the table above now reads: zero
penetrating contacts, `shoulder_pan` actuator force 0.0 N·m, drift 0.000 deg. Over the d7->d6
episode, max tracking error falls from 0.134 rad (7.7 deg) to 0.0082 rad (0.47 deg).

The reference grasps do not move: re-solving with and without this fix gives byte-identical
configurations (max joint delta 0.0 deg across all 28 shared squares). IK is kinematic and never
saw the contact force. The `docs/reachability.md` figures do not move either — the FK sampler has
no notion of collision geometry — only the clearance floor does.

### What it leaves

The episode no longer dies on the descent. It now reaches the `grasp` phase and aborts there, on
a bystander disturbed while the jaws close — the failure mode item 4 above was aimed at, not this
one. The base is no longer in the way of diagnosing it.

## The grip was sized against a hand the simulator does not have

With the base out of the way the descent was clean and the jaws closed on the pawn, and the pawn
still came out of them under the lift, sliding 25 mm down through the closed jaws over a 40 mm
rise and dropping. The pinch force at the start of the lift was ~10 N against a piece weighing
0.15 N, with friction 1.0 on every surface involved. Nothing about that adds up, and it did not:
the jaws were not closed on the pawn, they were crushing it.

MuJoCo collides a mesh as its convex hull. The fixed finger is a forked printed part, and its
hull fills the fork — the same inflation `collision.baseline_self_pairs` documents for the
shoulder bracket, on the one part of the arm where it decides what a grip is. Measured, with the
pinch site 20 mm out from the vendored frame:

| | mesh | what collides |
|---|---|---|
| fixed finger's face, from the pinch site | 20.0 mm | 8.4 mm |
| gap at `gripper` qpos 0 | 15.8 mm | 6.7 mm |
| gap per radian | 72.9 mm | 55.7 mm |

So a grip sized off the meshes — the previous `JAW_GAP_AT_ZERO` and `JAW_GAP_PER_RADIAN` — asked
for a 17 mm gap on an 18 mm pawn and actually commanded 11.9 mm: 6 mm of interference, resolved
by the contact solver as a piece squeezed hard enough to squirt out sideways when anything else
touched it. The way the numbers are taken is the only one that cannot disagree with the
simulator: close on a real piece at a real grasp pose, shift the piece along the closing axis and
bisect for the first contact, once per finger.

**The column this section sizes against is superseded.** Sizing the grip against the hull made
the pinch work and the pixels lie — the hull's face is 11.6 mm from the drawn one, so the drawn
fingers held nothing. The hand is now collided as slabs of its own meshes, the two columns agree
to under a millimetre, and the constants are re-measured against that; see *The hand that grips
is not the hand that is drawn* below. What survives from here is the method, not the figures.

## The jaws bear on the head, not on the stem

The fingers reach 8 mm below the TCP and 26 mm above it. A pawn's head — a 10 mm sphere at 42 mm,
1 mm wider than the 9 mm stem the TCP sits beside — stands inside that band, so the faces close
on the head first and the stem never gets touched. A sphere between two flat faces is two point
contacts: correctly sized to the head, with 2 mm of squeeze, the pinch held the pawn through the
lift and rolled it out during the transport.

`scene.grip_radius` is therefore the widest radius the piece presents *where the fingers are*,
not the radius at the TCP's own height, and `GRIP_SQUEEZE` is 4 mm — two of them spent reaching
the stem past the head, the rest the squeeze proper. This is why the pinch offset is per piece
and lives on the scene: it is a piece property, and one site sized for the whole set puts a
pawn 4 mm off the middle of the jaws.

## Only one finger moves

The fixed finger sits a piece-radius from the pinch and cannot get out of the way, which costs
two waypoints:

- **Going in.** A descent straight down the piece's axis drags that finger past the head, which
  is wider than the grip the site is sized to. Measured on the old frame, the clearance at the
  head's equator was 0.8 mm and the finger caught it, tipping the pawn 4 mm before the jaws
  closed on anything. The descent now stops 5 mm short along the closing axis, putting the piece
  on the moving jaw's side of the gap, and the grasp leg slides in as it closes.
- **Coming out.** Opening the jaw does not release the piece: it leaves it resting against the
  fixed finger with a few tenths of a newton of residual contact, and friction against 0.15 N of
  pawn wins. The pawn was carried up with the retreat, still touching, and flung across the board
  during the survey transit — an episode that then reported `placed=True` because the piece
  happened to land back on the target square. The release backs the hand out along the same axis
  before the retreat lifts.

## Fully open is not a posture for the board

`GRIPPER_OPEN` swings the moving jaw 100 mm past the pinch. The neighbouring square's centre is
46 mm away and the queen's stem starts 34 mm away, so closing from fully open sweeps the jaw
straight through whatever stands on the next square: measured at the `d7` grasp pose, the jaw
penetrates `d8`'s queen by up to 9.7 mm on its way in, and that is what the first episode to
reach the `grasp` phase aborted on. Nothing in a plan opens fully any more. `scene.ready_command`
opens only far enough to swallow the piece and its standoff, which reaches 34 mm at the pose
above and clears the queen by 3.8 mm.

The screening followed: `planner.solve_hover_pose` now screens at the opening the plan will be
holding rather than at the fully-open pose, which is not one the hand is ever in near the board.

## Friction has to resolve as stiffly as the normal direction

With the geometry corrected the grip was gentle — 2 mm of interference, both faces on the piece —
and the pawn slid out of it anyway, at every squeeze up to 6 mm. MuJoCo's default contact model
(pyramidal friction cone, `impratio` 1) resolves friction softly enough that a pinch creeps. The
scene now builds with an elliptic cone and `impratio` 10 (`scene.FRICTION_IMPEDANCE_RATIO`), on
which the same plan places the piece. It is a plateau, not a tuned point — 8, 10, 12 and 15 all
place it — bounded above: at 20 the contacts go unstable at this timestep and throw the piece
129 mm into the air.

## The episode that works

805 control steps, 16.1 s, 161 frames, no aborts. Maximum tracking error 0.0082 rad, maximum IK
error along the whole Cartesian path under 0.01 mm. The pawn ends on `d6`, upright, 51.4 mm from
where it started — one square. The largest movement of any other piece on the board is 0.09 mm,
which is the settling floor every piece shows in the first control step.

| phase | piece height above the surface | max tilt |
|---|---|---|
| descend | -0.1 mm (untouched) | 0.0 deg |
| grasp | 0.4 mm | 1.7 deg |
| lift | 38.4 mm | 2.2 deg |
| transport | 36.4 - 38.4 mm | 2.4 deg |
| place | back to the surface | 2.1 deg |
| release | -0.6 mm, hand withdrawing | 1.2 deg |
| retreat, survey_end | untouched, at rest | 0.0 deg |

The same plan places the piece for `b7 -> b6`, `c7 -> c6` and `e7 -> e6`, and for the seeds that
pick different anchor poses out of the solver.

## The hand that grips is not the hand that is drawn

Every number above was measured against the hull, and the hull was accepted as the hand. Looking
at the contact sheet says otherwise: through `grasp`, `lift` and `transport` the pawn rides
against the moving jaw with visible daylight on the other side, and the fixed finger never
touches it. Measured at the `d7` grasp, at the sized grip command:

| | drawn clearance to the pawn | what the engine reports |
|---|---|---|
| moving jaw | -0.03 mm, touching | 0.6 - 1.8 mm penetration |
| fixed finger | **+10.63 mm** | 1.96 mm penetration |

So the piece was carried by a contact force acting across a centimetre of air. That is not a
cosmetic problem in a dataset whose observation is a camera: the wrist view is the only thing the
policy sees, and it shows a hand gripping nothing. The recorded action is wrong in the same
direction and for the same reason — `grip_command` was sized against `GAP_AT_ZERO`, a hull
number, so the jaw angle in the dataset is one that grips in simulation and closes on nothing on
a real SO-101. Pixels and actions agreed with each other and disagreed with the world, which is
the worst version of this: nothing in the data contradicts itself.

The cause is the same convex hull, seen from the other side. The fixed finger is an L — a wide
mount plate, a forked finger reaching 100 mm down from its outboard edge — and one hull over an L
fills the mouth between its arms. The mouth is where the piece goes. The 11.6 mm inset recorded
above was that fill, measured and then built on.

### Collide the part, not its hull

`scene._split_hand_collision` cuts each jaw's collision mesh into slabs across the part's own
long axis and hulls each slab separately (`FINGER_SLAB_CUTS`, `JAW_SLAB_CUTS`). No slab reaches
both arms of the L, so no hull can span the mouth. Only the collision geoms change: the visual
meshes are untouched, and the renderer draws groups 0-2 while these are group 3, so not a pixel
of the observation comes from a proxy. Cutting finer than this buys nothing — the bulge over the
zone a piece can occupy is already under 1.4 mm at 16 mm slabs and 1.2 mm at 6 mm ones.

The moving jaw gets the same treatment for a milder version of the same fault: it is a curved
lever, and one hull across the curve stood 2-4 mm proud of its drawn face at approach openings.

What that is worth, from `pixi run gripper-probe`:

| | before | after |
|---|---|---|
| fixed face, from `gripperframe` | 11.59 mm | -0.96 mm (drawn: -0.96 mm) |
| gap at qpos 0 | 6.67 mm | 17.73 mm (drawn: 17.82 mm) |
| widest opening measured, vs drawn | -3.7 mm | -0.84 mm |
| drawn clearance at the sized grip, fixed finger | +10.63 mm | -2.00 mm |
| drawn clearance at the sized grip, moving jaw | -0.03 mm | -1.49 mm |

Both jaws now bear on the piece in the drawing, each about half the squeeze deep, and the fixed
face lands within a millimetre of `gripperframe` — which is where the vendored asset draws it.
The frame the arm shipped with was the fixed finger's face all along.

The gap is not linear in the joint angle and never was: the moving jaw bears with its pad below
~0.11 rad and with its tip above, at 78 and 56 mm/rad. `GAP_AT_ZERO` and `GAP_PER_RADIAN` are
replaced by `JAW_GAP_TABLE`, sampled either side of the knee and interpolated, refusing to
extrapolate past its ends. The measurement is a fixed point, not a reading: the pinch offset is
derived from `FIXED_FACE_INSET`, and moving the pinch site moves the pose the probe measures at.
Two iterations converge it to 0.01 mm.

`collision.piece_clearance` now reads the visual geoms rather than the collision ones. It always
meant "would the real hand hit this", and until the split the two were the same mesh. So does
occlusion in `survey.framing`, which casts against the groups the renderer draws — a collision
proxy blocking a camera ray is not an occlusion.

### What it costs: the reference set, and what it settles

`assets/reference_grasps.json` is re-solved at the corrected frame: 28 of 64 squares against the
committed 30. The two lost are **`b5` and `g5`**, at the far edge of reach — and `d8`/`e8`, the
black queen and king on their home squares, still solve. That is the question this document left
open, answered the other way: the pinch site moved 12 mm *back* toward the wrist rather than 8 mm
further out, because the fixed face it is measured from was never 11.6 mm inboard. `DEFAULT_BOARD_X`
was pinned at 0.300 on d8/e8 being liftable there, and they are. The board has not moved and does
not need to; nothing is waiting on a re-inspection.

## Wrist roll is unconstrained, and the observation shows it

A fingers-down target constrains the approach axis, not rotation about it (`docs/ik.md`), so
`wrist_roll` is free and the solver takes whatever value it converges on. The camera is bolted
to the gripper, so that free roll lands directly in the observation: in
`docs/episode-views/episode-d7-d6.png` the board's horizon swings through roughly 45 degrees
across the manipulation phases. Two runs of the same move can differ in image orientation
without differing in what the arm did, which is noise aimed at exactly the phases where
precision matters. Pinning roll — as a preference in the grasp solver, or as an extra residual —
is a decision this step surfaces but does not make.
