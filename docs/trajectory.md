# Single-episode trajectory

`pixi run generate-episode` plans, executes and records one pick-and-place episode and writes
its wrist-camera view to `docs/episode-views/`. Every number below comes from that task on this
host, or from the probes described alongside it; none are estimates.

The episode is `d7 -> d6`, a black centre pawn one square forward. Black, not white: only 30 of
64 squares have a collision-free fingers-down pose at all and every one of them is on ranks 5-8
(`assets/reference_grasps.json`), so a white pawn's `d2 -> d3` has no grasp pose at either end.
The plan's Step 7 acceptance criterion still names `d2 -> d4`; that test cannot pass as written.

## Shape of the plan

Ten legs: `survey_start approach descend grasp lift transport place release retreat survey_end`,
15.3 s at 50 Hz control, captured at 10 fps.

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

### What this leaves

A pinch frame of our own, and only that: a second site in the `gripper` body offset along the
closing axis to the midline between the finger faces, with the reference grasps re-solved against
it. The offset is roughly half an object width, not 6 cm — for the pawn's 18 mm stem the target
is near x = +0.001 against the site's -0.0079, so about 9 mm. It is not one constant for the set:
stem radii run 9 mm (pawn) to 12.5 mm (rook), so what the site should be pinned to — a nominal
opening, the fixed-finger face plus half a nominal width, or a per-piece-type offset — is a
decision this probe surfaces but does not make. Re-solving moves the frame every reference grasp
and every reachability figure was derived against, so `assets/reference_grasps.json` and the
30-of-64 count in `docs/reachability.md` do not survive the change unexamined.

## Wrist roll is unconstrained, and the observation shows it

A fingers-down target constrains the approach axis, not rotation about it (`docs/ik.md`), so
`wrist_roll` is free and the solver takes whatever value it converges on. The camera is bolted
to the gripper, so that free roll lands directly in the observation: in
`docs/episode-views/episode-d7-d6.png` the board's horizon swings through roughly 45 degrees
across the manipulation phases. Two runs of the same move can differ in image orientation
without differing in what the arm did, which is noise aimed at exactly the phases where
precision matters. Pinning roll — as a preference in the grasp solver, or as an extra residual —
is a decision this step surfaces but does not make.
