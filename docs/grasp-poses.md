# Vertical grasp poses

> **SUPERSEDED as the grasp strategy (2026-09-05).** Strictly vertical (fingers-down) grasping is
> too strict — it yields only ~4 usable pick-and-place moves and covers 28 static squares with
> rank 4 entirely absent. It is **no longer the constraint the dataset targets.** The direction is
> a tilted grasp (wrist allowed off vertical, up to ~45°), which raises the kinematic ceiling to
> ~37 fields (ranks 4-centre through 8). See `experiment-2-plan.md` → "Grasp feasibility: vertical
> is too strict". This file still documents the `reference_grasps.json` asset (which exists and is
> used by some trajectory work), but do **not** cite its fingers-down count as the graspable set
> or treat vertical reach as solved.

`src/vla_chess/assets/reference_grasps.json` holds one solved arm configuration per reachable
square: the TCP on the square centre, 2.5 cm above the playing surface, fingers pointing
straight down, and the arm clear of itself, the board and the other pieces. They exist so
trajectory work has verified grasp endpoints to plan to, instead of re-deriving them per
episode. 28 of the 64 squares qualify; the rest are absent rather than present-and-unusable.

Regenerate with:

```
pixi run solve-reference-grasps
```

Verify (slow suite): `pytest -m slow tests/integration/test_grasp_integration.py`. Inspect by
eye: `docs/reference-views.md`.

## Board placement

Solved at the scene's own board pose, `DEFAULT_BOARD_POSE` (x = 0.300) — there is no separate
grasp board distance, and there should not be one again. `--board-x` on the CLI exists for
exploring alternatives; what it must not do is end up committed as a second source of truth,
because reference poses solved at one distance land ~20 mm off every square centre at another,
which is under a piece's foot radius of right and so fails by looking almost correct.

### Why 0.300 and not 0.28 or 0.295

The board sat at 0.28 while d8 and e8 — the black queen and king on their home squares — had no
clean fingers-down pose. Rank 8 is the rank *nearest* the arm, so those two sit right in front
of the shoulder and the arm has to double back over itself to reach them; every pose that
reaches them folds the hand back into the shoulder or the upper arm. Moving the board out is
what resolves it, and the distance is squeezed from both sides:

| board centre x | d8/e8 | birds-eye framing | ranks 6-8 | rank 5 | rank 4 |
| --- | --- | --- | --- | --- | --- |
| 0.285 | both blocked | — | 22/24 | b5-g5 | none |
| 0.290 | d8 blocked, e8 out of reach | — | 22/24 | b5-g5 | none |
| 0.295 | both blocked | 64/64, 1.0% margin | 22/24 | b5-g5 | none |
| **0.300** | **clean** | **64/64, 6.2% margin** | **24/24** | **c5-f5** | none |
| 0.305 | clean | — | 24/24 | c5-f5 | none |

Only the 0.300 row is current. The rest were solved against the grasp frame of the time, and it
has moved twice since — out to `pinchframe`, then back in when the hand stopped being collided as
one hull per part (`docs/trajectory.md`). Re-solved at the frame in the tree, 0.300 gives
**28/64**: rank 5 keeps c5-f5 rather than b5-g5, which is what the row above records. What the
distance was chosen on is unchanged — d8/e8 still solve cleanly there — so the argument below
stands on its own evidence rather than on the count.

The floor is ~0.298: below it d8/e8 come back blocked. It moved out from ~0.293 when the grasp
frame moved from `gripperframe` to `pinchframe`, ~9 mm along the closing axis, which is what
closed the old 0.293-0.297 window and forced the survey pose to be re-searched rather than the
board to be nudged. Correcting the hand's collision geometry brought the frame ~12 mm back the
other way, which is why d8/e8 survive at this distance with room rather than by a hair.

The ceiling is set by the survey pose rather than by the arm, and it moves with that pose: the
one committed at 0.295 lost a corner at 0.297, and the pose searched at 0.300 has 6.2% of the
half-frame to spare — more than the 1.0% it replaced — while hiding fewer squares behind the
arm's own body (60/64 visible against 58/64). See `pixi run search-survey-pose`.

Both bounds were checked across solver seeds, since the near-shoulder squares are exactly where
a single-shot solve reports a search failure as a reach failure.

The cost is **d4 and e4**, which were reachable at 0.28 and now miss by 13 mm. That is a real
loss but a cheap one: ranks 1-4 were already almost entirely out of reach, so d4/e4 were two
isolated squares in a half of the board the arm cannot play from anyway, traded for the two
squares its own queen and king stand on. Ranks 6-8 — everything playing black's opening needs —
go from 22/24 to complete.

Both directions are pinned in `tests/unit/test_grasp.py::TestReferenceAsset`, the gain by
`test_the_black_queen_and_king_can_be_lifted_off_their_home_squares` and the cost by
`test_the_out_of_reach_frontier_is_absent`, so neither can move as a side effect.

The birds-eye ceiling is soft, and this move is what demonstrated it: `BIRDS_EYE_ARM_QPOS` had
been searched at 0.28 and left alone when the board moved, so its 1.0% was what survived that
move rather than a chosen figure. Re-searching it at 0.300 bought the margin back — see the note
on that constant in `scene.py`.

### What is not covered

**Ranks 1-4** have no fingers-down solution at any distance tried (at 0.300, rank 4 misses by
19-62 mm and rank 1 by 157-189 mm), and **a5/h5** miss by 20 mm. A real limit rather than a
tuning gap: the arm plays black, and capturing on white's half of the board needs a different
approach angle, not a different board placement.

**Before trusting any future self-collision rejection, run the exact-mesh check.** MuJoCo
collides meshes as their convex hulls, and these are C-shaped printed parts whose hulls fill the
cavity — measured, the hull inflates the shoulder bracket 3.9x and the gripper 2.6x by volume.
On the old d8 pose it reported 12.7 mm of penetration where the true triangle meshes were within
~0.1 mm of each other. The hull is conservative, so it never misses a real collision, but it
rejects freely: it is what made c8 look blocked for a whole revision of this document, and it is
why "d8 and e8 are impossible" survived here longer than it should have. A hull rejection means
*look closer*, not *impossible*.

One thing that check settled and that still holds: for a square in front of the shoulder there
is exactly **one** fingers-down solution, not a family. 150 random restarts returned 77
in-tolerance poses collapsing to one distinct configuration — three position constraints plus
two approach-axis constraints against five joints is a square system, so solutions are isolated.
Any claim of the form "N solutions were tried and all failed" is one solution tried N times.

## Why the earlier 0.42 recommendation was wrong

The sweep that produced it constrained the **TCP site's z axis** to vertical. The fingers point
along the site's local **x** (measured: `[0.997, -0.002, -0.080]`), so that constraint fixed the
gripper's roll and left the approach direction free — it selected *side* approaches and labelled
them top-down. A configuration it scored as 2.9 deg from vertical actually had its fingers 84.4
deg off vertical, i.e. horizontal. A horizontal gripper 2.5 cm above the board has to pass
through whatever stands between it and the target, which is why ranks 7 and 8 came back almost
entirely blocked, and why the board kept getting pushed further away to make room.

Three other faults in the same sweep, all of which independently invalidate its verdicts:

- **Squares were matched by radius, not position.** Candidates were solved with
  `shoulder_pan = 0` and matched on `hypot(x, y)`, then re-used with `shoulder_pan` set to the
  square's azimuth. That only aims at the square if the TCP lies in the arm's own vertical
  plane at pan 0, and it does not (lateral offset about -60 mm). The pose actually checked
  missed its labelled square by 47 mm on the centre files and 331 mm on the outer ones.
- **`wrist_roll` was swept after solving and never re-validated.** It moves the TCP by up to
  10.8 mm, against the 6 mm tolerance the square match used.
- **The grasped piece was never exempted.** The exemption keyed off geom names, and every geom
  in the vendored SO-101 asset is unnamed, so the set was empty and a successful grasp scored
  as a collision.

`tests/integration/test_grasp_integration.py::TestFingerAxis` guards the first of these.

## Self-collision caveat

The vendored asset declares no exclude pairs and its collision proxies overlap where the real
links do not — `base` and `shoulder` interpenetrate by 38 mm in *every* configuration,
including the resting pose. `collision.baseline_self_pairs` filters pairs that overlap in every
sampled configuration, but pairs that overlap only in some folded configurations still get
reported and may be false. Treat self-collision verdicts near the arm's own body as suspect
until the collision primitives are checked against the meshes.

At 0.300 nothing in the reference set depends on that: no square is reachable-but-blocked any
more, so every pose in the asset is one both the hull model and the true meshes accept, and the
whole class of false rejection has been designed out rather than argued around. It matters
again the moment the board moves back in, or a new approach angle puts the gripper near the
shoulder.

## Scope

These are static endpoint poses. Nothing here checks the approach, lift, transfer or place
*paths* between them; a pair of clean endpoints does not imply a clean trajectory.
