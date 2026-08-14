# Vertical grasp poses

`src/vla_chess/assets/reference_grasps.json` holds one solved arm configuration per reachable
square: the TCP on the square centre, 2.5 cm above the playing surface, fingers pointing
straight down, and the arm clear of itself, the board and the other pieces. They exist so
trajectory work has verified grasp endpoints to plan to, instead of re-deriving them per
episode. 30 of the 64 squares qualify; the rest are absent rather than present-and-unusable.

Regenerate with:

```
pixi run solve-reference-grasps
```

Verify (slow suite): `pytest -m slow tests/integration/test_grasp_integration.py`.

## Board placement

Solved at the scene's own board pose, `DEFAULT_BOARD_POSE` (x = 0.28) — there is no separate
grasp board distance, and there should not be one again. An earlier revision of this file
argued for 0.30 on the strength of the sweep below:

| board centre x | clean squares (ranks 5-8) | what runs out |
| --- | --- | --- |
| 0.28 | 27/32 | c8/d8/e8 self-collide — the arm folds too tightly |
| 0.30 | 30/32 | only a5/h5 |
| 0.32 | 28/32 | a5/b5/g5/h5 |
| 0.34 | 24/32 | most of rank 5 |
| 0.42 | 10/32 | ranks 5 and 6 both out of reach |

That argument was retired for two reasons. The decisive one is that 0.28 is the placement the
scene was **inspected at** — the resting and birds-eye framings, and every coverage figure
recorded with them, reproduce at 0.28 and nowhere else — and a square-count delta is not
grounds to move a pose that was signed off by eye. The second is that the two-squares-better
claim did not survive contact: solving at 0.28 with the collision screen now in the solver
yields **30 clean squares**, the same as 0.30, because c8's failure was a search failure rather
than a reach limit (the first random restart finds a clean pose; the old sweep only ever tried
one). Keeping a second board distance bought nothing and cost correctness — reference poses
solved at 0.30 land 20 mm off every square centre on a default scene, which is under a piece's
foot radius of right and so fails by looking almost correct.

`--board-x` on the CLI still exists for exploring alternatives; what it must not do is end up
committed as a second source of truth.

### What is not covered

**d8 and e8** are reachable fingers-down but have no collision-free solution at all: they sit
directly in front of the shoulder, and a 60-restart search finds ~186 in-tolerance poses for
each, every one of which folds the arm through its own body. This is load-bearing — the black
queen and king cannot be lifted off their home squares by this approach — and is pinned by
`tests/unit/test_grasp.py::TestReferenceAsset::test_the_two_blocked_home_squares_are_still_blocked`.
It wants either a non-vertical approach for those two squares or a different board placement,
and the latter is not free (see above).

**Ranks 1-4** have no fingers-down solution at any distance in the sweep (rank 4 misses by
17-61 mm, rank 1 by 157-184 mm). Also a real limit rather than a tuning gap: the arm plays
black, and capturing on white's half of the board needs a different approach angle or a
different board placement.

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
reported and may be false. Treat self-collision verdicts near the arm's own body — the d8/e8
exclusion above, whose contacts are `shoulder`/`upper_arm` against the gripper at depths of
5-17 mm — as suspect until the collision primitives are checked against the meshes. The
exclusion is the conservative direction (two squares withheld that might be usable), so it is
safe to ship, but it should not be treated as settled: if the proxies turn out to be too fat,
d8 and e8 may come back.

## Scope

These are static endpoint poses. Nothing here checks the approach, lift, transfer or place
*paths* between them; a pair of clean endpoints does not imply a clean trajectory.
