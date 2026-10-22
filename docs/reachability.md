# Reachability probe

`pixi run reachability-probe` regenerates this file from a fresh sample; every number below
comes from that sweep, none are estimates.

## Method

This is a coarse, forward-kinematics-only probe, **not** an IK-based one: `mink` integration
is Step 5. 60000 uniformly random joint configurations (within each arm joint's
own range, independent per joint — no self-collision or orientation feasibility check) are
forward-kinematically evaluated, and the gripper site (`gripperframe`, the TCP) world position
is recorded for each. A square is marked reachable if some recorded TCP position, restricted
to a height band of -0.01 to 0.12 m above the table
surface, falls within 2.0 cm of that square's centre.

This is a real bound on board *placement* — it does not mean every reachable square is
graspable at every orientation a pick/place motion needs. Step 5's actual IK convergence rate
near the board surface is the authority on that; per the plan's risk register, an IK
convergence problem there is treated as a reason to re-tune Step 3/4 reachability bounds, not
to swap solvers.

**Do not read the verdict below as a grasp count.** It is the loosest possible reading of
"reachable": any orientation, no collision check, no piece on the square. The graspable set is
much smaller and is **not** settled by this probe.

> **Grasp strategy updated (2026-09-05):** the old fingers-down count (~28–30 squares) is
> superseded. Strictly vertical grasping is too strict (~4 usable moves). The direction is a
> tilted grasp (wrist ≤ ~45° off vertical), which lifts the kinematic ceiling to ~37 fields
> (ranks 4-centre through 8). See `experiment-2-plan.md` → "Grasp feasibility: vertical is too
> strict". Do not treat vertical fingers-down as the grasp requirement or as solved.

**Both of those are still sim-only confirmations.** Every number this probe reports —
table height, board offset from the arm base, square pitch — is a value chosen in
simulation with no real SO-101 unit or physical chessboard behind it. This probe (and
Step 5's IK convergence check, when it exists) can only confirm internal consistency of
that guessed geometry, never that it matches real hardware. Treating "64/64 reachable in
sim" as the reachability question being closed would be wrong; it stays open — for Step 5
as much as for Step 3 — until someone measures an actual SO-101 mount height and board
placement and this probe is re-run against those numbers.

## Board pose probed

| | |
|---|---|
| Centre (x, y) | (0.300, 0.000) m |
| Yaw | 1.571 rad |
| Square pitch | 0.050 m |

## Per-square verdict

Ranks top-to-bottom 8→1, files left-to-right a→h (a white-side view of the board).

|  | a | b | c | d | e | f | g | h |
|---|---|---|---|---|---|---|---|---|
| 8 | reach | reach | reach | reach | reach | reach | reach | reach |
| 7 | reach | reach | reach | reach | reach | reach | reach | reach |
| 6 | reach | reach | reach | reach | reach | reach | reach | reach |
| 5 | reach | reach | reach | reach | reach | reach | reach | reach |
| 4 | reach | reach | reach | reach | reach | reach | reach | reach |
| 3 | reach | reach | reach | reach | reach | reach | reach | reach |
| 2 | reach | reach | reach | reach | reach | reach | reach | reach |
| 1 | **miss** | **miss** | reach | reach | reach | reach | reach | **miss** |

## Verdict

**61/64 squares reachable** — meets the 60/64 minimum (implementation-plan-step-1.md §3 acceptance criteria).

Unreachable squares: a1, b1, h1

## Randomization envelope (Step 4)

Step 4 plans to sample board pose per episode: pitch uniform in
[0.03, 0.05] m, xy offset up to
±3 cm, yaw offset up to ±15°, all
around the nominal centre pose above. Rather than trust the nominal pose alone, every grid
point of that envelope (135 combinations) is checked against the *same*
FK sample set — reachability sampling doesn't depend on the board pose being tested, only
the arm's own joint-space, so this reuses the nominal probe's samples at no extra simulation
cost.

Two independent things can fail here, and neither implies the other: a pose can have plenty
of FK samples landing near every square (reachable) while still placing squares inside the
arm base's own physical footprint (no clearance) — the FK sampler has no notion of the
board's geometry at all, so it cannot catch that case on its own. `min_square_clearance`
closes that gap: the planar distance from the arm's mount origin to the nearest square
centre, checked against the base collision meshes' own farthest-vertex radius
(7.2 cm) plus a standing piece's foot radius
(1.3 cm) plus a 1.0 cm margin.

| | Reachable squares | Clearance |
|---|---|---|
| Worst-case pose | 55/64 (pitch 0.050, centre (0.330, -0.030), yaw 1.309 rad) | — |
| Tightest-clearance | — | 8.2 cm (pitch 0.050, centre (0.270, -0.030), yaw 1.833 rad) |

**Worst case across the envelope: 55/64 reachable, 8.2 cm minimum square-to-base clearance**, against a 60/64 reachability floor and a 9.5 cm clearance floor (7.2 cm base footprint + 1.3 cm piece foot + 1.0 cm margin). This corner violates one or both floors — expected: see the resolution note below, this is not a Step 3 blocker.

### Resolution

No single nominal board offset clears both floors at every corner of this envelope — pushing
the board farther out trades a clearance failure for a worse reachability failure, and vice
versa (see `implementation-plan-step-1.md` §4, "Board pose feasibility"). This is not solved
by picking a different offset: Step 4's `SpecSampler` checks `min_square_clearance` (and,
once Step 5 exists, real IK feasibility) against each individual drawn `BoardPose` and
resamples on failure, the same reject-and-resample pattern already used for distractor-square
collisions and IK targets elsewhere in the plan. The nominal offset is set by the grasp
constraints in `docs/grasp-poses.md` rather than by this sweep, and the full pitch/xy/yaw
ranges are kept; only draws landing in this corner are rejected.
