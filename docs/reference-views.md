# Reference views

The images in `docs/reference-views/` are the visual sign-off on the scene's board pose and on
the reference grasps solved against it. They exist because the board placement is not settled by
any number in this repo — coverage counts have twice argued for a placement that turned out to
be wrong, and what settled it both times was looking at it. These are what was looked at.

Regenerate with:

```
pixi run render-reference-views
```

Deterministic: both images come from committed poses (`RESTING_ARM_QPOS`, `BIRDS_EYE_ARM_QPOS`,
`assets/reference_grasps.json`) rendered through fixed cameras, with no solving and no sampling.
A regenerated image that differs from the committed one means something upstream moved, which is
the point of committing them.

## `observation-views.png`

The wrist camera — the policy's whole observation — at the resting pose (left) and the birds-eye
survey pose (right), one row per board distance. The committed image compares the retired 0.280
placement against the current 0.300 one.

What it is for: judging framing by eye. The birds-eye row is where the placement and the survey
pose are judged together: all 64 squares are in frame with 6.2% of the half-frame to spare at
the worst corner, against 6.0% at 0.280 and 1.0% at the retired 0.295 (see `grasp-poses.md`).

## `back-rank-grasps.png`

The reference grasps for c8, d8 and e8 at the scene's board pose, from the side, the front and
above, jaws wide open.

Those three because they are the ones the board distance was chosen for. d8 and e8 are the black
queen and king on their home squares, directly in front of the shoulder, and the arm has to
double back over itself to point the fingers down at them; c8 is the neighbouring square that
was already accepted, so it is the control — d8 and e8 should read as the same kind of
configuration, not as something more exotic.

Signed off at x = 0.295 on 2026-08-15, together with the placement itself. **Re-rendered at
x = 0.300 on 2026-08-15 with a re-searched survey pose and awaiting a fresh sign-off** — the
placement changed because the grasp frame moved, so the previous sign-off does not carry over.

## Scope

Static poses only. These say nothing about the paths between them; a clean endpoint is not a
clean trajectory.
