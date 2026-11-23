# When a move is done

A policy flies a move for as long as it is asked to, and nothing in the closed-loop eval says
when it should stop: `eval_metrics` drives every rollout for `--max-frames` (750) frames and
scores where the piece stands after the last one. A deployment has no such number. The arm has to
recognise that the move is over.

## The rule

`control.done.DoneDetector` reads the end of a move off two signals a real arm reports as well as
the simulator does: the five arm joint angles, and whether the hand grips something. A move is
done once the hand has gripped for at least `DONE_GRIP_FRAMES` (5) frames in a row, and afterwards
the arm has stood within `DONE_REST_TOLERANCE` (0.1 rad, every joint) of its rest pose, not
gripping, for `DONE_HOLD_FRAMES` (15) frames: half a second at 30 fps. The grip requirement keeps
the rest the arm starts every move from out of it.

`eval_metrics` keeps driving to `--max-frames` and reports both scores. `placement_rate` is read
after the last frame, as before. `placement_at_done_rate` is read on the frame the move became
done, and a move that never became done is a failure, as it would be on the arm. `done_rate` is
how often the rule fired at all.

## What it showed on the round 1 policy

Replaying the 1000 recorded rollouts of the DAgger v3 round 1 policy over the 20 trained one-step
moves (`dagger_v3/round2/rollouts`), frame by frame:

| | Rollouts |
|---|---|
| Piece placed and released on its target at some frame | 136 |
| ... arm then came home with the piece still placed | 81 (4 of them disturbed it later) |
| ... piece knocked off while the arm was on its way home | 15 |
| ... arm never came home within 750 frames | 40 (16 of them placed at frame 750) |

Placement after 750 frames was 9.8%; placement at done, with the rule above, 7.5%. The rule
fires reliably once the arm is home: the policy settles within about 0.01 rad of the rest pose.
Few placements are lost while the arm stands at rest. They are lost on the way home, or because
the policy never heads home: the expert's episodes end the moment the arm is back, so nothing it
was trained on shows the arm staying there.

## Training for it

`dagger_takeovers --frame-order placed` takes rollouts over only on the frames after the piece was
placed and before the arm was home, so the expert demonstrates backing off and flying home from
wherever the policy left the arm. `--rest-hold` appends a still leg at rest to every recovery
(`control.recovery.hold_at_rest`), so the demonstrations also show the arm staying put once the
move is done. `runpod/dagger_v3_r1_placed.env` runs one such round from the round 1 policy.
