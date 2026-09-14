# Dex3 optical ownership and POSE re-entry — 2026-09-14

## Final implementation

- Optical Dex3 finger commands are sent only in POSE. Planner packets omit
  hand fields, allowing SONIC's normal fist.
- POSE entry retains the fresh measured-feedback gate. The first hand command
  uses that same feedback snapshot, preventing a second socket poll from
  invalidating admission mid-tick.
- Entry clears the previous optical command and seeds from measured positions,
  preserving source clocks and requiring five valid optical frames before
  rate-limited recovery.
- At the user's request, measured Dex3 positions have a uniform **0.02-rad**
  allowance beyond either nominal limit on every joint, for both hands.
  Admitted positions are projected into nominal command bounds. This replaces
  the earlier per-joint exceptions. Raw feedback is retained.
- Optical target tolerance remains 1e-4 rad. Nonfinite/malformed/stale feedback,
  larger excursions, source admission and slew limiting retain their checks.
  Inspire behavior is unchanged. No PC2 installation was performed.

## Diagnosis and offline evidence

A+X was recognized, but the measured-feedback gate rejected positions just
outside the optical model's limits. In `manager-OkXG9Q`, right index_1 exceeded
its nominal upper bound by 0.001347 rad and middle_1 by 0.001275 rad. A bounded
command held at the soft joint stop did not guarantee an in-bounds measurement.
After returning to PLANNER, all 2,077 remaining right-side samples failed
admission. The active simulator uses the local `scene_43dof.xml` and
`g1_29dof_with_hand.xml`; the separate deployment XML is not that active model.

The first attempted correction covered distal joints. With the requested
normal planner fist, `manager-P2Vsb4` also showed right index_0 at
1.571054697 rad versus its 1.57079632-rad nominal limit. A knuckle-specific
allowance corrected the settled refusal, then the user requested looser limits
and the implementation was simplified to the uniform 0.02-rad allowance.

The final check over `manager-P2Vsb4` admits all 1,804 finite measurements per
hand and rejects all 34,839 nonfinite samples per hand. This is a bounds check,
not proof of transport freshness or closed-loop performance.

Historical evidence files preserve earlier diagnostic stages:

- `stats.json`: per-joint statistics for the first capture.
- `selected_feedback.npz`: six representative measured/command rows, not the
  full optical capture.
- `bounded_comparison.json`: the earlier focused allowance removed 697
  enabled and 2,077 post-exit right-side refusals; left was unchanged.
- `knuckle_retest.json`: the intermediate knuckle allowance removed all 1,000
  right-side refusals in the last ~20 seconds of the subsequent capture.

## Verification and live confirmation

**271 focused Python tests and both C++ deployment tests pass.** Tests cover
uniform feedback-bound acceptance and outlier rejection, unchanged optical
bounds, raw feedback, stale data, source-clock preservation, recovery/slew,
manager POSE -> PLANNER -> POSE, absent planner hand fields, the entry
feedback-poll race, and exact runtime/replay command and state equality across
re-entry. Final review caught that replay initially missed the ownership reset;
the reset now lives in the shared HandTracker so both paths use it.

```sh
python -m pytest -q gear_sonic/tests/test_pico_hand_runtime.py gear_sonic/tests/test_pico_manager_optical.py
```

The user then confirmed: **“confirmed. a+x working on switching modes.”**
`live_confirmed_manager.log.gz` records four PLANNER -> POSE and four
POSE -> PLANNER transitions. `live_confirmation.json` captures the exact
confirmation and scope. The earlier balance repair has separate logged live
confirmation in [the output-ramp record](../sonic_output_ramp_20260914/README.md).

## Remaining validation

Physical G1/PC2 validation of these latest changes is pending. Existing SONIC
v1.1 deployment needs the C++ handoff/output-ramp changes; the manager changes
run on the workstation. Do not infer hardware qualification from MuJoCo.
Independent tracking loss/recovery, sustained recording, formal timing evidence,
open-thumb mapping and full-fist calibration remain separate follow-ups.
