# PICO hand tracking: post-merge status and qualification

PR [#9](https://github.com/caisarl76/GR00T-WholeBodyControl/pull/9) merged as
`d56efdb` on 2026-09-11. The September 11 pre-merge audit describes the old
`fix/xr-hand-tracking-watchdog` checkout, not the current state of `main`.
The original checkout and its ignored captures are retained for investigation.

## Completed integration work

| Previous audit item | Resolution in PR #9 |
| --- | --- |
| Untracked implementation | Modules, pinned assets, viewer, tests and documentation committed and merged. |
| Branch 74 commits behind | Optical changes integrated on `e9dd3af`, preserving main's manager, exporter, simulation and deployment features. |
| Hard-coded external-worktree tests | Older XR safety-v2 commits and their external-worktree tests excluded from the optical PR; those separate tests were not rewritten or qualified. |
| Recording documentation | Managed command/ACK controls and explicit legacy migration documented; guide added to index; historical test claims qualified. |
| Ignored issue handoffs | Known limitations published in the optical guide; detailed remaining evidence is retained below. Raw captures remain local, not packaged. |
| Wheel assets and clock provenance | Viewer HTML and Inspire XML included; 12 assets checked in a built wheel. Episodes preserve per-hand timestamp source: -1 unavailable, 0 device, 1 host content-change. |
| PC2 deployment patch | DDS receipt timestamps and age/validity fields included in C++ source. PC2 uses the previously approved rebuilt SONIC v1.1 executable. Merging does not replace running executables. |

The merged tests and build results are in the
[main integration record](../superpowers/plans/2026-09-11-pico-optical-main-integration.md).
No new PC2 installation or physical control run was performed to complete this
post-merge audit.

## Right index feedback near zero

A deterministic offline reproduction isolates the admission refusal from
source tracking: supply the recorded right-hand positions with fresh synthetic
transport metadata to `PicoHandRuntime.ready` and `HandTracker.step`.
At `capture_000035.npz`, row 42 in the Sep11 capture, right `index_0`
(position **5** in the seven-motor vector) is **-0.0007002827478572726 rad**.
The lower bound is zero; the original 0.0001-rad tolerance rejects it.
Changing only that measurement to zero admits the same sample.

| Capture directory under `outputs/` | Ticks | Finite feedback rows per side | Right rows beyond original tolerance | Enabled numerical rejection ticks | Largest lower excursion |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pico_dex3_real_20260911_141239/hands` | 9,528 | 4,331 | 1,364 | 702 | 0.000917662 rad |
| `pico_dex3_real_20260909_180008/hands` | 2,284 | 2,102 | 109 | 109 | 0.000586656 rad |

Only right `index_0` exceeds the original tolerance. All other thirteen
side/joint combinations stay inside it. Every enabled numerical rejection
matches recorded `FEEDBACK_UNAVAILABLE` in both captures. The longest interval
is 230 ticks (4.5953 seconds) in Sep11 and 109 ticks (2.167 seconds) in Sep9.
The previously working capture therefore also contains this failure.

Both the physical G1 and retargeting URDFs specify a zero lower limit for this
joint. Recorded values round-trip through float32 exactly; log conversion does
not explain the excursion. This is distinct from the earlier thumb-1 physical
versus optimizer range mismatch.

The offline-tested correction admits a **measurement-only** right
`index_0` lower excursion up to **0.001 rad**, projecting its startup
baseline to zero. Target/command limits, upper limits, other joints/sides,
Inspire checks, source admission and DDS freshness checks retain their existing
behavior. Raw measurements remain recorded without this projection.

This allowance covers the observed excursions; it is not a manufacturer
accuracy specification or an encoder calibration. Captures cannot distinguish
encoder offset, compliance or sensor noise. Validate the correction on the
established real setup before marking physical confirmation complete. Values
beyond the allowance must still block admission/hold; do not disable feedback
validation to make POSE entry succeed.


### Offline validation result

Three new runtime admission cases fail on the merged baseline and pass with
the correction. Tracker/runtime/replay regressions pass (103 tests), including
boundary/outlier rejection, raw measurement preservation, stale feedback,
five-frame admission, slew limiting and strict target validation. Real
retargeter tests also verify the pinned right index joint order (36 tests).
Manager/controller/wire compatibility checks pass (63 tests): 202 focused
tests total. An independent review found no blocking findings.

| Replay check | Sep11 baseline → corrected | Sep9 baseline → corrected |
| --- | ---: | ---: |
| Right feedback-unavailable ticks | 702 → 0 | 109 → 0 |
| Command bound/nonfinite violations | 0 → 0 | 0 → 0 |
| Rate violations / held-command changes | 0 → 0 | 0 → 0 |
| Early recovery/tracking violations | 0 → 0 | 0 → 0 |
| Reported late/nonholding freezes | 6 → 2 | 2 → 2 |

Left command arrays are bit-identical. Rates of 0.5 and 2.0 rad/s were inferred
from command saturation; baseline replay reproduces both recorded command
arrays exactly at those rates, but the original launch flags were not logged.
Corrected right commands differ by up to 0.001713 / 0.255247 rad: removing a
rejection allows later tracking/recovery, so the effect is not limited to the
initial measurement projection. These are counterfactual replays against fixed
recorded feedback, not measured physical outcomes.

Both corrected reports still return `replay_safety_checks_pass=false` and
`d0_qualified=false`. The two remaining freeze events in each capture already
occur on the baseline: bilateral startup DISABLED→SOURCE_STALE while still
WAITING (Sep11 row 5412, Sep9 row 288). Review the checker's distinction between
pre-admission waiting and loss after tracking; these events are retained, not
suppressed in this fix.

Captures are 191.425 and 46.565 seconds, have no pose labels, and use host
content-change timing. Body maximum gaps are about 221 / 222 ms. Offline P99
processing was 3.386 / 14.249 ms during concurrent replay, not a live-host
qualification measurement. No overall qualification pass is claimed.
The [machine-readable replay evidence](../artifacts/pico_hand_feedback_20260911/replay_summary.json)
retains both baseline and corrected failures, missing gates and the minimal
reproduction shard hash.

## Intermittent raw articulation: unresolved

The user reported that the headset visualization bent the left fingers during
a failure, while the robot fingers stayed straight. Tracking later recovered;
no permanent root-cause fix was established.

The Sep11 capture above contains nearly straight raw landmarks on both sides,
including active samples rejected for feedback. Maximum intersegment bends:

| Raw finger | Sep11 left | Sep11 right | Sep9 left | Sep9 right |
| --- | ---: | ---: | ---: | ---: |
| Index | 0.01485° | 0.01511° | 71.6° | 55.3° |
| Middle | 0.01794° | 0.01814° | 74.3° | 77.6° |

The Sep11 thumb maximum is below 0.031° on both sides. The input schema is the
same in both recordings. The loss is visible before anatomical wrist-frame
conversion, optimization, command smoothing or DDS output. The right feedback
rejection does not explain the missing raw left articulation.

On recurrence, preserve a capture and note exactly when deliberate bare-hand
open/close was performed. Use the [read-only viewer](pico_optical_hand_tracking.md#inspect-pico-input-and-recorded-dex3-actions)
to compare raw landmarks, flags, target, command and measured positions. Body
FPS, an active hand flag, and changing wrist positions do not establish finger
articulation. If the headset bends correctly while the captured raw skeleton
stays straight, the next distinguishing evidence is the original SDK JSON
versus atomic binding snapshots for the same event. That boundary comparison
has not yet been recorded.

Do not synthesize finger motion from a static sample or label all such failures
as timestamp staleness: wrist movement can keep the sample changing while the
finger geometry remains straight.

## Open-thumb handoff: deferred to another session

User explicitly deferred spread-thumb geometry. The existing direction
correction is implemented, but the robot thumb was still observed pointing
inward; regression tests do not prove the visual mismatch fixed.

The prior model audit found matching thumb axes, joint order and origins
between the default SONIC 43-DOF scene and pinned retargeting URDFs, accounting
for the fused palm. The separate `gear_sonic_deploy/g1` XML has different
lengths and is not the default SONIC simulator model. Verify the actually active
model when resuming; this comparison does not rule out a different launch.
Retain the exact source pose and target/command/measured joints for the failing
spread-thumb frame rather than changing signs based on a screenshot alone.

## Qualification still requiring operator evidence

User-confirmed functional checks: MuJoCo Dex3 and Inspire finger movement;
real G1 + Dex3 on PC2 `.223`, SONIC v1.1, both hands, pinch, wrist rotation and
open/close with increased hand rate. Those checks do not complete the formal
D0/D1, sustained recording, or physical Inspire gates.

Use the established deployment and stop procedure. The operator performs the
following checks; the replay/viewer commands are read-only.

1. Keep body tracking active, hide only the left bare hand for 1–2 seconds,
   then restore it. Verify left command hold, continued right articulation,
   and bounded recovery. Repeat three times per side.
2. Record controller pickup and replacement, source restart and recovery,
   and the established independent policy stop. Manager Ctrl+C does not serve
   as the policy stop.
3. Capture at least ten minutes at the exact intended `--hand-max-rate`, with
   bilateral articulation and independent occlusions. Record the rate, manager
   revision, deployment revision, checkpoint, source clock type, and event times.
4. Exercise managed recording start, stop/save, abort and exporter restart.
   Verify RECORDING acknowledgement, durable IDLE after save, no unintended
   new episode after restart, and expected episode contents. Raw hand NPZ
   capture alone is not evidence of a saved LeRobot episode.
5. Review live manager and deployment timing, not only replay throughput.
   Original-APK host content-change timing is not device acquisition timing.
   Keep any unavailable formal evidence explicitly pending.

For an existing capture, in the teleop environment from the checkout being
verified:

```bash
# Set CAPTURE to the hand shard directory and RATE to the actual capture rate.
python -m gear_sonic.scripts.replay_pico_hands --profile dex3 \
  --max-rate "$RATE" --input "$CAPTURE"/capture_*.npz \
  --output-dir "${CAPTURE}_replay"
cat "${CAPTURE}_replay/report.json"
python gear_sonic/scripts/view_pico_hands.py --capture-dir "$CAPTURE" --port 8766
```

Replay reports bounded-command, hold/recovery and timing evidence along with
missing qualification gates. Do not invent pose labels or mark missing live
measurements complete to make `d0_qualified` true. Physical Inspire validation
is separate from the confirmed MuJoCo Inspire result.

## Full-fist calibration: separate feature

DexPilot matches fingertip geometry within Dex3 limits; it does not implement
a user-calibrated fist-to-full-travel gesture. Increasing `--hand-max-rate`
changes speed, not the target pose. Full-fist calibration needs its own desired
endpoint definition and validation; it remains unimplemented and is not part
of the feedback correction.
