# One-second balance disturbance at SONIC mode switches

The live PICO test exposed a second defect after the reference-handoff repair:
changing VR/upper-body data availability started a one-second ramp on all 29
motor position targets. During that interval, deployment blended current policy
targets with a fixed snapshot of measured joint positions. The first 20-ms tick
retained only 0.1184% of the policy's target offset from that snapshot. This
suppressed leg balance corrections while `last_action` still reported the raw,
unattenuated policy action.

SMPL packets include auxiliary `vr_position` data, even though encoder mode 2
does not consume VR targets. Consequently, both SMPL↔PLANNER transitions triggered
the ramp. Disabling optical finger input did not avoid it.

The repair removes this post-policy ramp, its state, and availability-edge
triggers. The dedicated `InitControl()` standing ramp is byte-for-byte unchanged.
Manager VR entry checks, recalibration and reference-target smoothing remain.

## Evidence

The [live analysis](switch_metrics.md) contains nine actual A+X switches. Playback
was continuous. All four POSE→PLANNER switches had a first-second tilt increase
above both adjacent windows. One representative event was **4.00° → 10.48° →
3.33°**, with maximum measured joint speed **0.932 → 7.360 → 2.246 rad/s**.
Eight switches had their largest first-second raw-action change around 0.5–0.6 s,
mainly at ankle-pitch joints. No switch-local control gap exceeded 25.8 ms.

The earlier automated handoff test omitted VR fields, so it never activated this
motor ramp. The new isolated replay retains those fields from the first live
pose packet and repeats its last body frame with fresh frame indices. It uses
SONIC v1.1, the same models/GPU/scene and `--live-pose-playback` for both binaries.
Only the generic motor-ramp removal differs between the compared source versions.

| Run | Output ramps | Min root height (m) | Max root tilt (°) | Physics/wall ratio |
| --- | ---: | ---: | ---: | ---: |
| Original comparison, ramp present | 6 | 0.768751 | 5.920664 | 0.81549 |
| Repeated baseline, ramp present | 6 | 0.764425 | 6.463138 | 0.96752 |
| Corrected deployment | 0 | 0.783309 | 4.526309 | 0.96160 |

Each run completed six switches over 38 seconds without a fall. Initial support
was removed after three seconds; whole-run extrema exclude the first five
seconds. The first baseline ran substantially slower, so the repeated baseline
is the primary comparison. In that repeated baseline, POSE→PLANNER first-second
peak tilts were **6.40°, 5.88°, 5.49°**, versus **4.13°, 4.29°, 4.26°** after
repair. Corrected return-to-planner peaks were below their pre-switch maxima.

![Matched-speed comparison](comparison.png)

This is a controlled standing-pose replay, not a repeat of arbitrary live body
motion. It supports the identified mechanism and repair, and the subsequent live operator retest confirmed that the one-second balance
loss was gone. Physical G1 confirmation remains pending. The live capture also has later
fall/reset episodes beginning 5.4 seconds after its final switch; their separate
cause has not been established. They must not be dismissed as post-stop data.

## Checks

- The regression compiles the real `CreatePolicyCommand()` body with production
  joint ordering and scales. It failed before removal (`ramped q_target at 0`)
  and passes after removal, checking all 29 motor targets and action-history
  consistency across 51 changing-action ticks.
- 72 focused Python tests passed; both existing CTest tests passed.
- The release executable rebuilt successfully. Independent scoped review found
  no blocking issue. No PC2 installation or real-robot command was performed.
- The small compiled harness stubs inference/gain scaling and primes the old
  ramp state. It does not test live input edges or dynamics; replay covers the
  actual packet-triggered ramp path.

## Artifacts and reproduction

`comparison.json` has individual switch windows. Compressed CSVs and logs retain
all three replay runs; `live_before/` retains the live diagnostic measurements.
`manifest.json` identifies both binaries, modified source and source/packaged
fixture hashes. Model hashes match the preceding
[handoff evaluation](../sonic_mode_handoff_20260914/manifest.json).

`pose_fixture.npz` contains only the body/VR fields needed from the live packet;
controller buttons, hands and recording toggles are excluded. The runner repeats
the last body frame and preserves VR fields, exactly as in the recorded test.
`run_config.json` records actual local paths. For another checkout, copy and edit
it to use available absolute paths, the packaged fixture, disposable model
copies, and a new output directory. No weights are included.

Run the same script for `baseline` and `patched` in a private loopback-only
namespace, with an existing simulation Python environment:

```bash
SIM_PYTHON=/absolute/path/to/existing/venv/bin/python
unshare --user --map-root-user --net sh -c \
  'ip link set lo up && exec "$@"' sh \
  "$SIM_PYTHON" "$PWD/run.py" /absolute/path/to/config.json \
  --variant baseline --scenario nominal
```

Repeat with `--variant patched`. The runner refuses ordinary host networking and
has bounded initialization/runtime and fall-threshold cleanup. Analyze archived
measurements without starting control:

```bash
"$SIM_PYTHON" compare.py
"$SIM_PYTHON" analyze_switches.py
```

The latter script is specific to the archived live capture and regenerates its
report; it is not a general pass/fail qualification tool. `command_age_ms=-1` in
replay physics CSVs is an unimplemented placeholder, not a latency measurement.

## Live confirmation after repair

The user repeated ten live A+X switches using rebuilt commit `20805c8`, then
explicitly confirmed: **“Balance loss is gone.”** This test used `--hand-input off`
to keep the unrelated right-thumb feedback gate from blocking POSE entry.
Both directions were exercised with support disabled. There were zero paused
reference ticks and zero root-height/tilt fall-threshold crossings during the
logged control interval. The operator used several intervals shorter than the
requested five seconds; these are recorded results, not a timing qualification.

See [live_after/summary.json](live_after/summary.json) and its compressed source
measurements. The three test processes were stopped afterward. Optical-hand
integration subsequently passed repeated A+X switches; see the
[hand ownership record](../sonic_hand_mode_ownership_20260914/README.md). Physical PC2/G1 confirmation remains pending;
no PC2 installation was performed. The earlier sustained-POSE fall/reset episodes
remain a separate investigation and are not claimed resolved by these switches.
