# PICO optical hand tracking implementation

Continuation of session `01a069cc-56a8-7133-88a0-24f8aa3310c9` from
`77cbdbe`. The user authorized implementation on 2026-09-07. The design is
`docs/superpowers/specs/2026-09-04-pico-optical-hand-tracking-fullbody-design.md`.

## Work and verification

- [x] Review the design against the real manager, exporter, and backend paths.
- [x] Atomic per-side XR snapshots, strict parsing, and hardware-free parser tests.
- [x] Verified vendored assets and official 0.4.6 retargeting; four real smoke calls.
- [x] Independent source validation, hold/recovery, and one filter/limiter.
- [x] Manager integration in POSE and PLANNER_VR_3PT with explicit profiles.
- [x] Fresh measured Dex3 feedback and manager-owned 50 Hz pacing.
- [x] Debounced controller chords and terminal T/C/S controls.
- [x] Strict packed-message u8 support and restart-safe recorder command/ACK state.
- [x] Exporter integration, nonblocking saves, and exact profile-specific frames.
- [x] Inspire q6/status integration using the existing PC2 safety bridge.
- [x] Behavioral regression tests, replay tooling, launch instructions, diagnostics.
- [ ] Recorded PICO replay qualification (D0), simulation (D1), then hardware gates.

## Integration decisions identified during review

The streamers run synchronously and currently own their sleep calls. In managed
mode the manager must own pacing in every state, including OFF, POSE_PAUSE,
duplicate body frames, and missing body data. Hand freshness and recorder
heartbeats must not depend on a body frame arriving.

An exporter restart invalidates queued recording intentions. The manager must
send NONE/sequence 0 until the new exporter adopts it, and then restart its
command sequence from the acknowledged value. It must never replay an old
START into a fresh exporter automatically. Tracking-exit uncertainty clears
only on fresh status proving capture inactive.

Fresh g1_debug transport alone does not prove Dex3 feedback freshness: existing
output substitutes zeros or republishes cached states. Expose per-side receive
age and validity from getStateWithTime(). Require physical and local transport
age below 100 ms, finite in-limit q7, and initialized measured commands before
entering optical tracking. A side with no measured position emits no command.

Use elapsed monotonic time for the 40 ms debounce and 200 ms chord deadline;
counting two samples alone does not establish 40 ms. Controller loss cancels a
candidate and requires neutral input before rearming.

## Qualification evidence

Software checks and hardware qualification are separate. Record exact commands
and results here as completed. Do not mark real PICO replay, simulation, or
attached-hand stages complete without their specified artifacts.


### Verified software

- Final combined manager, control, hand runtime/tracking, wire, recorder,
  exporter, replay, PC2 and original staleness/entry/ramp tests: **205 passed**.
  Run the `test_pico_*.py` files except the three separate-environment files
  listed below, plus `test_xr_staleness_watchdog.py`, `test_vr3pt_entry_gate.py`
  and `test_initial_pose_ramp.py`. The environment was
  `/home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python` with
  `PYTHONPATH=/home/jihun/work/GR00T-WholeBodyControl/.venv_data_collection/lib/python3.10/site-packages:.`,
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, and `-p no:cacheprovider`.
- Original dashboard, hand-safety bridge/process pipeline, upper-body bridge
  and XR protocol regressions: **108 distinct tests passed, 1 skipped**.
  Process tests needed permission to open localhost sockets outside the sandbox
  and passed on rerun. The skip requires `XR_TELEOPERATE_WORKTREE`, an optional
  external source checkout. No robot peer was used.
- All **30 changed Python files** pass Ruff lint and formatting checks;
  `git diff --check` is clean. One existing Python escape-sequence deprecation
  warning remains in the combined test run.

- Atomic XR parser/getter and C++ physical-feedback serializer: **34 tests
  passed**, including concurrent snapshots and production `-ffast-math` builds.
  Command: `PYTHONPATH=. python3 -m pytest -q
  gear_sonic/tests/test_pico_hand_snapshot.py
  gear_sonic/tests/test_pico_dex3_feedback.py`.
- Pinned real retargeter: **27 tests passed**, no skips, including all four
  profile/side combinations. Environment:
  `/home/jihun/work/IsaacLab/env_isaaclab/bin/python`, with
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `-p no:cacheprovider`, and
  `gear_sonic/tests/test_pico_hand_retargeting.py`.
- Full deployment C++ sources compiled and linked into
  `/tmp/pico-dex3-feedback-build/g1_deploy_onnx_ref`. The worktree contains
  Git LFS placeholder SDK libraries, so the offline link check used existing
  real libraries from the original worktree through temporary CMake paths.
  The executable was not run; install/fetch real SDK libraries for a local build.
- Headset source patch: `git apply --check` passed against the pinned upstream
  `TrackingData.cs`. Unity compilation/APK installation were not performed.
- Both replay CLIs ran with real retargeters and explicitly synthetic 40-tick
  inputs. Reports and arrays are under
  `/tmp/pico-hand-replay-synthetic-k_95ujyd/`. Both reports have zero emitted
  bound, rate, hold and recovery violations, and a unilateral freeze held at
  100 ms. They correctly report `synthetic=true`, `d0_qualified=false`.

### Remaining qualification work

1. Test the rebuilt host binding with the original APK using the authorized
   per-side content-change clock. A patched headset APK is optional for stronger
   device-timestamp evidence. Confirm native hand/body wrist alignment on the
   actual headset stream; host-timed captures do not qualify device freshness.
2. Record real labelled captures for D0, including source loss/restarts and
   body/opposite-hand cadence. Repeat on the target host without concurrent
   test load. The short synthetic smoke had processing p99 **22.47 ms Dex3**
   and **18.50 ms Inspire**, exceeding the 5 ms target; this timing gate remains
   open and may require implementation changes before live qualification.
3. Run D1 simulation and inspect individual finger identity, wrist invariance,
   recovery, and body mode transitions. No simulator or robot was launched during
   the initial software-only verification recorded here. Subsequent user-operated
   MuJoCo Dex3/Inspire and real G1 + Dex3 checks confirmed motion; formal D0/D1
   qualification remains incomplete. See the optical guide’s remaining-work
   section for unresolved thumb mapping and intermittent source articulation.
4. Before attached Dex3 Stage A, enforce the specified 20% travel envelope;
   `--hand-max-rate 0.5` limits speed only. Complete the staged crane tests.
   Inspire remains at its fixed Stage A 800–1000 count envelope and
   5 counts/100 ms limit; expanding it requires later qualification.
5. Complete sustained live recording cycles, durable save/error recovery and
   PC2 ownership/restart tests on the real deployment topology.

The software implementation is ready for these qualification steps, not
approved for unrestricted hardware operation. Changes remain uncommitted.
