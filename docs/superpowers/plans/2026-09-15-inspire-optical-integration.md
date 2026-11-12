# Inspire optical integration implementation plan

**Goal:** Integrate the working physical Inspire bridge corrections into native optical q6 tracking and qualify locally before any physical rollout.

**Architecture:** Retain the native six-motor optical protocol and status endpoint, port the corrected DDS actuation behavior, and keep the verified PR11 body handoff implementation. Preserve manager-state-before-body/hand provenance and body-before-mode-command commit ordering. Inspire planner modes hold bounded hand targets and remain recordable; independent optical loss holds one side.

**Spec:** `/home/jihun/work/GR00T-WholeBodyControl/docs/inspire-optical-hand-tracking-handover-20260915.md` (user-provided integration instruction).

## Source preservation

- Integration branch: `work/inspire-optical-integration`, base `6501289` (PR11 merged as `818d9d2`).
- Original optical and Inspire worktrees remain untouched. Frozen source and SHA256 manifests: `/tmp/inspire-optical-integration-snapshots/{optical,inspire}`.
- Use the tested optical base rather than resurrecting older uncommitted files already superseded by PR9–11. Port working physical behavior and concurrent dataset corrections from the frozen Inspire source by interface, not whole-manager replacement.
- Physical target is Inspire PC2 `.222`, not Dex3 PC2 `.223`. No remote writes, installs, or control in this task. SONIC v1.1; MuJoCo first.

## Tasks

- [x] Port physical bridge corrections in `inspire_ftp_real_bridge.py` and `run_pico_inspire_bridge.py`: measured initialization, separate participant discovery/disposal, 200ms guard, write-completion pacing, bounded drain, fault evidence, shutdown and explicit range/slew flags. Test native q6, sequence and both-write acknowledgement invariants.
- [x] Extend status with explicit physical feedback ages; validate, consume and record ages without refreshing stale samples on status receipt. Update real/sim publishers and runtime tests.
- [x] Preserve native Inspire PLANNER/frozen hold commands and recording. Test provenance ordering, per-side tracking and repeated mode changes using the existing manager harness and simulator transport. Keep Dex3 admission/ownership unchanged.
- [x] Adapt receive-age exporter/schema fields and mode-aware dataset cleaning from the frozen Inspire tree. Preserve native41 q6 versus legacy43 q7 distinction, locomotion and intentional holds; test dry-run and malformed/stale input handling.
- [x] Run focused Python, native MuJoCo plant, exporter and C++ regressions. Prepare SONIC v1.1 simulator/manager commands and unique capture locations. Live headset outcomes and physical qualification remain explicit pending evidence.

## Remaining hardware TODO — 2026-09-22

- [ ] **Needed to be tested:** Left Grip+A recording after the chord timing fix.
  Restart the workstation manager from the updated checkout, release all controls,
  hold Left Grip first for at least one second, then hold right-controller A for
  at least 0.3 seconds. Verify `Controller action: record`, exporter RECORDING
  acknowledgement, and exactly one start. Release all controls and repeat to
  stop/save; verify IDLE and a saved episode. Also test Left Grip+B abort.
  Software regression tests pass; this corrected sequence has not been confirmed
  on the physical controllers. Holding grip alone no longer expires the window.
- [ ] Capture and inspect PLANNER translational locomotion. The latest inspected
  episode includes PLANNER turning, but no nonzero translation requests.

## Validation boundaries

Synthetic packets and hardware-free DDS fixtures validate software contracts, not physical hand motion. Do not substitute commands for measured state. Preserve source clock provenance and raw errors. No residual loosening or q7 projection for optical poses. Default physical envelope stays conservative; opt-in range/slew settings do not constitute optical hardware qualification.

## Completed validation

- 384 core optical/manager/simulator regressions, 50 physical bridge fixture tests,
  and 71 exporter/cleaner tests passed (505 Python tests total).
- Integrated deployment built; both C++ handoff and hand-decoder regressions passed.
- Ruff and scoped `git diff --check` passed. Hydrated LFS files were excluded from
  source checks/staging; original source snapshots remain unchanged (525 optical,
  550 controller files verified by SHA256 after integration).
- Independent review reproduced an unsafe opening on authorized mode transitions;
  the corrected bridge now holds completed writes with a bounded original-source
  timeout, invalidates acknowledgement, and requires matching new provenance.
  OFF/PAUSE/session changes still open. The reviewer reran 50 bridge tests and
  confirmed the issue resolved.
- Launch commands and qualification limits: `docs/source/inspire_optical_integration.md`.
- The user subsequently confirmed bilateral live MuJoCo tracking and A+X switching. Physical optical qualification remains pending.

## Corrected physical rollout decision

Keep original PC2 SONIC at `/home/unitree/GR00T-WholeBodyControl`, commit
`087f9ac`. User verification establishes no custom one-second mode-switch ramp
in that source or executable. The different installation
`/home/unitree/gear_sonic_inspire_deploy_dcf5e72` contains the fork ramp from
`7877831`. Earlier rebuild advice conflated ramp removal and local reference
handoff adaptations. No rebuild solely for ramp removal is needed. Do not deploy
local handoff changes under this decision or describe them as active on PC2.
Preserve standing-startup smoothing and Inspire initialization. Publish optical
q6 using the separate Python bridge alongside the original SONIC and headless
driver. The earlier C++/model candidate archive remains a local historical
artifact only; selected instructions are in `inspire_optical_real_test.md`.
