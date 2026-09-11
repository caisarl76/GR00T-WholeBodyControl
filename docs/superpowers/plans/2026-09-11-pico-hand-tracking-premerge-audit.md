# PICO optical hand tracking — pre-merge audit, 2026-09-11

PR preparation paused at the user's request. No commit, push, PR creation or
merge was performed by this audit. Target under consideration is
caisarl76/GR00T-WholeBodyControl main (fork/main e9dd3af).

## Integration work before merge

- Branch fix/xr-hand-tracking-watchdog (HEAD 77cbdbe) is 13 commits ahead and
  74 behind fork/main. Manager, exporter, simulator, launcher, deployment and
  pyproject files overlap incoming main changes. Reconcile against current
  main and test the resulting tree; overlap is not proof of a textual conflict.
- Core optical modules, assets, viewer, tests and docs remain untracked; 20
  tracked non-LFS files are modified. Current committed HEAD alone does not
  contain the implementation tested by the user.
- 732 apparent asset/library modifications were checked against their HEAD
  LFS SHA256 pointers: all match. They are materialized assets, not new content.
- Existing XR safety commits pull in tests with an external hard-coded
  /home/jihun/work/unitree_official/xr_teleoperate-hand-tracking-fix import.
  Make the source dependency reproducible or explicitly separate that scope
  from the optical PR. A fresh checkout is not sufficient for those tests.
- Recording tutorial still promises all-mode recording and automatic c/x
  keyboard subscription. New exporter defaults disable that legacy path and
  reject managed packets in legacy mode. Update the migration instructions.
- Add the optical guide to the documentation toctree. Update the old plan's
  'No simulator or robot was launched' claim with user-confirmed runs, while
  preserving missing formal qualification evidence.
- Carry unresolved issue handoffs into PR-visible documentation: outputs/ is
  ignored. The operator guide currently explains the thumb correction without
  identifying its unconfirmed visual result.
- Verify wheel assets: new viewer HTML and Inspire simulation XML are not in
  explicit package-data patterns. Editable-checkout tests do not prove wheel
  inclusion; inspect/build wheel before claiming installed-package support.
- Preserve hand timestamp provenance in episode diagnostics/schema. Raw NPZ
  records timestamp_source, but LeRobot episode hand fields only carry
  source_timestamp_ns, so device and host-content clocks cannot be distinguished.

## Known issues / follow-up decisions

- Open-thumb geometry: explicitly deferred by user; existing correction remains
  in source. It must not be described as a visually verified fix.
- Intermittent missing articulation: user again confirmed working tracking,
  but Sep11 failed capture had nearly straight raw landmarks on both hands.
  Headset visualization was reported to bend correctly. The APK/service/binding
  boundary remains unresolved; no permanent fix was established.
- Right index0 feedback around -0.0007 rad triggers out-of-bounds rejection and
  can block POSE entry. Review measured-feedback tolerance separately from
  commanded target limits; do not broadly weaken checks.
- Human fist is fingertip-geometry retargeting, not a calibrated full-close
  gesture. Open/fist endpoint calibration is a possible future feature, not
  an existing promised behavior.
- User confirmed real G1 + Dex3 on PC2 .223, SONIC v1.1, both hands, higher rate,
  pinch, wrist rotation and open/close. User has not yet reported the requested
  independent tracking-loss/recovery, stop drill, or sustained run results.
- Formal D0 replay qualification remains false in inspected reports. The
  real 500-frame direction replay passed command safety checks but lacks labels,
  duration/source-freeze/opposite-hand evidence and exceeds offline timing
  targets. Do not equate it with full real-hardware qualification.
- Sustained managed recording/save/discard/restart qualification and separate
  Inspire physical validation are not established by Dex3 motion confirmation.

## Validation scope

Fresh focused tests were run during this audit; see final results below.
Tests use existing teleop and data-collection environments. No dependency
installation, simulator, XR connection or real robot command was started.
The initial broad collection failed on missing LeRobot in teleop; exporter
suite was then run in data-collection, avoiding mixed torch/torchvision.
Socket-based test failures under sandbox were rerun with localhost permission.
Full rebuilt C++ deployment and merged-main validation remain future work.

Final fresh results:
- Core selection: 411 passed, 1 skipped, 10 socket-permission failures/errors.
- Localhost rerun: 11 passed (covers all 10 blocked cases plus one already
  passing test); no remaining test failures in this selection.
- Recording exporter in data-collection environment: 16 passed.
- Combined unique result: 437 passed, 1 skipped. The skip is the optional
  cross-repository check without XR_TELEOPERATE_WORKTREE configured.
- These results apply to the current dirty worktree, not a merge with main.

Recommended sequence: finalize PR scope -> reconcile selected changes with
main -> fix clean-checkout/dependency/docs/provenance gaps -> retain explicit
known-issue disclosures -> rerun checks on the resulting commit -> create PR.
Do not claim formal hardware qualification while its evidence remains absent.
