# PICO optical main integration plan

**Goal:** Create a reviewable PR against caisarl76/GR00T-WholeBodyControl main,
using the tested optical implementation and explicit known limitations.
**Architecture:** Apply only the optical worktree delta to main. Preserve main's
existing Inspire/VLA/deployment behavior. The older XR safety v2 commits remain
in the source branch, outside this PR; optical watchdogs remain included.
**Spec:** 2026-09-11-pico-hand-tracking-premerge-audit.md and the user-approved
pre-merge cleanup scope. Python/NumPy, DexPilot 0.4.6, C++/DDS/ZMQ, MuJoCo.

- [x] Preserve original dirty checkout; create integration worktree on main.
- [x] Resolve manager/exporter/simulator/deploy overlaps with regression checks.
- [x] Preserve device-vs-host hand clock provenance in managed episodes.
- [x] Include viewer/model assets in wheel; inspect built wheel.
- [x] Update recording migration guide, toctree, qualification and known issues.
- [x] Run core/native/recording/simulation tests and compile deployment.
- [x] Review final diff; commit selected source/assets without local LFS noise.
- [x] Push integration branch and create PR; do not merge automatically.

No PC2 installation, headset modification, or real control starts are part of
this integration. Open-thumb geometry, upstream intermittent articulation,
right measured-zero tolerance and full-fist calibration remain documented work.


## Integration verification (2026-09-11)

- Python core: 328 passed and 599 subtests passed; the sole sandbox-blocked
  localhost feedback test passed separately (1 passed).
- MuJoCo startup, Inspire source ownership, plant and ZMQ regressions:
  75 passed using the simulation environment, without launching live control.
- Manager integration review also covered main's IsaacTeleop readers and
  Inspire controller contract (91 passed and 599 subtests passed).
- Release C++ build completed with BUILD_DEPLOY_TESTS=ON and BUILD_ROS2=OFF.
  Used existing system GoogleTest via FETCHCONTENT_SOURCE_DIR_GOOGLETEST;
  no dependency installation was needed.
- CTest hand-field decoder passed with sanitizer outside sandbox tracing.
  C++ unit executable: 9 passed; encoder parity skipped because
  SONIC_PARITY_CASE_JSON was absent. The existing FK test requires the
  untracked reference/bones_072925_test fixture and could not be validated.
- Built a wheel without downloading dependencies and verified 12 viewer,
  retargeter and Inspire model assets byte-for-byte against their sources.
  Simulation still requires the checkout's existing LFS mesh assets.
- Independent manager/native/runtime review found no substantive regressions.
  Recording/deploy review found a legacy exception-cleanup data-loss regression.
  Fixed it with eight exit/failure regressions (33 exporter tests passed); the
  reviewer accepted the fix. Managed recording is unchanged.
- Launcher regressions: 6 passed, including dry-run with an occupied ZMQ port.
- Owned source whitespace checks pass. Vendored hash-pinned hand assets retain
  upstream whitespace/CRLF; the optional Unity patch retains diff context.

Counts above overlap; they are separate suites, not an additive total.
No real robot, PC2 install or headset changes were performed for this PR.

Final exporter/runtime/recorder and main target-frame conversion suite: **107 passed**
after the legacy cleanup fix. Staged scope: 71 files; no model binaries, SDK
binaries, raw captures, or LFS materialization changes.

PR: https://github.com/caisarl76/GR00T-WholeBodyControl/pull/9
Integration commit: `c5c6d8c`. Original tested worktree is preserved; merge is
a separate action.
