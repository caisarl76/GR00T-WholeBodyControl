# PICO Manipulation Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a MANIPULATION teleop mode in which hand-only operator motion moves only the robot arms and hands. Waist and height come from the joysticks, and walking happens only in FROZEN mode. The plan includes the safety interlocks from the PRD and the gates that must pass first.

**Architecture:**
- **Phase A (gates):** probe and logging scripts, plus one isolated IK prototype. No integrated feature code.
- **Phase B (feature), streamer side:** small pure-Python modules under `gear_sonic/utils/teleop/manipulation/` (chords, sticks, tracking, hands, lock gate, params), composed by a `ManipulationStreamer` that `pico_manager_thread_server.py` calls under `--manipulation`.
- **Phase B (feature), deploy side:** two header-only C++ units (`waist_lock.hpp`, `input_watchdog.hpp`) with gtests, wired into `zmq_manager.hpp` / `g1_deploy_onnx_ref.cpp`.
- **Messages:** existing optional-field ZMQ planner messages, plus 3 new optional fields.

**Tech Stack:** Python 3.11 (numpy, scipy, pink/pinocchio via `decoupled_wbc` IK, pyzmq, msgpack), pytest; C++20, gtest (`run_tests` target, auto-globs `unit_tests/*.cpp`); MuJoCo sim (`gear_sonic/scripts/run_sim_loop.py`).

**Spec:** `docs/superpowers/specs/2026-11-09-pico-manipulation-mode-prd.md` (approved). Section numbers (§) below refer to it. Read it before any task.

## Global Constraints

- **Gate order is mandatory:**
  1. Tasks 1–4 (G0, G1, G3 plus infra) come first.
  2. Task 5 (G2) is isolated: not imported by the streamer or the deploy until Task 13.
  3. **Task 6 is a hard stop:** no Phase B task starts until PRD §11 has G0, G2, G3 and **G1b** recorded as passed. G1 stays recorded as FAIL; since 2026-11-11 its walking role is carried by G1b (Task 3b), and its other measured values (`h_default`, `idle_honors_height`, `cmd_to_z`, `h_walk_min = None`) are still used.
- Sessions launched **without** `--manipulation` (streamer) or without `--waist-lock` / `--input-watchdog-ms` (deploy) must behave exactly as today, including the A/B + left-grip recording shortcuts. The only wire difference allowed in such sessions is **additive** `g1_debug` keys (`t_mono_ns` from Task 1; `waist_lock_alpha`, `lock_ack_seq`, `waist_lock_enabled`, `watchdog_latched` from Tasks 11–12). Planner, pose, command and manager_state payloads stay byte-identical.
- Legs are never hard-locked. Only MuJoCo joints 12/13/14 (waist) are ever overridden.
- Every numeric default lives in `ManipulationParams` (Task 6) or the C++ constructors' defaults. No magic numbers elsewhere.
- Gate-derived values (`h_default`, `h_walk_min`, `idle_honors_height`, per-backend loss signal) come only from PRD §11.
- Until G3 says otherwise, a backend has `stick_control_enabled = False`. That disables walking, turning, waist and height on it, leaving only arm and hand tracking.
- The hand blend starts from the **actual last-sent hand command**, never from a recomputed `f(trigger, grip)`.
- Python tests: `source .venv_teleop/bin/activate && pytest gear_sonic/tests/manipulation -v`.
- C++ tests: `cd gear_sonic_deploy && just build`, then run the built `run_tests` binary (in `target/release/` of the g1_deploy_onnx_ref project) with `--gtest_filter=<Suite>.*`.
- **No commits and no branch changes without the user's explicit authorization for this execution.** Each task ends with a **Checkpoint** step: report the changed files, the test commands run and their output, and the suggested commit message. If the user has authorized commits, commit on the current branch with that message plus the session's attribution line; otherwise leave the changes in the working tree. Never use bare `git stash`.
- Every sim run that measures robot motion (gates, `e2e_stack`, the Task 16 campaign) launches `run_sim_loop.py --no-elastic-band`. The default support band holds the pelvis at about 0.96 m and masks height and torso motion (found in G1).
- Sim gates and the campaign run on **one host**, so every source timestamp is on the same `CLOCK_MONOTONIC` (`time.monotonic_ns()` in Python, `std::chrono::steady_clock` in the deploy). The Task 1 sync contract applies to every analysis.

## Review Focus

1. **Recording shortcuts stay as they are (user decision, 2026-11-10).** A + left grip > 0.5 toggles recording and B + left grip > 0.5 aborts it, edge-triggered (`pico_manager_thread_server.py:2127-2132`), in every session including `--manipulation`. Known overlap, the same as in today's VR_3PT: while the left grip is held (for example, holding an object), a chord containing A (ABXY) toggles recording and one containing B (B+Y, ABXY) aborts it. This is accepted and documented in the user guide. Test: Task 14 `test_manipulation_recording_shortcuts_unchanged` pins it.
2. **Headset disconnect mid-MANIPULATION.** The reader timestamp stops, the streamer stops sending, and the deploy watchdog latches. On reconnect the operator may be in a different pose. Expected: no arm command uses the new controller pose until both arms are re-anchored. The first commands after resume repeat the held command, and later deltas are relative to the new anchor. Walking stays latched until the operator's sticks are neutral. Tests: Task 13 (`test_reader_stall_then_resume_reanchors_both_arms`) and Task 16 (9e2, integrated).
3. **IK target outside reach.** The operator reaches farther than the robot can. Expected: IK returns the closest solution within joint limits and never raises. A failure counts toward the 0.5 s loss rule only if it is a solver error, not a reach limit. A large FK error is a reach limit only if re-seeding from the posture doesn't reduce it; otherwise it is a stall and is recovered (PRD §6.6). Tests: Task 5 (`test_unreachable_target_returns_limited_solution`), Task 13 (`test_ik_reach_limit_is_not_failure`, `test_ik_stall_regression_g2_sweep`).
4. **`waist_cmd` at its limit while the stick is held.** Expected: it clamps silently, with no wrap-around and no growing internal value, so moving the stick back responds immediately. Test in Task 8 (`test_clamped_accumulator_responds_immediately_on_reverse`).
5. **Re-entry into MANIPULATION after walking turned the robot.** Heading changed in FROZEN. Expected: the new anchor uses the current headset yaw, so operator-forward maps to torso-forward. Test in Task 9 (`test_anchor_uses_current_headset_yaw_on_reentry`).

---

## Phase A — Gates (no feature code)

### Task 1: Sim ground-truth publisher + synchronized gate recorder

**Files:**
- Modify: `gear_sonic/utils/mujoco_sim/configs.py`. Add to `SimLoopConfig`: `publish_ground_truth: bool = False`, `ground_truth_port: int = 5560`, `gt_bodies: list[str] = []`.
- Modify: `gear_sonic/utils/mujoco_sim/base_sim.py` (`BaseSimulator`: when enabled, PUB once per sim step)
- Modify: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/output_interface/zmq_output_handler.hpp` (`pack_combined_state`: add key `t_mono_ns` (uint64, `steady_clock::now()` at pack time) and bump the map size). This is instrumentation, not feature code.
- Create: `gear_sonic/scripts/gates/__init__.py`, `gear_sonic/scripts/gates/recorder.py`
- Test: `gear_sonic/tests/manipulation/test_gate_recorder.py`

**Interfaces:**
- Produces:
  - ZMQ topic `sim_gt` on `tcp://*:5560`. Payload is msgpack:
    - `t_mono_ns` (int, `time.monotonic_ns()` at publish);
    - `t_sim` (float, `mj_data.time`);
    - `pelvis_pos` [3] and `pelvis_quat_wxyz` [4], from `qpos[0:3]` and `qpos[3:7]`;
    - `bodies`: `{name: {"pos": [3], "quat_wxyz": [4]}}` for each name in `gt_bodies`, from `mj_data.xpos` / `xquat`. Task 16 uses this for the wrists and the object.
  - `g1_debug` gains `t_mono_ns`.
  - **Source time per stream**, which the recorder uses:
    - `sim_gt.t_mono_ns`;
    - `g1_debug.t_mono_ns`;
    - `pose.timestamp_monotonic` (seconds; the reader sample time, already sent today);
    - `manip_debug.t_mono_ns` (Task 13);
    - `manager_state` has none and is used for event ordering only.
  - `class GateRecorder(streams: dict[str, tuple[str, int, str]])`, mapping a name to `(host, port, topic)`.
    - `.poll_all() -> None`
    - `.save(path: str) -> None`. Per stream it writes `<name>_recv_ns` (receipt `time.monotonic_ns()`), `<name>_src_ns` (source time, or −1 if the stream has none) and `<name>_raw` (object array of decoded dicts). Also `<name>_t_sim` for `sim_gt`.
  - `@dataclass(frozen=True) class StreamReq: period: float; absent: tuple[tuple[str, str], ...] = ()` — `absent` lists **expected-absence intervals** as `(start_marker, end_marker)` names from the `markers` stream.
  - `def check_sync(npz, required: dict[str, StreamReq], window: tuple[float, float]) -> SyncReport`. `SyncReport` is a dataclass: `ok: bool`, `reasons: list[str]`, `skew_p99_ms: dict`, `max_gap_ms: dict`, `rtf: float | None`.
  - `def align(npz, signals: dict[str, Callable[[dict], np.ndarray]], window, hz: float = 100.0) -> dict[str, np.ndarray]`. It resamples each extracted signal onto one source-time grid.
  - Decoders: `g1_debug` → `zmq_state_subscriber._unpack_msgpack_zmq`; `pose` / `planner` / `manager_state` / `manip_debug` → `run_data_exporter.unpack_pose_message`; `sim_gt` → msgpack.
- **Sync contract** (constants in `recorder.py`):
  - **Alignment** uses source time only. Receipt time is used only to measure skew.
  - **Skew** = `recv_ns − src_ns` per message. Reject the window if any stream has a skew below −1 ms (clock mismatch), a p99 above 20 ms, or a max above 50 ms.
  - **Gaps:** reject if a stream's source-time gap inside the window exceeds `max(3 × period, 0.1 s)`, **unless** the gap lies inside one of that stream's declared `absent` intervals. The interval is widened by `max(3 × period, 0.1 s)` at the start and by 1.0 s at the end, to allow for restart.
  - **Expected absence must be observed:** for each declared interval, the stream must have **no** messages between `start_marker + max(3 × period, 0.1 s)` and `end_marker`. Otherwise reject with `fault not applied`.
  - **Completeness:** every required stream must cover `[window[0] − 1.0 s, window[1]]` (the baseline plus the analysis), apart from its declared absences.
  - **Independent evidence never gets absences.** `sim_gt`, `g1_debug` (recorded directly from the deploy, never through a fault proxy) and `markers` must be continuous in every scenario. `check_sync` rejects an `absent` declaration on them.
  - **Real-time factor:** `rtf` = Δ`t_sim` / Δ`t_mono` over the window. Reject if `rtf < 0.9`, because the deploy runs on wall-clock time and a slow sim distorts every timing metric.
  - **Interpolation:** linear for scalars, and unwrap-then-linear for angles. No extrapolation past one period. Grid points without data are NaN, and analyzers treat NaN as missing.

- [ ] **Step 1:** Write the tests:
  - `test_recorder_timestamps_and_decodes`: two tcp PUBs send 3 `sim_gt` and 2 `g1_debug` messages, each with `t_mono_ns`. Poll, save, reload. Assert the counts, `src_ns` equal to what was sent, `recv_ns ≥ src_ns`, and the decoded `pelvis_pos`.
  - `test_sync_rejects_missing_stream`, `test_sync_rejects_gap` (a 150 ms hole in a 20 ms stream), `test_sync_rejects_negative_skew`, `test_sync_rejects_high_skew` (60 ms), `test_sync_rejects_partial_coverage` (the stream starts 0.5 s into the baseline), `test_sync_rejects_slow_sim` (rtf 0.8).
  - `test_align_uses_source_time`: two streams with the same source times but receipt times offset by 15 ms → aligned values are identical.
  - `test_align_no_extrapolation`.
  - `test_declared_absence_accepted`: `manip_debug` silent between markers `kill` and `restart` → ok.
  - `test_gap_outside_declared_absence_rejected`: a 300 ms `manip_debug` gap 2 s after `restart` → rejected.
  - `test_declared_absence_not_observed_rejected`: markers declare a kill, but `manip_debug` keeps arriving → rejected (`fault not applied`).
  - `test_absence_on_independent_stream_rejected`: an `absent` declaration on `g1_debug` → rejected, regardless of the data.
- [ ] **Step 2:** Run `pytest gear_sonic/tests/manipulation/test_gate_recorder.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement the recorder, the sim publisher (non-blocking PUB, created only when `publish_ground_truth`) and the deploy `t_mono_ns` key.
- [ ] **Step 4:** Run the tests. Expected: PASS. Then:
  1. Run `cd gear_sonic_deploy && just build`.
  2. Launch the sim with `--publish-ground-truth`, plus the deploy in sim.
  3. Record 10 s.
  4. Run `check_sync` with `{"sim_gt": StreamReq(<sim step period from SimLoopConfig>), "g1_debug": StreamReq(0.02)}`. Expected: `ok`, and the skew p99 is printed (proving both clocks agree).
- [ ] **Step 5:** Checkpoint (suggested message `feat(gates): sim ground truth, source timestamps and sync-checked recorder`).

### Task 2: G0 — twist diagnosis

**Files:**
- Create: `gear_sonic/scripts/gates/g0_twist_log.py` (CLI: `--duration 60 --out g0.npz`). It records streams `pose` (5556), `g1_debug` (5557), `sim_gt` (5560), `markers` (operator keypresses) and `operator_ref` (below).
- Create: `gear_sonic/scripts/gates/g0_analyze.py` (CLI: `g0.npz`)
- Create: `gear_sonic/scripts/gates/g0_input_path.md` (Step 5 inspection record)
- Test: `gear_sonic/tests/manipulation/test_g0_analyze.py`

**Interfaces:**
- Consumes: `GateRecorder`, `check_sync`, `align` (Task 1).
- **What G0 can and cannot claim.** The PICO body estimate is derived from the same devices the operator wears, so it can't prove the operator was still. G0 therefore reports two separate results:
  - **`robot_side`** (hops B and C). It compares the reference the robot received with what SONIC targeted and what the robot did. This is fully determined by recorded data. **The gate pass depends on this result.**
  - **`reference_side`** (hop A). Is twist present in the PICO reference? Attribution to the PICO solver needs **independent operator-motion evidence**. Without it, the result is `"twist_in_reference_operator_unverified"` (or `"none"`) and is never reported as `"pico"`.
- **`operator_ref` stream (independent operator evidence).**
  - At startup, `g0_twist_log.py` lists the backend functions whose names mention `tracker` or `motion`, and records any raw (non-solver) per-device pose it finds for an **extra tracker strapped to the operator's sternum**, at ≥ 50 Hz, with `t_mono_ns`.
  - If the backend gives no raw pose for that tracker while body tracking runs, the stream is absent, and the log prints `operator_ref unavailable`.
  - The headset yaw is recorded too, as `headset_yaw`. It is supporting evidence only: a still head does not prove a still chest.
- **Calibration segment (frame and joint-mapping check).** This runs in the same recording, before the reaches, cued by operator keypress markers:
  - `turn_body`: the operator turns the whole body ±30°, three times;
  - `twist_chest`: the operator twists the chest ±20° with the hips still, three times.

  A level's comparison is **valid** only if, over its calibration segment, the two quantities being compared have correlation ≥ 0.9, the same sign, and a least-squares gain in [0.8, 1.2], fitted at the best lag L ∈ [0, 0.3] s (the same bound as the reaches; ruling 2026-11-10). The gain and L are reported. An invalid level makes every hop at that level `"unmapped"`.
- **Signals.** All yaws are in degrees, each relative to its own baseline (the 1.0 s before the segment). Baseline subtraction removes constant offsets only. Equivalence comes from the calibration check, not from the subtraction.
  - **PICO reference (`pose` stream).** Use the last frame of each message, stamped `timestamp_monotonic`.
    - `ref_pelvis_yaw` = yaw of `body_quat_w`. This is the root orientation (wxyz, z-up after `smpl_root_ytoz_up`), not `smpl_joints`, which are root-local positions.
    - `ref_spine_twist` = the twist of R(spine1)·R(spine2)·R(spine3) about the SMPL local up axis (+y), by swing-twist decomposition, from `smpl_pose` body-pose indices 2, 5 and 8 (SMPL joints 3, 6, 9).
    - `ref_chest_twist_joints` = yaw of the root-local shoulder line (SMPL joints 16→17) relative to the hip line (1→2), about the root-local up axis, from `smpl_joints`. **This is the chest reference used for robot_side and reference_side**: the policy's encoder (mode `smpl`, `policy/release/observation_config.yaml`) reads `smpl_joints` positions and the root orientation, never `smpl_pose` (see `g0_input_path.md`). `ref_spine_twist` is reported as a diagnostic only (ruling 2026-11-10).
    - **Reach segmentation:** use the right wrist `smpl_joints[21]`. A reach starts at > 10 cm displacement and ends when it returns within 3 cm.
  - **Deploy target (`g1_debug`):** `tgt_pelvis_yaw` = yaw of `base_quat_target`. This is the streamed `body_quat_w` after the deploy's heading correction (`output_interface.hpp:197-215`), so hop B at pelvis level measures the deploy path, not the policy.
  - **No waist target exists in POSE mode.** `body_q_target` is copied from the streamed `joint_pos` (`output_interface.hpp:191-195`), which the streamer fills with zeros except the wrists (`pico_manager_thread_server.py:1420-1476`). The policy consumes `smpl_pose` directly, so SONIC's internal chest target is not observable. `body_q_target[12]` is recorded but used only by the placeholder guard below.
  - **Measured:** `meas_pelvis_yaw` = yaw of `base_quat_measured`; `meas_waist_yaw` = `body_q_measured[12]`; `sim_pelvis_yaw` from `sim_gt` (IMU cross-check); `meas_torso_yaw` = yaw of `body_torso_quat` (cross-check only).
  - **Operator:** `op_chest_yaw` from `operator_ref`, when present.
- **Like-for-like pairs.** Each hop compares the same kind of quantity, and each pair's validity comes from the calibration check:

| Hop | Pelvis level | Chest level |
|---|---|---|
| A (operator vs PICO reference) | — | `op_chest_yaw` vs `ref_pelvis_yaw + ref_chest_twist_joints` (calibrated on `turn_body` + `twist_chest`) |
| B (PICO reference → deploy target) | `ref_pelvis_yaw` vs `tgt_pelvis_yaw` (calibrated on `turn_body`) | — (not observable: no waist target in POSE mode) |
| C (deploy target → robot) | `tgt_pelvis_yaw` vs `meas_pelvis_yaw` | — |
| BC (PICO reference → robot, end to end) | — | `ref_chest_twist_joints` vs `meas_waist_yaw` (calibrated on `twist_chest`; 3 SMPL spine joints vs the 1 G1 waist yaw) |

  - **Placeholder guard.** A target channel whose peak-to-peak over its calibration segment is < 0.5° while the reference moves > 10° is `"target_unobservable"`. It is never fitted, and any pair using it is `"unmapped"`. This catches a future change that routes a placeholder into a compared target.
  - **Chest result is a residual, not an attribution.** BC compares the streamed chest twist (from joint positions) with the measured waist yaw. A BC twist is recorded as `chest_residual`: an end-to-end, policy/robot-side residual. It doesn't reveal SONIC's internal chest target or isolate why the policy produced the motion. What lies on that path is recorded by the Step 5 inspection, not assumed.

  - Hop B, C and BC peaks per reach are peak |Δy(t) − g·Δx(t − L)|. `g` is the calibration gain, and the lag L ∈ [0, 0.3] s minimizes RMS over [reach start, reach end + 2.0 s].
  - **Reference twist** per reach = peak |Δref| at each level.
  - **Hop A** per reach (only with `operator_ref`): `op_still` = peak |Δop_chest_yaw| < 1°. Then `pico_added` = peak |Δref_chest − Δop_chest| > 2°.
- **Produces:** `analyze_twist(npz) -> dict`:
  - `calibration`: per pair, the correlation, gain and valid flag;
  - `reaches`: per reach, the reference twist, the hop B/C/BC peaks, `L`, `imu_vs_sim_deg`, and `op_still` / `pico_added` (or `None`);
  - `sync`;
  - `robot_side`: `"none"`, `"inconclusive"`, or a `+`-joined subset of `pipeline`, `sonic` and `chest_residual`, in that order;
  - `reference_side` ∈ {`"none"`, `"pico"`, `"operator_moved"`, `"twist_in_reference_operator_unverified"`, `"inconclusive"`};
  - `reasons`.
- **Rules.**
  - A hop introduces twist if its peak exceeds 2° at a valid level.
  - **`robot_side`** is the set of hops (B → `pipeline`; C → `sonic`; BC → `chest_residual`) that introduce twist, joined with `+`, if all reaches agree. Otherwise it is `"inconclusive"`. It is also `"inconclusive"` if any of these holds:
    - `check_sync` fails for `pose` (0.02), `g1_debug` (0.02) and `sim_gt` (the sim step) over the calibration window or any reach window;
    - fewer than 5 reaches are detected;
    - any B/C/BC pair is `"unmapped"`;
    - any B/C/BC peak lies in [1.5°, 2.5°];
    - `imu_vs_sim_deg` > 1°;
    - there is any NaN in a window.
  - **`reference_side`:**
    - `"none"` if the reference twist is < 1.5° at both levels in all reaches;
    - otherwise, without `operator_ref` (or with hop A `"unmapped"`): `"twist_in_reference_operator_unverified"`;
    - with it: `"pico"` if every reach has `op_still` and `pico_added`; `"operator_moved"` if any reach lacks `op_still`; `"inconclusive"` on disagreement or in the ambiguous band.
  - **Gate pass** = `robot_side != "inconclusive"`. `reference_side` is recorded as is. It doesn't block, because MANIPULATION doesn't use the pose stream.

- [ ] **Step 1:** Write the tests, using synthetic npz files built with the Task 1 save format:
  - `test_body_quat_w_yaw_not_joint_positions`: `body_quat_w` turns 10° while `smpl_joints` stay fixed → reference twist 10°.
  - `test_spine_twist_swing_twist`: a pure 8° spine twist with a 20° forward bend → 8° ± 0.1°.
  - `test_calibration_valid_and_gain`: measured waist = 0.9 × spine twist → valid, gain 0.9; measured waist = 0.5 × spine twist → gain out of range → `"unmapped"` → `robot_side == "inconclusive"`.
  - `test_placeholder_target_unobservable`: `body_q_target` constant 0 and `base_quat_target` constant while the reference turns 30° → that pair is `"target_unobservable"` and `robot_side == "inconclusive"`.
  - `test_calibration_sign_flip_invalid`.
  - `test_robot_side_pipeline`: reference pelvis flat, target pelvis 6°, measured follows target → `"pipeline"`.
  - `test_robot_side_sonic`: reference and target pelvis flat, measured pelvis 8° → `"sonic"`.
  - `test_robot_side_chest_residual`: reference spine flat, measured waist yaw 6°, pelvis level flat → `"chest_residual"`, never `"sonic"`.
  - `test_robot_side_label_order`: pelvis target 6° off the reference and a chest residual of 6° → `"pipeline+chest_residual"`.
  - `test_robot_side_none_when_target_tracks_reference_with_lag`: target = reference delayed by 150 ms → `"none"`.
  - `test_reference_unverified_without_operator_ref`: reference twist 10°, no `operator_ref` → `reference_side == "twist_in_reference_operator_unverified"`, never `"pico"`.
  - `test_reference_pico_with_still_operator`: `operator_ref` flat, reference 10° → `"pico"`.
  - `test_reference_operator_moved`: `operator_ref` turns 6° and the reference follows → `"operator_moved"`.
  - `test_inconclusive_on_gap`, `test_inconclusive_on_disagreeing_reaches`, `test_inconclusive_on_ambiguous_band` (2.2°), `test_inconclusive_on_imu_sim_mismatch`, `test_inconclusive_on_missing_stream`, `test_inconclusive_on_four_reaches`, `test_inconclusive_on_missing_calibration_segment`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement both scripts.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Inspect the active deploy code path, from the received `smpl_pose` / `body_quat_w` to the policy observation buffers, for the configuration G0 runs (`deploy.sh --input-type zmq_manager sim` and its observation config). Write `gear_sonic/scripts/gates/g0_input_path.md`: each hop as `file:line`, which config selects it, and every transformation applied to the spine rotations and to the root (heading, frame, reordering, filtering). Mark each hop "verified by reading" or "not verified" (for example, a branch selected at runtime that couldn't be traced). The G0 §11 row cites this file. It doesn't change `robot_side`; it records what `chest_residual` spans.
- [ ] **Step 6:** Run G0 in sim with today's POSE mode:
  - Terminals per `docs/source/user_guide/teleoperation.md`, from the worktree root with `PYTHONPATH=$PWD` (the worktree has no venvs; use these absolute interpreters):
    - sim: `/home/jihun/work/GR00T-WholeBodyControl/.venv_sim/bin/python gear_sonic/scripts/run_sim_loop.py --no-elastic-band --publish-ground-truth`. The support band must be off; otherwise it holds the pelvis and masks robot motion (found in G1).
    - deploy: `cd gear_sonic_deploy && (printf 'y\n'; sleep 100000) | bash deploy.sh --input-type zmq_manager sim`;
    - streamer: `/home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python gear_sonic/scripts/pico_manager_thread_server.py --manager`;
    - recorder: `/home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python -m gear_sonic.scripts.gates.g0_twist_log --duration 150 --out g0.npz`;
    - analysis: `/home/jihun/anaconda3/envs/sonic/bin/python -m gear_sonic.scripts.gates.g0_analyze g0.npz`.
  - If a spare tracker is available, strap it to the sternum.
  - Sequence, each part marked by keypress: 3 s still; `turn_body`; `twist_chest`; 3 s still; then 5 hand-only right reaches, each followed by ≥ 3 s still.
  - Run `g0_analyze.py g0.npz`.
  - Fill the G0 row of PRD §11 with `robot_side`, `reference_side`, the calibration gains and any `target_unobservable` channels, the Step 5 input-path summary (with any "not verified" hops), the per-reach peaks and lags, the sync report, and whether `operator_ref` was available. `robot_side == "inconclusive"` is not a pass: fix the cause it names and re-run.
- [ ] **Step 7:** Checkpoint (suggested message `feat(gates): G0 staged twist diagnosis with mapping calibration`).

### Task 3: G1 — height probe and walking band

**Files:**
- Create: `gear_sonic/scripts/gates/g1_height_probe.py`. It drives the deploy directly over ZMQ (PUB, port 5556) with `build_command_message(start=True, planner=True)` and `build_planner_message`, without PICO, while recording `sim_gt` and `g1_debug`.
- Create: `gear_sonic/scripts/gates/g1_analyze.py`
- Test: `gear_sonic/tests/manipulation/test_g1_analyze.py`

**Interfaces:**
- Consumes: `GateRecorder`; `gear_sonic/utils/teleop/zmq/zmq_planner_sender.py:build_planner_message`.
- Produces: `analyze_height(npz) -> dict` with keys:
  - `idle_honors_height: bool`: pelvis z changes > 3 cm between the height 0.8 and 0.4 commands in IDLE;
  - `cmd_to_z: list[tuple[float, float]]`;
  - `h_default: float | None`: the command value whose settled pelvis z matches the −1-sentinel standing z, linearly interpolated on `cmd_to_z`. `None` when calibration fails (see below);
  - `band_assumption: str | None`;
  - `walk_table: dict[float, dict[str, list[bool]]]`: per tested height, per direction (`fwd`, `back`, `turn`), the pass/fail of each of the 5 runs;
  - `h_walk_min: float | None`;
  - `gate_pass: bool` and `reasons: list[str]`.

- **`h_default` calibration fails (→ `None`)** when the standing z (from the −1 sentinel) is not bracketed by `cmd_to_z`, when `cmd_to_z` is not monotonic over the bracketing pair, or when the standing z varies by > 1 cm across the three −1 settles in the probe.
- **Walk trial:** one direction at one height for **10 s of that motion alone**: forward (movement 0.3 m/s), backward, or turning in place. Then 3 s stationary. Each (height, direction) pair is tested **5 times**. A trial passes if pelvis z > 0.3 m throughout and pelvis z stays within ±3 cm of that height's stationary z. A height passes only if all 15 trials pass.
- **Tested heights:** the grid {0.80, 0.75, …, 0.40} (step 0.05), plus the −1 sentinel, plus `h_default` rounded to 0.01.
- **Band:** scan the grid downward from 0.80 and stop at the first failing height. The band is [lowest passing height before that stop, 0.80].
  - `h_walk_min` = lowest passing height + 0.03. **If that exceeds 0.80** (e.g. only 0.80 passed), the band is **unknown**: `h_walk_min = None`, and the PRD §6.3 rule `H_walk_min := H_default` applies.
  - A failure at any height means **no height at or below it is in the band**, even if a lower height passed.
  - **Interpolation is an assumption, not evidence.** Heights between two passing grid points are *assumed* walkable because their neighbours passed. They were not tested. `analyze_height` returns `band_assumption = "interpolated between grid points, step 0.05"`, and §11 records it next to `h_walk_min`. The PRD §6.3 gantry confirmation at `H_walk_min` is what turns the assumption into evidence on the robot. Heights below the lowest tested point are never in the band.
  - If 0.80 fails, `h_walk_min = None`.
- **`gate_pass`** requires all of:
  - `h_default` is not `None`;
  - the −1 sentinel passes;
  - `h_default` itself passes the walk trial.

  `h_walk_min = None` alone is still a pass, because the PRD §6.3 unknown-band rule applies (`H_walk_min := H_default`).

- [ ] **Step 1:** Write tests on synthetic npz data:
  - `test_idle_ignores_height` → `idle_honors_height is False`;
  - `test_h_default_interpolated`, with map [(0.6, 0.62), (0.8, 0.78)] and standing z 0.74 → `h_default ≈ 0.75`;
  - `test_h_default_none_when_not_bracketed`, `test_h_default_none_when_standing_z_drifts`;
  - `test_walk_band_margin`: everything passes down to 0.50, and 0.45 fails → `h_walk_min == 0.53`;
  - `test_gap_in_band_truncates`: 0.80–0.70 pass, 0.65 fails, 0.60–0.40 pass → `h_walk_min == 0.73`;
  - `test_one_direction_fail_fails_height`: turning fails 1 of 5 at 0.55 → 0.55 is not in the band;
  - `test_only_top_passes_band_unknown`: 0.80 passes, 0.75 fails → `h_walk_min is None` (not 0.83).
  - `test_band_assumption_reported`.
  - `test_top_fails_band_none`: 0.80 fails → `h_walk_min is None`, and `gate_pass` depends only on the sentinel and `h_default`;
  - `test_untested_below_grid_not_in_band`;
  - `test_gate_fails_when_default_height_walk_fails`;
  - `test_gate_fails_when_h_default_none`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. Probe sequence:
  1. Settle 3 s at −1.
  2. In IDLE and IDLE_SQUAT each: steps to {0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2} with 3 s holds, then a 0.1 m/s ramp from 0.8 to 0.2 and back.
  3. Run IDLE↔IDLE_SQUAT switches at 0.5.
  4. Settle at −1 twice more: once between stages and once at the end.
  5. Run the walk trials in this order: the −1 sentinel, then `h_default`, then the grid from 0.80 downward. Each height gets 5 trials per direction.

  On a fall (pelvis z < 0.3 m), record that trial as failed, mark every remaining trial as not run, save, and exit with a message naming the height. Not-run trials count as failures. Because of the order, the band above the fall is already complete.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Run the probe in sim, then `g1_analyze.py`. Fill PRD §11 G1 with `idle_honors_height`, `h_default`, `cmd_to_z`, `h_walk_min`, the full `walk_table`, `gate_pass` and the reasons.
- [ ] **Step 6:** Checkpoint (suggested message `feat(gates): G1 height probe and walking band`).

### Task 3b: G1b — restricted walking-height probe (added 2026-11-11)

G1 failed and stays recorded as failed. G1b checks only the restricted PRD §6.3 behaviour (walking at standing height uses −1). For pass criteria, see PRD §9 G1b.

**Files:**
- Create: `gear_sonic/scripts/gates/g1b_walk_handoff_probe.py` (CLI `--out outputs/gates/g1b.npz`, `--h-default 0.7984`). Reuse the planner-message helpers, `GateRecorder` and the motion-evidence logic of `g1_height_probe.py` / `g1_analyze.py` by importing them, not by copying.
- Create: `gear_sonic/scripts/gates/g1b_analyze.py` (CLI `g1b.npz`; JSON on stdout, summary on stderr).
- Test: `gear_sonic/tests/manipulation/test_g1b_analyze.py`

**Interfaces:**
- Consumes: Task 1 `GateRecorder`, `check_sync`, `align`; Task 3 trial and motion-evidence helpers.
- Produces: `analyze_handoff(npz) -> dict` with:
  - `trials`: per direction × run: `pass`, `max_dev`, `disp_m` / `dyaw_rad`, `fail`;
  - `switch_steps`: per switch, `{kind: "to_sentinel"|"to_height", dz_m}`;
  - `falls`, `sync`, `gate_pass`, `reasons`.

- [ ] **Step 1:** Write tests on synthetic npz data:
  - `test_handoff_pass`;
  - `test_switch_step_over_1cm_fails` (1.2 cm);
  - `test_trial_dev_over_3cm_fails`;
  - `test_no_motion_evidence_fails`;
  - `test_sync_gap_in_switch_window_fails`;
  - `test_fall_fails`;
  - `test_missing_switch_marker_fails`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. Probe sequence (sim launched with `--no-elastic-band --publish-ground-truth --gt-bodies pelvis`):
  1. Settle 3 s at −1.
  2. For each of fwd, back and turn (×5 each):
     - Ramp the IDLE_SQUAT height from 0.60 to `h_default` at 0.1 m/s, then hold 1 s.
     - `switch:to_sentinel`: IDLE −1 for 0.5 s.
     - Walk 10 s with −1 (SLOW_WALK 0.3 m/s; turn uses the static mode at 0.5 rad/s, as in G1).
     - Stop: IDLE −1 for 0.5 s.
     - `switch:to_height`: IDLE_SQUAT `h_default`, held 3 s.

  Each phase is marked. A switch step is the median z over the 0.5 s after the marker minus the median over the 0.5 s before it, so each switch window must be stationary and synced.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Run the probe in sim, then `g1b_analyze.py`. Fill the PRD §11 G1b row with measured values only.
- [ ] **Step 7 (added 2026-11-11, offline, no sim):** Derive the still-standing thresholds. Use the recorded `g1_debug_raw` (leg `body_dq`, `base_ang_vel`) and `sim_gt` of `g1b_runA.npz` and `g1b_runB.npz`.
  - Slide a 1 s window at 100 Hz over every static segment and every walk-stop segment. Label each window stationary by the sim-ground-truth criteria: z slope < 2 mm/s, xy < 2 cm/s, yaw < 0.05 rad/s.
  - Choose `still_dq_max` and `still_gyro_max` as large as possible with **zero** windows that pass the feedback check but are not stationary by ground truth.
  - Report:
    - the thresholds;
    - false "still" (must be 0) and false "moving" counts;
    - the per-trial delay between ground-truth stillness and check pass.
  - Test: `test_still_thresholds_zero_false_still`, on synthetic data where one window moves only in xy.
  - Fill the §11 G1b note. If zero false "still" is impossible, record that: walking after a height transition stays refused.
- [ ] **Step 8 (added 2026-11-11, user-approved):** Run the new still-standing check exactly as pre-registered in PRD §9 G1b Step 8: feature `D`/`G`, development on A+B, 0.8 margin, freeze hashes, run C, a single validation.
  - Script: `gear_sonic/scripts/gates/g1b_still2.py` with the subcommands `develop A B`, `freeze` and `validate C`.
  - Probe: `--hold-s 8 --iso-hold-s 8` on `g1b_walk_handoff_probe.py`.
  - Tests:
    - `test_disp_feature_uses_received_window`;
    - `test_dev_scope_excludes_first_0_5s`;
    - `test_validate_refuses_changed_hash`;
    - `test_verdict_failed_on_any_false_accept`.
  - Record the outcome in §11 G1b. On PASS, set `still_check_adopted = True` and the thresholds in Task 6 params.
- [ ] **Step 9 (added 2026-11-12, user review):** Run the decision-time stress test exactly as pre-registered in PRD §9 G1b Step 9. Thresholds are unchanged from Step 8.
  - Sim: tracked option `--disturbance-port` (default off), applying a pelvis wrench via PUSH/PULL and publishing `pelvis_wrench` in `sim_gt`.
  - Probe: new `g1b_still3_probe.py` (`--mode calibrate|validate`).
  - Analysis: new `g1b_still3.py` with the subcommands `calibrate D0`, `freeze` (manifest of every repo-local module imported, plus the probe, the sim files and both JSONs) and `validate D`.
  - Existing gate files must stay byte-unchanged, to keep the Step 8 freeze provenance.
  - Tests:
    - `test_effective_uses_stationarity_not_z_band`;
    - `test_calibration_picks_smallest_2of2`;
    - `test_manifest_covers_imported_helpers`;
    - `test_validate_refuses_changed_helper`;
    - `test_verdict_order`;
    - `test_disturbance_not_a_command_change`;
    - `test_sim_wrench_applied_and_published`.
  - Record the outcome in §11 G1b. FAILED sets `still_check_adopted = False`.
- [ ] **Step 6:** Checkpoint (suggested message `feat(gates): G1b restricted walking-height probe`).

### Task 4: G3 — capture backend probe

**Files:**
- Create: `gear_sonic/scripts/gates/g3_backend_probe.py` (CLI: `--input-source {xrt,isaac-teleop} --out g3_<backend>.jsonl`)

**Interfaces:**
- Produces a JSONL, one line per sample at ≥ 50 Hz, with: `t_mono`, the reader-wide timestamp, and per controller the raw pose, stick, trigger and grip. It also dumps **every** attribute or field the backend exposes that could be per-controller validity or a timestamp:
  - XRT: list `dir(xrt)` once at start and call every zero-arg `get_*` / `is_*` that mentions controller, valid, track or time;
  - Isaac: every field of the per-controller snapshot from `IsaacTeleopClient._get_tracker_data()`, including the `is_valid` used for body joints (`gear_sonic/tests/test_input_readers.py`).

- [ ] **Step 1:** Implement the script. It's a probe with no logic worth unit-testing; YAGNI on tests.
- [ ] **Step 2:** Per backend, run the protocol and mark each phase with an operator keypress written into the log:
  1. hold both controllers still for 30 s;
  2. cover the left controller (hand behind back) for 5 s;
  3. turn the left controller off for 5 s, then on;
  4. repeat steps 2–3 for the right controller;
  5. swing fast (≈ 2 m/s) for 5 s.
- [ ] **Step 3:** From the logs, decide per backend which signal (`validity` / `timestamp` / `none`) distinguishes "still" (phase 1) from "lost" (phases 2–3) with zero false positives in phase 1. Fill PRD §11 G3 with the signal per backend and `stick_control_enabled = (signal != "none")`. Write down the fast-swing peak speed, which sanity-checks the 6 m/s outlier limit.
- [ ] **Step 4:** Checkpoint (suggested message `feat(gates): G3 backend probe`).

### Task 5: G2 — isolated arm-only IK prototype + benchmark

**Files:**
- Create: `gear_sonic/utils/teleop/manipulation/__init__.py` (empty), `gear_sonic/utils/teleop/manipulation/arm_ik.py`
- Create: `gear_sonic/scripts/gates/g2_ik_bench.py`
- Test: `gear_sonic/tests/manipulation/test_arm_ik.py`

**Interfaces:**
- Produces:
  ```python
  class ArmIK:
      def __init__(self, robot_model=None)  # default instantiate_g1_robot_model(); builds a reduced pinocchio model:
                                            # root = torso_link, joints = 7 left + 7 right arm joints only
      def fk_wrist_in_torso(self, arm_q14: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]  # (p[3], R[3x3]) of {side}_wrist_yaw_link
      def solve(self, targets: dict[str, tuple[np.ndarray, np.ndarray]], q_init14: np.ndarray,
                q_posture14: np.ndarray) -> tuple[np.ndarray, bool]  # (arm_q14 within limits, solver_ok)
  ```
  Arm joint order = the arm part of `_get_upper_body_joint_indices` (entries 3..16 of the 17-joint list). The waist is excluded entirely because the targets are torso-frame. Adapt `decoupled_wbc/control/teleop/solver/body/body_ik_solver.py` (pink: FrameTask per wrist, `WeightedPostureTask` toward `q_posture14`, joint limits) onto the reduced model.

- [ ] **Step 1:** Write the tests:
  - `test_only_arm_joints_in_model`: `model.nq == 14`;
  - `test_fk_ik_roundtrip`: 100 random in-limit q → FK → `solve` from a perturbed init; position error < 1 cm, orientation error < 5°;
  - `test_unreachable_target_returns_limited_solution`: target 2 m away → returns in-limit q, `solver_ok is True`, no exception;
  - `test_solution_within_joint_limits`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement `ArmIK`.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Implement `g2_ik_bench.py` (method per §6.2):
  - targets: ≥ 1,000 wrist trajectories, taken from SMPL wrist joints 22/23 of existing recorded episodes (via `run_data_exporter` outputs, `teleop.smpl_joints`) and converted to torso frame with the anchor-delta mapping of §6.2, plus random reachable targets from FK of random in-limit q;
  - 100 warm-up ticks, then 10,000 timed ticks, each covering both arms + `fk_wrist_in_torso` + target transforms;
  - print CPU model and p50 / p99 / max.

  Run it on the teleop PC.
- [ ] **Step 6:** Fill PRD §11 G2. Pass = p99 < 10 ms. If it fails, stop and report; don't proceed.
- [ ] **Step 7:** Checkpoint (suggested message `feat(gates): G2 arm-only IK prototype and benchmark`).

### Task 6: Gate checkpoint + parameters (HARD STOP)

**Files:**
- Create: `gear_sonic/utils/teleop/manipulation/params.py`
- Test: `gear_sonic/tests/manipulation/test_params.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class ManipulationParams:
      deadzone: float = 0.15
      waist_yaw_rate: float = 0.5; waist_pitch_rate: float = 0.3
      waist_yaw_limits: tuple[float, float] = (-0.6, 0.6); waist_pitch_limits: tuple[float, float] = (-0.1, 0.4)
      height_rate: float = 0.1; height_limits: tuple[float, float] = (0.2, 0.8)
      height_quantum: float = 0.01; height_min_interval_s: float = 0.2
      walk_start: float = 0.20; walk_stop: float = 0.12
      chord_hold_s: float = 0.15
      controller_stale_s: float = 0.1; outlier_speed: float = 6.0; reacquire_samples: int = 3
      recover_valid_s: float = 0.5; ik_fail_loss_s: float = 0.5
      ik_err_pos_m: float = 0.03; ik_err_rot_deg: float = 10.0   # PRD §6.6 tracking-error check
      ik_reseed_gain: float = 0.5; ik_reseed_rate: float = 1.0     # stall if re-seeded err ≤ gain·err; rad/s per joint
      takeover_tol: float = 0.05; takeover_blend_s: float = 0.2
      lock_ack_timeout_s: float = 0.5; feedback_max_age_s: float = 0.1
      still_check_adopted: bool = False  # PRD §11 G1b: Step 8 PASS (run C) withdrawn by Step 9 FAILED (2026-11-12, run D); −1-only scope everywhere
      still_window_s: float = 1.0; still_disp_max: float = 0.0135184; still_gyro_mean_max: float = 0.0037496   # D_max [rad], G_max [rad/s]
      walk_settle_s: float = 4.5; walk_height_handoff_s: float = 0.5   # PRD §6.3; walk_settle_s from PRD §11 G1b
      # gate-derived — copy from PRD §11
      h_default: float = <G1>; h_walk_min: float | None = <G1>; idle_honors_height: bool = <G1>
      cmd_to_z: tuple[tuple[float, float], ...] = <G1>   # used by Task 16 height metrics
      loss_signal: dict[str, str] = <G3>   # {"xrt": "validity"|"timestamp"|"none", "isaac-teleop": ...}
      def stick_control_enabled(self, backend: str) -> bool  # loss_signal[backend] != "none"
  ```

- [ ] **Step 1: Checkpoint.** Open PRD §11. The G0, G1b, G3 and G2 rows must be filled and passed:
  - **G0:** `robot_side` is not `"inconclusive"`. `reference_side` is recorded but doesn't block.
  - **G1:** recorded (FAIL as measured; its values are used). **G1b:** `gate_pass` is True **and** Task 3b Step 7 is recorded (FAILED, not adopted). Step 8 PASSED in sim (run C), but **Step 9 FAILED (run D, 2026-11-12), so `still_check_adopted = False` everywhere and the −1-only scope applies**. It decides only `still_check_adopted` and doesn't block Phase B. On hardware, the −1-only scope applies until Step 9 (PRD §9) PASSES in sim **and** the PRD §6.3 gyro-bias precondition holds. A Step 9 FAILED sets `still_check_adopted = False`.
  - **G3:** a signal recorded for each backend.
  - **G2:** p99 < 10 ms.

  If any gate isn't passed, **stop and report to the user.** Do not start Phase B.
- [ ] **Step 2:** Write `test_gate_values_present`: `h_default` lies in [0.2, 0.8]; `loss_signal` has keys `xrt` and `isaac-teleop`; `stick_control_enabled` matches `loss_signal`.
- [ ] **Step 3:** Run. Expected: FAIL.
- [ ] **Step 4:** Implement `params.py` with the §11 values.
- [ ] **Step 5:** Run. Expected: PASS. Checkpoint (suggested message `feat(manipulation): parameters from gate results`).

---

## Phase B — Feature (only after Task 6)

### Task 7: Chord arbiter

**Files:**
- Create: `gear_sonic/utils/teleop/manipulation/chords.py`
- Test: `gear_sonic/tests/manipulation/test_chords.py`

**Interfaces:**
- Produces:
  ```python
  class Chord(Enum): ABXY = 1; BY = 2; AB = 3; XY = 4
  class ChordArbiter:
      def __init__(self, hold_s: float)
      def update(self, a: bool, b: bool, x: bool, y: bool, now: float, policy_running: bool) -> Chord | None
  ```
  Rules (§5.3):
  - Not armed until all four buttons have been seen released after construction.
  - If `policy_running` and all four are pressed: return `ABXY` on the first such tick, regardless of arming or hold, then disarm. It fires once per press.
  - Otherwise a chord, including `ABXY` when not running (start), fires once its exact set has been stable for `hold_s` while armed. Then disarm.
  - Re-arm only when all four are released.
  - A single button is never a chord.


- [ ] **Step 1:** Write the tests:
  - `test_startup_held_buttons_do_not_start`: all held from t=0 for 1 s with running=False → never `ABXY`; release, then press and hold 0.15 s → `ABXY`;
  - `test_stop_fires_immediately_when_running`: the first tick with all four pressed → `ABXY`;
  - `test_stop_fires_without_rearm_after_toggle`: `BY` fires; then, while B+Y are still held, A+X are added → `ABXY` on that tick;
  - `test_staggered_abxy_never_fires_toggle`: for every `itertools.permutations("abxy")`, press one button every 0.03 s, running=True → the only fired chord is `ABXY`;
  - `test_toggle_fires_after_hold`: B+Y at 0.149 s → `None`; at 0.15 s → `BY`;
  - `test_rearm_required`: after `BY`, release Y and press it again (B held) → no fire until all four have been released;
  - `test_stop_fires_once_per_press`;
  - `test_single_button_with_grip_is_not_a_chord`: B alone held 1 s → `None`;
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Checkpoint (suggested message `feat(manipulation): chord arbiter`).

### Task 8: Stick commands — waist, height, walk gate

**Files:**
- Create: `gear_sonic/utils/teleop/manipulation/sticks.py`
- Test: `gear_sonic/tests/manipulation/test_sticks.py`

**Interfaces:**
- Consumes: `ManipulationParams`.
- Produces:
  ```python
  class RateAccumulator:
      def __init__(self, rate: float, limits: tuple[float, float], deadzone: float, value: float)
      value: float
      def update(self, stick: float, dt: float) -> float   # rate * sign * (|s|-dz)/(1-dz), clamped
  class HeightCommand:
      def __init__(self, p: ManipulationParams)
      value: float            # -1.0 until first deflection
      def update(self, stick: float, dt: float, now: float, floor: float | None = None) -> float  # returns height to SEND
  class WalkGate:
      def __init__(self, p: ManipulationParams, backend: str)
      def update(self, stick_mag: float, height_value: float) -> tuple[bool, str | None]  # (walk_requested, refusal)
      def walk_floor(self) -> float   # h_walk_min, or h_default - 0.01 when h_walk_min is None
  ```
  HeightCommand rules (§5.4):
  - On the first tick out of the deadzone, set `value = h_default` and send `h_default`.
  - Then integrate, clamped to `height_limits` and to `floor` when given.
  - The sent value changes only if |Δ| ≥ `height_quantum` **and** ≥ `height_min_interval_s` since the last change.
  - On release (back in the deadzone), send the exact `value` at the next allowed slot.

  WalkGate rules (§6.3):
  - Hysteresis: start at > `walk_start`, stop at < `walk_stop`.
  - Refuse (all reasons are warnings; the robot stays standing; a refused request needs a stick release and a new deflection):
    - `not p.stick_control_enabled(backend)` → `"stick control disabled for backend"`;
    - **while `not p.still_check_adopted` (−1-only scope, PRD §6.3; required on hardware until the gyro-bias precondition holds):**
      - the height has changed at any time this session → `"walking disabled after height change"`;
      - the static command changed < `walk_settle_s` ago (including a walk stop) → `"height settling"`;
    - **once adopted:**
      - `height_value != -1` and `height_value < walk_floor()` → `"raise height to walk"`;
      - static command changed < `walk_settle_s` ago → `"height settling"` (sending −1 during a transition restarts the timer);
      - `still` is False → `"not still: <signal>"`.
  - **Planner height while walking (PRD §6.3, band unknown only):**
    - `planner_height(phase, height_value) -> float`, where `phase` is `static` / `handoff_in` / `walking` / `handoff_out`.
    - It returns −1 in `handoff_in`, `walking` and `handoff_out`, and `height_value` in `static`.
    - Each hand-off lasts `walk_height_handoff_s` in the static mode. `height_value` (`HeightCommand.value`) is never modified.
    - With a known band (`h_walk_min` set), it always returns `height_value`.
  - `update` therefore also takes `now: float`, `static_cmd_changed_t: float` (the last change of the sent static mode or height field), `still: tuple[bool, str | None]` (from `StillnessCheck`, Task 13) and `watchdog_latched: bool` (from feedback, Task 13).
  - In every scope, including untouched −1, a new walk needs:
    - a completed controlled stop (stick released and hand-off out done);
    - a fresh deflection from neutral;
    - `watchdog_latched` False, with the latch clear counting as a stop for the settle timer;
    - a valid, fresh walk-stick controller (PRD §6.6; invalid or stale → stick centered).

- [ ] **Step 1:** Write the tests:
  - `test_deadzone_holds`;
  - `test_full_deflection_rate`: yaw, stick 1.0 for 1 s at dt 0.05 → 0.5 ± 1e-6;
  - `test_linear_past_deadzone`: stick 0.575 → half rate;
  - `test_clamped_accumulator_responds_immediately_on_reverse`: hold +1 for 5 s (clamped at 0.6), then −1 for 0.05 s → value < 0.6;
  - `test_height_sentinel_until_first_use`;
  - `test_height_first_use_starts_at_default`;
  - `test_height_quantum_and_interval`: full deflection at 50 Hz → sent value changes at most every 0.2 s and by ≥ 0.01;
  - `test_height_final_value_sent_on_release`: release at t; at the next allowed slot the sent value equals `value` exactly;
  - `test_height_floor_clamps_while_walking`;
  - `test_walk_hysteresis`: 0.19 → False; 0.21 → True; 0.15 → True; 0.11 → False;
  - `test_walk_refused_below_floor`;
  - `test_walk_allowed_at_sentinel`;
  - `test_walk_floor_without_band`: `h_walk_min=None` → floor = `h_default - 0.01`;
  - `test_walk_refused_when_stick_control_disabled`.
  - −1-only scope (`still_check_adopted=False`):
    - `test_walk_disabled_after_any_height_change`: −1 → 0.80 → walk → refused `"walking disabled after height change"`, for the rest of the session;
    - `test_sentinel_after_height_change_stays_disabled`: −1 → 0.80 → −1 sent again, held 10 s, stick released and freshly deflected → refused `"walking disabled after height change"` (sending −1 restores nothing);
    - `test_sentinel_walk_needs_settle_after_stop`: at −1, a walk stop then a re-request 3 s later → refused `"height settling"`; at 4.5 s → allowed;
    - `test_untouched_sentinel_walks`: −1 held 10 s → allowed with no stillness input.
  - Every scope (PRD §6.3 item 3; −1 bypasses no interlock):
    - `test_sentinel_walk_needs_stop_and_rearm`: untouched −1, walk, then the stick held deflected through the stop and for 10 s → no new walk until it is released and deflected again;
    - `test_watchdog_latch_counts_as_stop`: untouched −1 while walking, `watchdog_latched` true → static; requests refused `"watchdog latched"` while latched; after the clear, refused `"height settling"` until 4.5 s after the clear, and then only on a fresh deflection;
    - `test_sentinel_does_not_bypass_interlocks`: untouched −1 with lock release pending, stick control disabled, or the walk-stick controller invalid/stale (stick counts as centered, PRD §6.6) → no walk starts; a controller loss while walking ends the walk through the controlled stop.
  - Adopted scope (`still_check_adopted=True`):
    - `test_walk_refused_while_height_settling`;
    - `test_elapsed_time_alone_not_enough`: after `walk_settle_s`, `still` False → refused `"not still"`; a later True with the stick held does not start walking, but a re-deflection does;
    - `test_transition_to_sentinel_restarts_timer`: −1 sent 1 s after 0.80 → `"height settling"`.
  - `test_walking_sends_sentinel_height_cmd_unchanged`: from `height_value = h_default`, the phase sequence static → handoff_in (0.5 s) → walking → handoff_out (0.5 s) → static sends `h_default`, −1, −1, −1, `h_default`; `HeightCommand.value` is unchanged throughout.
  - `test_known_band_sends_height_cmd_while_walking`: with `h_walk_min = 0.6` and height 0.7, walking sends 0.7.
  - `test_lowered_walk_refused_no_auto_raise`: at 0.6 with the band unknown, the walk is refused and the sent height stays 0.6.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Checkpoint (suggested message `feat(manipulation): stick commands and walk gate`).

### Task 9: Controller mapping + loss detection

**Files:**
- Create: `gear_sonic/utils/teleop/manipulation/tracking.py`
- Modify: `gear_sonic/scripts/pico_manager_thread_server.py:170-199`. Move `_compute_rel_transform` into `tracking.py` as `compute_rel_transform`, and import it back under the old name so behavior is unchanged.
- Test: `gear_sonic/tests/manipulation/test_tracking.py`

**Interfaces:**
- Produces:
  ```python
  def unity_to_robot(pose7_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]   # (p[3], R[3x3]) via compute_rel_transform(identity)
  def headset_yaw(R_head: np.ndarray) -> float
  class ArmAnchor:
      def __init__(self, p_c0, R_c0, psi0: float, p_w0, R_w0, k: float = 1.0)
      def target(self, p_c, R_c) -> tuple[np.ndarray, np.ndarray]   # §6.2 formulas, verbatim
  @dataclass
  class ControllerSample:
      p: np.ndarray; R: np.ndarray; stick: tuple[float, float]; trigger: float; grip: float
      valid: bool | None; sample_ts: float | None; t: float      # t = reader monotonic receipt time
  class TrackState(Enum): TRACKING = 0; LOST = 1; RECOVERING = 2
  class ControllerLossDetector:
      def __init__(self, p: ManipulationParams, signal: str)    # "validity" | "timestamp" | "none"
      def update(self, s: ControllerSample) -> TrackState
      def ready_to_reanchor(self) -> bool    # True once, after RECOVERING has been valid for recover_valid_s
      def force_lost(self) -> None           # used by IK-failure rule
  ```
  Loss rules (§6.6):
  - `validity`: `valid is False` → LOST.
  - `timestamp`: `sample_ts` unchanged for > `controller_stale_s` → LOST.
  - Outlier, any signal: |Δp| / (Δ`t`) > `outlier_speed` → the sample is discarded (the caller holds) until `reacquire_samples` consecutive in-limit samples arrive.
  - From LOST, valid samples → RECOVERING; after `recover_valid_s` continuous, `ready_to_reanchor()` → True and the state → TRACKING.
  - A pose that is merely identical never causes LOST.

- [ ] **Step 1:** Write the tests:
  - `test_identity_delta_returns_anchor`;
  - `test_forward_maps_to_torso_forward_with_rotated_headset`: psi0 = +90°; controller moves +0.1 along the W y axis (operator forward) → `p_T == p_w0 + [0.1, 0, 0]`;
  - `test_rotation_delta_about_operator_z`: 30° about W z with psi0 = 0 → `R_T == Rz(30°) @ R_w0`;
  - `test_anchor_uses_current_headset_yaw_on_reentry`: two anchors with psi0 0° and 90° map the same W displacement to different torso axes;
  - `test_stationary_30s_never_lost` (each signal kind; identical poses, advancing `t`, and advancing `sample_ts` for the timestamp signal);
  - `test_timestamp_stale_is_lost`;
  - `test_validity_false_is_lost`;
  - `test_fast_swing_not_outlier`: 2 m/s at 50 Hz;
  - `test_spike_discarded_then_reacquired`: a 0.5 m spike on one sample → discarded; 3 good samples → accepted;
  - `test_recovery_requires_0_5s_valid`;
  - `test_rel_transform_alias_unchanged`: old and new functions give the same output on a fixed pose.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS. Also run `pytest gear_sonic/tests -v` to confirm nothing else broke.
- [ ] **Step 5:** Checkpoint (suggested message `feat(manipulation): controller mapping and loss detection`).

### Task 10: Hand takeover

**Files:**
- Create: `gear_sonic/utils/teleop/manipulation/hands.py`
- Test: `gear_sonic/tests/manipulation/test_hands.py`

**Interfaces:**
- Produces:
  ```python
  class HandTakeover:   # one per hand
      def __init__(self, p: ManipulationParams, f: Callable[[float, float], np.ndarray], side: str)
      def hold(self, trigger: float, grip: float, last_sent: np.ndarray) -> None   # on MANIPULATION exit
      def update(self, trigger: float, grip: float, now: float) -> tuple[np.ndarray, str | None]  # (command, prompt)
      live: bool        # True when no hold is pending (initial state True)
  ```
  Rules (§5.1):
  - While held, return `last_sent` (stored copy, **not** `f(held)`).
  - Takeover when |trigger − trigger_held| ≤ `takeover_tol` **and** |grip − grip_held| ≤ `takeover_tol` on the same tick.
  - After takeover, the command blends linearly from `last_sent` to `f(live)` over `takeover_blend_s`, then it is `f(live)`.
  - Prompt format: `"{L|R} trigger: press more" | "{L|R} trigger: release a bit" | "{L|R} trigger: matched"`, and the same for grip, joined with `", "`. Prompt is `None` once live.

- [ ] **Step 1:** Write the tests:
  - `test_held_returns_last_sent_not_recomputed`: `last_sent ≠ f(held)` → returned exactly;
  - `test_takeover_both_within_tol` (0.04 → live);
  - `test_no_takeover_one_input_outside` (0.06);
  - `test_mixed_trigger_matched_grip_off`;
  - `test_approach_from_above_and_below`;
  - `test_blend_continuity`: at takeover + 0.1 s the command is the midpoint; at +0.2 s it equals `f(live)`;
  - `test_prompt_directions`: live above held → "release a bit"; below → "press more";
  - `test_sides_independent`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Checkpoint (suggested message `feat(manipulation): hand soft takeover`).

### Task 11: Waist lock — deploy blender + protocol fields + lock-release gate

**Files:**
- Create: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/waist_lock.hpp`
- Test: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/unit_tests/test_waist_lock.cpp`
- Modify: `gear_sonic/utils/teleop/zmq/zmq_planner_sender.py:64` (`build_planner_message` gets optional `waist_lock: bool | None = None`, `lock_req_seq: int | None = None`, `ctrl_state: Sequence[int] | None = None`, each packed as `i32` shape [1] / [1] / [2] and omitted when `None`)
- Modify: `gear_sonic_deploy/.../include/input_interface/input_command.hpp:71` (`PlannerMessage`: `std::optional<int> waist_lock, lock_req_seq`; `std::optional<std::array<int,2>> ctrl_state`)
- Modify: `gear_sonic_deploy/.../include/input_interface/zmq_manager.hpp:775-945` (decode the new fields like `upper_body_position`; expose `GetWaistLockState() -> {bool lock, uint32_t req_seq, steady_clock::time_point stamp, int mode}`)
- Modify: `gear_sonic_deploy/.../src/g1_deploy_onnx_ref.cpp` (`--waist-lock` CLI flag; in `CreatePolicyCommand` around `:3155-3160`, after the loop, apply the blender to `q_target` at MuJoCo 12/13/14 using `upper_body_joint_positions_buffer_[0..2]`; `last_action` untouched)
- Modify: `gear_sonic_deploy/.../include/output_interface/output_interface.hpp`. This is the **serialization path**: `ZMQOutputHandler::publish` → `create_output_data_map` → `pack_combined_state`, which loops over `output_data_map_` (`zmq_output_handler.hpp:395-399`). Add to the `OutputInterface` base:
  - `struct ManipFeedback { double waist_lock_alpha = 0; uint32_t lock_ack_seq = 0; bool waist_lock_enabled = false; bool watchdog_latched = false; };`
  - `void SetManipFeedback(const ManipFeedback&)`, a mutex-guarded snapshot;
  - `create_output_data_map` writes the four keys into `output_data_map_` (as doubles, matching the map type) from **one** snapshot read.
- Modify: `g1_deploy_onnx_ref.cpp` `Control()`. After the blender's `Apply` and before the output `publish` loop (`:4002-4010`), call `SetManipFeedback({blender.alpha(), blender.ack_seq(), flag, false})` on every output interface. Task 12 replaces `false` with `watchdog.latched()`. Alpha and ack come from the same `Apply` call, and every published sample carries them.
- Modify: `gear_sonic_deploy/deploy.sh` (pass `--waist-lock` through, following the `--motor-kp-scale` pattern at `:322`)
- Create: `gear_sonic/utils/teleop/manipulation/lock_gate.py`
- Create: `gear_sonic/scripts/gates/tcp_fault_proxy.py`: `class TcpFaultProxy(listen_port: int, target: tuple[str, int])` with `.start()`, `.set_delay(s: float)`, `.blackhole(on: bool)` (stop forwarding both ways while keeping the TCP connections open, like a dead Wi-Fi link) and `.stop()`. It forwards raw TCP bytes, so ZMQ sees a real network path.
- Create: `gear_sonic/tests/manipulation/e2e_stack.py`. It provides the pytest fixture `sim_stack(deploy_flags: list[str], proxy: bool)`:
  - it launches `run_sim_loop.py --no-elastic-band --publish-ground-truth` and the built deploy binary in sim with `zmq_manager` input, using the same arguments `deploy.sh` composes (read them from `deploy.sh`; don't call it interactively);
  - optionally, it puts `TcpFaultProxy` instances on the planner path (5556) and the feedback path (5557);
  - it waits for `g1_debug`, sends the start command, and yields handles `{send(msg), feedback_reader, sim_gt_sub, proxies}`.

  These tests run only when `MANIP_E2E=1`. When it is set, a missing binary or a sim failure is a test **failure**, not a skip.
- Test: `gear_sonic/tests/manipulation/test_lock_gate.py`, `gear_sonic/tests/manipulation/test_planner_message_fields.py`, `gear_sonic/tests/manipulation/test_lock_protocol_e2e.py`

**Interfaces:**
- Produces (C++):
  ```cpp
  struct WaistLockInput { bool flag_enabled; bool msg_lock; double msg_age_s; int planner_mode;
                          std::array<double,3> waist_cmd; uint32_t req_seq; };
  class WaistLockBlender {
   public:
    WaistLockBlender(double ramp_in_s = 0.5, double ramp_out_s = 0.2, double fresh_s = 0.1, double max_rate = 1.0);
    std::array<double,3> Apply(const WaistLockInput& in, const std::array<double,3>& q_policy, double dt);
    double alpha() const; uint32_t ack_seq() const; uint64_t freshness_misses() const;
  };
  ```
  - `Apply` is called on **every** control tick whenever the deploy has received at least one planner message carrying `lock_req_seq`, **independent of `--waist-lock` and of `has_upper_body_data_`**. Acknowledgment is protocol and is always processed. Motor override is a separate decision.
  - Override active iff `flag_enabled && has_upper_body && msg_lock && msg_age_s < fresh_s && planner_mode ∈ {IDLE(0), IDLE_SQUAT(4)}`. `WaistLockInput` gains `bool has_upper_body`. With the flag off, α is identically 0.
  - α moves toward 1 at 1/`ramp_in_s` while active and toward 0 at 1/`ramp_out_s` otherwise.
  - The applied waist target follows `waist_cmd` at ≤ `max_rate`.
  - Output = (1−α)·q_policy + α·applied.
  - `ack_seq` = the `req_seq` of the latest Apply call.
  - Freshness misses are counted on each active→stale edge.
- Produces (Python):
  ```python
  class LockReleaseGate:
      def __init__(self, timeout_s: float)
      def request(self, now: float) -> int                      # new seq (monotonic, starts at 1)
      def on_feedback(self, ack_seq: int, alpha: float, now: float) -> None
      def status(self, now: float) -> str   # "idle" | "pending" | "released" | "timed_out"
      def reset(self) -> None
  ```
  `released` only if a sample with `ack_seq == current seq` and `alpha == 0.0` arrives before `timeout_s`. Once `timed_out`, it stays so until `reset()`.
- **Sequence protocol:**
  - The streamer increments `lock_req_seq` on **every** change of the requested lock state (lock → release or release → lock) and sends the current seq on every message.
  - Cancelling a pending release (the stick returns to neutral before the release is confirmed) is a new request: re-lock with a new seq, and `gate.reset()`.
  - A deploy without `--waist-lock` still echoes `lock_ack_seq` with `waist_lock_alpha == 0` and `waist_lock_enabled == false`, so a release is confirmed at once and walking isn't blocked by a lock that doesn't exist.

- [ ] **Step 1:** Write gtests `WaistLock.*`:
  - `OffWhenFlagDisabled`: α stays 0;
  - `RampIn`: α = 0.5 ± 0.01 after 0.25 s;
  - `RampOut`: α 1 → 0 in 0.2 s;
  - `ShortGapNoEffect`: 50 ms age → α unchanged;
  - `LongGapRampsOutThenIn`: 150 ms gap → ramp-out starts once age ≥ 0.1, and resuming ramps in;
  - `WalkModeInactive`: mode 2;
  - `KneelInactive`: mode 6;
  - `RateLimitedTarget`: cmd step of 1 rad → applied moves ≤ 1 rad/s;
  - `AckEchoesSeq`;
  - `AckWhenFlagDisabled`: `flag_enabled = false`, req_seq 7 → `ack_seq() == 7` and `alpha() == 0` on the first Apply;
  - `AckWithoutUpperBody`: flag on, `has_upper_body = false` → ack echoes, and α ramps to 0;
  - `OutputUnchangedWhenFlagDisabled`: the returned waist equals `q_policy` exactly;
  - `BlendFormula`.

  Write pytest `test_lock_gate.py`:
  - `test_stale_ack_ignored`: ack = seq−1 with α 0 → pending;
  - `test_matching_ack_nonzero_alpha_pending`;
  - `test_released_on_match`;
  - `test_reordered_and_duplicate_samples`;
  - `test_delay_300ms_releases` (0.3 < 0.5);
  - `test_delay_700ms_times_out`;
  - `test_timed_out_sticky_until_reset`.

  Write pytest `test_planner_message_fields.py`:
  - new fields absent when `None`; the payload is byte-identical to today's for an old-style call;
  - fields present with the correct `i32` encoding when set.

  Write `test_lock_protocol_e2e.py` (real sim + deploy, `MANIP_E2E=1`). The planner messages are IDLE with a 17-joint upper body, and the gate is fed from live `g1_debug` samples:
  - `test_ack_and_alpha_arrive_in_one_sample` (deploy `--waist-lock`):
    1. Lock (seq 1) for 1.0 s → samples reach α = 1 with ack 1.
    2. Release (seq 2).
    3. Every sample decodes all four keys. The first sample with ack 2 has α < 1. α is non-increasing after it. No sample after it carries ack < 2. The gate reaches `released` within 0.5 s, on a sample whose α is exactly 0.
  - `test_lock_disabled_on_deploy_releases_immediately` (no `--waist-lock`): lock (seq 1), then release (seq 2) → every sample has `waist_lock_enabled == False` and α == 0, and the gate is `released` on the first sample with ack 2.
  - `test_feedback_delay_300ms_releases` and `test_feedback_delay_700ms_times_out`: `TcpFaultProxy.set_delay` on the feedback path.
  - `test_feedback_blackholed_times_out_and_stays`: the gate is `timed_out` and stays so after the proxy resumes, until `reset()`.
  - `test_cancel_then_retry`: release (seq 2); before its ack, re-lock (seq 3) and reset the gate; then release (seq 4). Samples with ack 2 or 3 never make the gate `released`. It is released only on ack 4 with α 0. α rises again after seq 3.
  - `test_default_session_keys_additive`: without any new flag, the old keys are unchanged and only the additive keys are new. It also saves one captured sample to `gear_sonic/tests/manipulation/data/g1_debug_additive.msgpack` for the compat test below.

  Write `test_feedback_consumers_compat.py` (unit, no sim). It feeds the captured sample, and the same sample with the additive keys stripped, through **every existing `g1_debug` consumer**, and asserts identical outputs:
  - `zmq_state_subscriber._unpack_msgpack_zmq` + `_convert_lists_to_numpy`, which `run_data_exporter` and `run_vla_inference` use: the same values for every pre-existing key;
  - `run_data_exporter._add_cpp_state_features` in a **default** session: an identical feature dict, with **no new columns**, so the dataset schema is unchanged;
  - `pico_manager_thread_server.FeedbackReader` parsing: the same `body_q`, hands and `full_body_q`;
  - `gear_sonic_deploy/visualize_motion.py`: its decode block (`:120-140`) is moved unchanged into `_parse_state(result) -> dict` so it can be called. The output is identical.

  The ROS 2 output handler doesn't read `output_data_map_` (checked: `ros2_output_handler.hpp`), so it is unaffected.
- [ ] **Step 2:** Run all suites. Expected: FAIL.
- [ ] **Step 3:** Implement the blender, the gate, the message fields, decode, the CLI flag, the `CreatePolicyCommand` hook (`Apply` runs per the rule above; `q_target[12..14]` is overwritten only when the flag is set **and** `has_upper_body_data_`; otherwise `q_target` is untouched), `ManipFeedback` plumbing, the proxy, the e2e fixture, and the `deploy.sh` passthrough.
- [ ] **Step 4:** Run the unit suites, then `MANIP_E2E=1 pytest gear_sonic/tests/manipulation/test_lock_protocol_e2e.py -v`. Expected: all PASS. Also run the existing `run_tests` binary in full.
- [ ] **Step 5:** Checkpoint (suggested message `feat(deploy): waist lock blender with end-to-end sequenced release ack`).

### Task 12: Deploy input watchdog with latch and catch-up limiter

**Files:**
- Create: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_watchdog.hpp`
- Test: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/unit_tests/test_input_watchdog.cpp`
- Modify: `zmq_manager.hpp:576-610` (when `--input-watchdog-ms` is given, use `InputWatchdog`: on `kForceStop`, substitute `MovementState(IDLE, {0,0,0}, latched_facing, -1, current_height)`; engage `CatchUpLimiter` on timeout, and pass the upper-body buffer through `Step` while engaged). Without the flag, the existing 1 s `PLANNER_TIMEOUT` path stays untouched.
- Modify: `g1_deploy_onnx_ref.cpp` (`--input-watchdog-ms <int>` flag; feed `watchdog.latched()` into `ManipFeedback` from Task 11), `deploy.sh` (passthrough)
- Modify: `zmq_planner_sender.py` `build_planner_message` (optional `op_neutral: bool | None = None`, packed `i32` [1], omitted when `None`), `input_command.hpp` `PlannerMessage` (`std::optional<int> op_neutral`), and the `zmq_manager.hpp` decode.
- Test: `gear_sonic/tests/manipulation/test_watchdog_e2e.py` (uses the Task 11 `sim_stack`, `MANIP_E2E=1`)

**Interfaces:**
- Produces:
  ```cpp
  class InputWatchdog {
   public:
    enum class Decision { kFollow, kForceStop };
    InputWatchdog(double timeout_s, double neutral_rearm_s = 0.15);
    Decision Update(double since_last_msg_s, bool msg_is_neutral, double now_s);
    bool latched() const;
  };
  class CatchUpLimiter {
   public:
    CatchUpLimiter(double max_rate = 1.0, double tol = 0.01);
    void Engage(const std::array<double,17>& applied_now);
    std::array<double,17> Step(const std::array<double,17>& target, double dt);
    bool engaged() const;   // false once every joint is within tol of target
  };
  ```
  - **`msg_is_neutral` := `op_neutral == 1` && mode ∈ {IDLE, IDLE_SQUAT} && movement == 0.** The message content alone is not enough, because the streamer sends static, zero-movement messages during the lock-release handshake and while `WalkGate` refuses, even though the operator is still holding the walk stick.
  - `op_neutral` is computed by the streamer from the **physical** sticks, before any gate, lock or loss logic. It is 1 iff the left-stick magnitude is < `walk_stop` **and** |right X| is < `deadzone`.
  - A controller that is LOST or frozen reports `op_neutral = 0`: an unknown stick is never neutral.
  - A message without `op_neutral` counts as **not neutral**. With `--input-watchdog-ms` set, a streamer that doesn't send it can never clear a latch; the deploy prints a warning once.
  - The latch clears only after `neutral_rearm_s` of **consecutive** neutral messages.

- [ ] **Step 1:** Write gtests:
  - `InputWatchdog.ForceStopAfterTimeout` (200 ms);
  - `InputWatchdog.ResumedWalkStillStopped`: messages resume with walk → kForceStop;
  - `InputWatchdog.ShortNeutralNotEnough` (100 ms);
  - `InputWatchdog.NeutralRearmThenFollow` (150 ms neutral, then walk → kFollow);
  - `InputWatchdog.DeflectedThroughoutNeverFollows`;
  - `CatchUpLimiter.LargeGapStaysEngagedPastOneSecond`: 2 rad gap, after 1 s still ≥ 0.99 rad away and engaged;
  - `CatchUpLimiter.FollowsMovingTarget`: per-step change ≤ 1 rad/s · dt;
  - `CatchUpLimiter.ReleasesWithoutStep`: on the tick it disengages, |output − target| ≤ 0.01;
  - `InputWatchdog.StaticMessageWithoutOpNeutralNotNeutral`: IDLE, movement 0, `op_neutral` absent → still latched after 1 s;
  - `InputWatchdog.StaticMessageOpNeutralZeroNotNeutral`: the lock-handshake shape (IDLE, movement 0, `op_neutral = 0`) → still latched.

  Write `test_watchdog_e2e.py` (real sim + deploy with `--waist-lock --input-watchdog-ms 200`; planner path through `TcpFaultProxy`). It drives a scripted message sequence that mirrors what `ManipulationStreamer` sends: the FROZEN walk request, lock release, then walking.
  - `test_blackhole_stops_walk_within_200ms`: walk at 0.3 m/s, then blackhole → `watchdog_latched` within 200 ms of the last message delivered, and the pelvis xy speed from `sim_gt` falls to < 0.05 m/s. Report the time to standstill.
  - `test_reconnect_with_stick_deflected_stays_stopped`:
    1. Walk, then blackhole for 1 s, then restore.
    2. The sender resumes as the streamer would with the stick still deflected: a fresh lock-release handshake (static messages, `op_neutral = 0`), then movement requests with `op_neutral = 0`.
    3. Feedback is delayed 300 ms throughout.
    4. Assert: no pelvis translation > 2 cm over 3 s, and `watchdog_latched` stays true.
    5. Then send `op_neutral = 1` for 100 ms → still latched. Then for ≥ 150 ms → it clears, and the next deflected walk request moves the robot.

  Killing the real streamer process is covered in Task 16 (9e), because it needs the real streamer.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement and wire.
- [ ] **Step 4:** Run the full `run_tests`, then `MANIP_E2E=1 pytest gear_sonic/tests/manipulation/test_watchdog_e2e.py -v`. Expected: PASS.
- [ ] **Step 5:** Checkpoint (suggested message `feat(deploy): latched input watchdog with operator-neutral re-arm`).

### Task 13: Reader + feedback changes and `ManipulationStreamer`

**Files:**
- Modify: `gear_sonic/scripts/pico_manager_thread_server.py:838-877` (`PicoReader._run`: add `sample["controllers"] = {"left": ControllerSample, "right": ControllerSample}` and `sample["headset_R"]` from `xrt.get_left/right_controller_pose()`, `get_headset_pose()`, the axes, triggers and grips. `valid` / `sample_ts` are filled per the G3 signal, otherwise `None`.)
- Modify: `gear_sonic/utils/teleop/input_readers.py:423-450` (`IsaacTeleopReader`: same keys, from `get_pose_by_name` and the controller snapshot)
- Modify: `pico_manager_thread_server.py:1649-1712` (`FeedbackReader`: store `received_monotonic`; add `age_s() -> float | None` (`None` if never received); also parse `waist_lock_alpha`, `lock_ack_seq`, `watchdog_latched` when present)
- Create: `gear_sonic/utils/teleop/manipulation/streamer.py`
- Test: `gear_sonic/tests/manipulation/test_streamer.py` (fakes: `FakeReader`, `FakeSocket` capturing decoded planner messages, `FakeFeedback`, `FakeIK` wrapping `ArmIK`)

**Interfaces:**
- Consumes:
  - `ManipulationParams`, `ChordArbiter` / `Chord` (Task 7);
  - `RateAccumulator`, `HeightCommand`, `WalkGate` (Task 8);
  - `ArmAnchor`, `ControllerLossDetector`, `ControllerSample`, `TrackState`, `headset_yaw` (Task 9);
  - `HandTakeover` (Task 10), `LockReleaseGate` (Task 11), `ArmIK` (Task 5);
  - `build_planner_message` with the new fields (Task 11);
  - `compute_hand_joints_from_inputs`, `YawAccumulator` (existing, `pico_manager_thread_server.py:754`, `:520`).
- Produces:
  ```python
  class ManipulationStreamer:
      def __init__(self, socket, reader, feedback, arm_ik: ArmIK, hand_fn, params: ManipulationParams,
                   backend: str, waist_lock: bool, poll_hz: int = 20)
      def try_enter_manipulation(self, now: float) -> str | None   # None = entered; else refusal reason, nothing sent
      def enter_frozen(self, now: float) -> None
      def run_manipulation_once(self, now: float) -> None
      def run_frozen_once(self, now: float) -> None
      waist_cmd: np.ndarray | None        # (yaw, roll=0, pitch); created once from measured waist
      last_sent_arms: np.ndarray | None   # 14
      last_sent_hands: dict[str, np.ndarray]
  def next_mode_manipulation(current: StreamMode, chord: Chord | None) -> StreamMode
  ```
  - **Message content.** MANIPULATION sends: mode IDLE, or IDLE_SQUAT when `height != -1 and not idle_honors_height`; movement 0; facing = the heading; height = `HeightCommand`; `upper_body_position = [*waist_cmd, *arm_q14]` in `_get_upper_body_joint_indices` order; hands; `waist_lock` (only if the flag is set); `lock_req_seq`; `ctrl_state`; `op_neutral` (Task 12 definition, in both MANIPULATION and FROZEN). No `vr_3pt_*`.
  - **Debug topic `manip_debug`**, on the same PUB socket, once per planner message. The deploy doesn't subscribe to it. Fields:
    - `t_mono_ns`;
    - a copy of every planner field sent: mode, movement, facing, height, `upper_body_position`, hands, `waist_lock`, `lock_req_seq`, `op_neutral`;
    - the wrist targets `target_p` [2×3] and `target_R` [2×9], NaN while held;
    - `track_state` [2];
    - `takeover_live` [2] and the takeover prompts;
    - `lock_gate_status`;
    - `entry_refusal` (string or empty);
    - `chord_fired` and `record_event`.

    This is the evidence source for Task 16. Under `--manipulation`, recorded episodes take the planner fields from it too.
  - **FROZEN** sends the arm snapshot = `last_sent_arms`, hands = `last_sent_hands`, plus `waist_cmd`. Walking, turning (`YawAccumulator` on right X) and height (right Y) go through `WalkGate`. When the waist lock is on, locomotion is sent only when `LockReleaseGate.status == "released"`.
  - **Entry guard (§6.5):** feedback `age_s() < feedback_max_age_s`, both controllers TRACKING, and IK `solver_ok` for both anchor targets.
  - **Anchor source (§6.2):** `last_sent_arms` if set, else measured arms. On recovery, the anchor is FK of the held IK solution at the current `waist_cmd`.
  - **Mode transitions under `--manipulation` (§5.2):**
    - OFF + ABXY → PLANNER;
    - PLANNER + BY → MANIPULATION;
    - MANIPULATION ↔ FROZEN on BY;
    - any running + ABXY → OFF;
    - AB / XY in FROZEN step the locomotion mode;
    - everything else (A+X, left-axis click) → unchanged.

- [ ] **Step 1:** Write the tests:
  - `test_entry_refused_without_feedback`, `test_entry_refused_stale_feedback` (age 0.15), `test_entry_refused_controller_lost`, `test_entry_refused_ik_failure`. Each asserts that the refusal reason is returned and the socket received **no** planner message.
  - `test_never_anchors_to_zeros`.
  - `test_first_entry_waist_from_measured`, `test_reentry_keeps_waist_cmd`.
  - `test_message_has_no_vr3pt_and_waist_first`.
  - `test_waist_only_preserves_arm_joints`: controllers still, waist stick at full range for 2 s → sent arm q change < 1°.
  - `test_manip_to_frozen_snapshot_is_last_sent_ik`, `test_manip_to_frozen_keeps_heading`, `test_frozen_hands_hold_last_sent`.
  - `test_frozen_walk_waits_for_lock_release`: with the lock on, movement stays 0 until a matching ack with α 0 arrives.
  - `test_lock_reengaged_after_walk_stops`: stick < 0.12 and static mode → the next message has `waist_lock = 1`.
  - `test_frozen_walk_refused_on_disabled_backend`, `test_sticks_ignored_on_disabled_backend` (waist, height and heading unchanged).
  - `test_left_lost_while_walking_stops`, `test_lost_arm_holds_and_stick_centered`.
  - `test_recovery_reanchors_to_held_command`: inject a measured-vs-commanded offset; the sent arm q is continuous (no step > 0.01 rad) across recovery.
  - `test_reader_stall_then_resume_reanchors_both_arms`:
    1. Stall the reader for 1 s, then resume with both controllers 20 cm away from their pre-stall poses.
    2. Until each arm's `ready_to_reanchor()` fires, every sent arm q equals the held command exactly.
    3. On the re-anchor tick there is no step > 0.01 rad.
    4. Afterward, moving a controller by +5 cm along operator-forward moves `target_p` by +5 cm ± 1 mm along torso-forward, relative to the held pose (not to the pre-stall anchor).
  - `test_op_neutral_reflects_physical_stick`: during a pending lock release (static message), with the walk stick at 0.5 → `op_neutral == 0`; with the stick at 0.05 → 1; with the left controller LOST → 0.
  - `test_manip_debug_mirrors_planner`: each `manip_debug` message has the same planner fields as the planner message sent on that tick, plus `t_mono_ns`.
  - `test_ik_failure_holds_then_lost_after_0_5s`.
  - **`StillnessCheck(p).update(fb_sample, now) -> tuple[bool, str | None]`**: this is the G1b Step 8 feature, matching `g1b_still2.py` exactly. Reuse its window function rather than re-deriving it. Only samples received by `now` count. The window must cover `still_window_s`, with the newest sample's age < `feedback_max_age_s` and no gap > 2 feedback periods. Still iff both:
    - `D` < `still_disp_max`: `D` = max over the 12 leg joints of |`body_q`(newest) − `body_q`(last sample at or before now − `still_window_s`)|, in rad;
    - `G` < `still_gyro_mean_max`: `G` = ‖mean `base_ang_vel` over the window‖, in rad/s.
    - It is used only when `still_check_adopted`. **Deferred (2026-11-12): not built while no check is adopted (Step 9 FAILED). The three tests below come with it. WalkGate keeps its `still` input and adopted-scope tests with injected values.**
    - `test_stillness_requires_full_fresh_window`: a 0.9 s window → not still; a stale sample → not still `"feedback"`; a gap → not still.
    - `test_stillness_threshold_each_signal`: one leg joint's net displacement over the limit → `"body_q_disp"`; mean gyro over the limit → `"base_ang_vel_mean"`.
    - `test_stillness_matches_gate_feature`: use a small feedback slice of `g1b_runC.npz` committed as a test fixture (`gear_sonic/tests/manipulation/fixtures/g1b_runC_fb_slice.npz`, ≤ 1 MB). `StillnessCheck` and `g1b_still2`'s feature must give identical decisions. If the fixture is missing, the test fails; it never skips.
  - **IK tracking-error check (PRD §6.6; required because G2 found an unreported stall):**
    - `test_ik_stall_reseeds_without_jump`: the warm solution's FK is 15 cm off with `solver_ok` True, and the posture-seeded one is on target. Expect: a stall is detected; each tick the sent arm q moves ≤ `ik_reseed_rate`·dt per joint; the sent FK error ends < 3 cm; `manip_debug` logs the re-seed.
    - `test_ik_reach_limit_is_not_failure`: real `ArmIK`, target 2 m away. Re-seeding doesn't help, so the closest solution is sent; after 2 s there is no loss and no warning (Review Focus 3).
    - `test_ik_stall_regression_g2_sweep`: real `ArmIK`. Replay the seeded G2 random sweeps (reachable FK targets that the warm solver stalled on, p99 8.7 cm) through the streamer's IK step, holding each final target for 2 s. Expect: the sent FK error at the end of every hold is < 3 cm / 10°.
  - `manip_debug` also carries, per side, `ik_err_pos`, `ik_err_rot` and `ik_reseed` (none / stall / reach_limit).
  - `test_next_mode_table`: every (mode, chord) pair, including A+X and left-click doing nothing.
  - `test_idle_squat_used_when_idle_ignores_height`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement the reader and feedback changes and `streamer.py`.
- [ ] **Step 4:** Run `pytest gear_sonic/tests -v`. Expected: all PASS.
- [ ] **Step 5:** Checkpoint (suggested message `feat(manipulation): ManipulationStreamer and reader controller samples`).

### Task 14: Wire `--manipulation` into the manager loop

**Files:**
- Modify: `gear_sonic/scripts/pico_manager_thread_server.py`:
  - `StreamMode`: add `MANIPULATION = 6`;
  - `run_pico_manager` (`:1900-2160`): when `manipulation=True`, build `ManipulationStreamer` + `ChordArbiter`, replace the inline transition block with `next_mode_manipulation`, dispatch to `run_manipulation_once` / `run_frozen_once`, and print entry refusals;
  - recording events (`:2127-2132`): unchanged in both modes.
  - `run_pico_manager` gets an optional `reader=None` parameter. When given, it is used instead of `_init_input_source(...)`. This is the test and replay hook; behavior is unchanged when it is `None`.
  - the controller-dict helpers (`get_controller_inputs`, `get_controller_axes`, `get_axis_clicks`, `get_abxy_buttons`, `:614-750`) also take the dict path for any reader whose class is registered in `_ISAAC_TELEOP_READERS`. The Task 16 `ReplayReader` registers itself there, and nothing else changes.
  - argparse (`:2162+`): `--manipulation`, `--waist-lock`.
- Modify: `docs/source/user_guide/teleoperation.md` (new section "Manipulation mode": launch flags, the controls table from PRD §5.1, the recording shortcuts (unchanged: A/B + left grip), with a **warning box at the top of the section**: "Holding the left grip while pressing B+Y switches mode **and aborts the current recording**, discarding the episode. Release the left grip before B+Y, or stop recording first." prompts, the backend limitation from §6.6, the walking height band)
- Test: `gear_sonic/tests/manipulation/test_manager_wiring.py`. A scripted fake reader is passed through `reader=`, the PUB socket is replaced by a capture, and the decoded `manager_state` messages are asserted.

**Interfaces:**
- Consumes: `ManipulationStreamer`, `next_mode_manipulation` (Task 13).

- [ ] **Step 1:** Write the tests:
  - `test_default_launch_path_unchanged`: with `manipulation=False`, drive a scripted button sequence through `run_pico_manager` for N ticks. The mode trajectory equals today's for A+X / B+Y / left-click / ABXY.
  - `test_default_recording_shortcuts_unchanged`: `manipulation=False`, A + left grip 0.8 → exactly one `toggle_data_collection = True`; B + left grip 0.8 → exactly one `toggle_data_abort = True`.
  - `test_manipulation_recording_shortcuts_unchanged`: `manipulation=True`, same inputs as the default test → the same events. With the left grip held at 0.9, a B+Y entry produces exactly one `toggle_data_abort` (the documented overlap).
  - **Re-centre rule (implemented in the manager wiring):** after a controller recovers from LOST or a reader stall, each stick is ignored (reads as centered, `op_neutral` computed as not neutral) until it has been seen inside the deadzone once.
  - `test_stick_must_recentre_after_reconnect`: a reader stall with the right stick held at 0.8 in MANIPULATION, then resume still deflected → `waist_cmd` unchanged until the stick re-centres.
  - `test_manipulation_launch_disables_pose_and_vr3pt`.
  - `test_manager_state_reports_manipulation_mode`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the full `pytest gear_sonic/tests -v`. Expected: PASS.
- [ ] **Step 5:** Checkpoint (suggested message `feat(teleop): --manipulation launch mode`).

### Task 15: Data recording

**Files:**
- Modify: `gear_sonic/scripts/run_data_exporter.py` (`_handle_planner_message` `:370`, `_add_sonic_pose_features` `:667`, `_add_cpp_state_features` `:618`, `_handle_manager_state` `:356`)
- Modify: `pico_manager_thread_server.py` manager_state send (`:2133`): add `capture_backend` (string → int code, 0 = xrt, 1 = isaac-teleop) **only when `manipulation=True`**, so default-session manager_state stays byte-identical.
- Test: `gear_sonic/tests/manipulation/test_exporter_fields.py`

**Interfaces:**
- Produces per frame:
  - `teleop.upper_body_position` (17, f32; zeros + `teleop.upper_body_valid = False` when absent);
  - `teleop.waist_cmd` (3) = `upper_body_position[:3]`;
  - `teleop.height_cmd` (the sent `height`);
  - `teleop.waist_lock` (sent value, −1 when absent);
  - `teleop.ctrl_state` (2);
  - `observation.waist_lock_alpha`, `observation.waist_lock_enabled`, `observation.watchdog_latched` from g1_debug, **only in episodes whose manager_state carries `capture_backend`** (i.e. `--manipulation` sessions). Default-session datasets keep today's schema even though `g1_debug` now carries the additive keys.
- Produces per episode: `capture_backend`, `waist_lock_enabled` in episode metadata.

- [ ] **Step 1:** Write `test_new_planner_fields_recorded` (a packed planner message with the new fields goes through the handler, then the frame has the exact values) `test_old_messages_still_record` (an old-style message gives defaults, no exception) and `test_default_session_schema_unchanged` (a default-session episode with the new `g1_debug` keys gives exactly today's column set).
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Checkpoint (suggested message `feat(exporter): record manipulation commands and lock state`).

### Task 16: Sim test harness, metrics script, and the §7 test campaign

**Files:**
- Create: `gear_sonic/scripts/manipulation_sim_tests.py`. It holds the harness, `ReplayReader`, `SCENARIOS`, and the CLI `--scenario <id> --runs 5 --lock {on,off} --out <dir>`.
- Create: `gear_sonic/scripts/manipulation_metrics.py`. It holds `compute_metrics`, `passes`, `REQUIRED_EVIDENCE`, and the CLI `<dir> --test <id>`.
- Create: `gear_sonic/utils/mujoco_sim/scenes/g1_manip_object.xml`. Copy the default G1 scene and add a fixed table and a 5 cm, 0.1 kg free-joint cube.
- Modify: `gear_sonic/utils/mujoco_sim/configs.py` / `base_sim.py`:
  - `scene_override: str | None = None`;
  - `object_init_pose: list[float] | None = None` (7, in world);
  - `apply_wrench: list[str] = []`, where each entry is `"body,fx,fy,fz,t0_sim,t1_sim"` applied through `mj_data.xfrc_applied`.
- Test: `gear_sonic/tests/manipulation/test_metrics.py`, `gear_sonic/tests/manipulation/test_replay_reader.py`

**Process topology (every scenario run).** Each is a separate OS process on one host:
- **P1, sim:** `run_sim_loop.py --no-elastic-band --publish-ground-truth --gt-bodies left_wrist_yaw_link,right_wrist_yaw_link[,cube]`, plus the scene, object and wrench options per scenario.
- **P2, deploy binary (sim):** per scenario, `--waist-lock` and `--input-watchdog-ms 200`. It connects to the planner path through P4.
- **P3, streamer:** `manipulation_sim_tests.py --role streamer --scenario <id> --t0-ns <T0>`. It runs `run_pico_manager(manipulation=True, reader=ReplayReader(script, t0_ns=T0), ...)`. This is the real manager loop and the real `ManipulationStreamer`, and its feedback comes through P4.
- **P4, two `TcpFaultProxy`s:** one on the planner path (deploy → proxy → P3's PUB) and one on the feedback path (P3's `FeedbackReader` → proxy → deploy 5557).
- **P5, recorder (harness process):** `GateRecorder` connects **directly**, not through the proxies, to:
  - P3 `manip_debug` and `manager_state`;
  - P2 `g1_debug`;
  - P1 `sim_gt`;
  - plus the harness's own `markers` stream: `(name, t_mono_ns)` for motion start/end, toggles and fault on/off, all in `T0`-relative script time converted to absolute monotonic time.

**Evidence and sync windows per scenario.** Each scenario declares its analysis windows (marker pairs) and a `StreamReq` per stream, from which `REQUIRED_EVIDENCE` is generated:
- `sim_gt`, `g1_debug` and `markers` are always required to be continuous. Feedback and network faults act on the proxied paths, never on the recorder's direct connections.
- `manip_debug` and `manager_state` are continuous, except in **9e**: one declared absence `(kill, restart)`. In **9e2** the streamer stays up and keeps publishing locally during a network blackhole, so `manip_debug` stays continuous there.
- `TcpFaultProxy` writes its own `markers` (`proxy_<path>_blackhole_on/off`, `proxy_<path>_delay=<s>`) plus `proxy_<path>_last_forward` with the source time of the last forwarded byte before a blackhole. Metrics measure the latch time from that marker, and the marker shows the fault was actually applied.

**Fault mechanisms.** Each is named so it's clear what each test actually exercises:

| Fault | Mechanism | Process |
|---|---|---|
| Controller freeze (pose and stick), occlusion (`valid=False` or stale `sample_ts`), spike, fast swing, reader stall | `ReplayReader` script | P3 |
| IK failure | `ReplayReader` scenario flag that wraps `ArmIK` in a failing shim for scripted ticks | P3 |
| Feedback delay / blackhole | feedback `TcpFaultProxy` | P4 |
| Stale `α == 0` sample, reorder, duplicate (8d) | **in-process** `FaultyFeedback` wrapper around `FeedbackReader` in P3, since TCP can't reorder | P3 |
| Network drop | `blackhole(True)` on **both** proxies; connections stay open | P4 |
| Streamer crash | `SIGKILL` P3; scenario 9e then restarts P3 and checks it re-enters only through start + entry | harness |
| Measured-vs-commanded arm offset (9) | `apply_wrench` on `right_wrist_yaw_link` | P1 |
| Object contact (7b) | `g1_manip_object.xml` + `object_init_pose` | P1 |

**7b object setup and its precondition.**
- `object_init_pose` = the cube center at the right palm, computed from FK of a fixed pre-grasp arm configuration that the script drives the controller to.
- The script closes the trigger and grip to 1.0 over 1 s, then raises the controller by 5 cm.
- **Precondition:** the cube must follow the wrist within 1 cm, in the `right_wrist_yaw_link` frame.
- If it doesn't, the run is recorded `setup_failed`. That counts as missing evidence, so the test fails, never passes.

**Interfaces:**
- Consumes: `GateRecorder`, `check_sync`, `align` (Task 1); `TcpFaultProxy`, `sim_stack` launch logic (Task 11); `ArmIK.fk_wrist_in_torso` (Task 5); `ManipulationParams` (`cmd_to_z`); `manip_debug` (Task 13).
- Produces:
  ```python
  class MissingEvidence(Exception): ...
  REQUIRED_EVIDENCE: dict[str, dict[str, StreamReq | list[str]]]   # test id → per-stream StreamReq (incl. declared absences) and required "stream.field" paths
  def compute_metrics(run_dir: str, test_id: str) -> dict   # raises MissingEvidence; runs check_sync on the test window
  def passes(test_id: str, runs: list[dict | Exception]) -> tuple[bool, list[str]]   # all 5 runs present, none an exception, every §7.2 threshold met
  ```
- **Metric keys** (§7.1 windows: baseline 1.0 s, settle 2.0 s, final = mean over the last 0.5 s, oscillation = peak-to-peak over the last 1.0 s):

| Key | Source |
|---|---|
| `pelvis_yaw_{peak,final}_deg`, `pelvis_xy_{peak,final}_m`, `osc_{xy_p2p_m,yaw_p2p_deg}` | `sim_gt` pelvis |
| `waist_err_final_deg[3]`, `waist_peak_dev_deg` | `g1_debug.body_q_measured[12:15]` vs `manip_debug.upper_body_position[:3]` |
| `wrist_pos_err_final_m{L,R}`, `wrist_rot_err_final_deg{L,R}` | FK of `body_q_measured` arm joints (torso frame) vs `manip_debug.target_p` / `target_R` |
| `wrist_peak_m{L,R}` (entry, toggle, loss), `wrist_T_change_m` (test 4) | FK of `body_q_measured`, relative to baseline |
| `arm_joint_change_deg` (test 4) | `body_q_measured` arm joints |
| `cmd_ref_max_step_rad`, `applied_ref_max_rate_rad_s` | `manip_debug.upper_body_position` per tick; `g1_debug.body_q_target` rate |
| `height_err_final_m`, `first_height_cmd`, `pelvis_z_speed_first_0_5s` | `sim_gt` pelvis z vs `cmd_to_z(manip_debug.height)` |
| `hand_cmd_step_at_toggle`, `hand_cmd_max_step_at_takeover`, `takeover_events` | `manip_debug` hands, `takeover_live`, prompts |
| `object_slip_m` | `sim_gt.bodies.cube` in the `right_wrist_yaw_link` frame, from 0.2 s before to 1.0 s after each toggle |
| `walk_start_vs_ack_s`, `alpha_at_walk_start` | first `manip_debug.movement ≠ 0` vs the first `g1_debug` sample with the matching `lock_ack_seq` and α == 0 |
| `t_latch_ms`, `t_standstill_s`, `moved_while_latched_m` | `markers` (`proxy_planner_last_forward`, or `kill` for 9e) vs `g1_debug.watchdog_latched` and `sim_gt` xy speed |
| `track_states`, `chords_fired`, `entry_refusals`, `record_events`, `fell` | `manip_debug` / `manager_state` / `sim_gt` pelvis z < 0.3 m |

- [ ] **Step 1:** Write the tests:
  - `test_baseline_and_windows`: a synthetic trace with a known step gives exact peak and final values.
  - `test_oscillation_p2p`.
  - `test_orientation_error_metric`: a 12° known target-vs-FK offset → 12° ± 0.1°.
  - `test_hand_step_at_toggle_metric`, `test_object_slip_in_wrist_frame`: the cube moves together with the wrist → slip 0.
  - `test_ref_continuity_metric`.
  - `test_missing_evidence_raises`: one test per stream class. Drop `sim_gt.bodies.cube` from a 7b run, drop `g1_debug.lock_ack_seq` from an 8d run, drop `manip_debug.target_R` from a test-3 run → `MissingEvidence`.
  - `test_sync_failure_fails_run`: a run with a 200 ms `g1_debug` gap in its window → that run is an exception, and `passes` is False with reason `sync`.
  - `test_9e_kill_window_accepts_marked_absence`: `manip_debug` absent between `kill` and `restart` → metrics computed. The same run with a `g1_debug` gap → `sync` failure.
  - `test_9e_fails_if_streamer_not_killed`: `manip_debug` continuous despite the `kill` marker → `fault not applied`.
  - `test_feedback_faults_do_not_relax_sync`: 8b/8d runs have **no** declared absences. Their `g1_debug` is recorded directly and must be continuous; the fault is evidenced by the proxy markers and `manip_debug.lock_gate_status`.
  - `test_setup_failed_fails_7b`.
  - `test_passes_requires_all_runs`: 4 of 5 → False.
  - One threshold test per §7.2 row, e.g. `test_t1_lock_on_waist_threshold_0_5deg`, on synthetic metric dicts.
  - `test_required_evidence_covers_every_test_id`: the keys of `REQUIRED_EVIDENCE` equal the §7.2 id list.
  - `test_replay_reader_fault_timeline` (in `test_replay_reader.py`): a frozen controller repeats the exact sample and `sample_ts`; a spike appears on exactly one sample; stick values hold while frozen.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement the harness, scenarios, sim options, scene and metrics. Scenarios map 1:1 to the §7.2 ids: 1, 2, 3, 4, 5, 6, 7, 7b, 7c, 8, 8b, 8c, 8d, 9, 9b, 9c, 9d, 9e, 9e2, 9e3, 9f, 10, 11, 12. Test 12 is the only operator step. It uses the same topology, but P3 runs with the real reader (`--input-source xrt|isaac-teleop`) instead of `ReplayReader`. The harness prints the instructions and writes `markers` on operator keypresses. Loss on the real backends is produced physically (cover the controller, or turn it off).
- [ ] **Step 4:** Run the unit tests. Expected: PASS. Then run one smoke scenario end to end: `python gear_sonic/scripts/manipulation_sim_tests.py --scenario 1 --runs 1 --lock off`. Expected: a run directory whose `compute_metrics` returns without `MissingEvidence`.
- [ ] **Step 5:** Run the campaign in PRD §9 order:
  1. lock off: 1–5, 7b, 7c, 9–9d, 10;
  2. lock on: 1–5, 8–8d;
  3. FROZEN, toggles, chords and watchdog: 6–8, 9e, 9e2, 9e3, 9f, 11;
  4. both backends: 12.

  Write a results table into a new PRD §12 "Sim results": test id, lock setting, backend, pass/fail, key metrics, and the fault mechanism used. Report failures and missing evidence as they are, and don't tune thresholds to pass.
- [ ] **Step 6:** Checkpoint (suggested message `test(manipulation): multi-process sim harness, evidence-checked metrics, campaign results`).

Real-robot rollout (PRD §9 step 5, gantry then free-standing) is an operator activity after Task 16 passes. It is not a coding task.
