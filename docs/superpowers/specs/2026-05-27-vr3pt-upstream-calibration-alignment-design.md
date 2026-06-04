# VR_3PT Controller-Only Path: Realign to Upstream Calibration Contract

**Status:** draft (design phase, awaiting user spec approval)
**Date:** 2026-05-27
**Branch:** to be created at implementation time from current `pico4u-g1-real-teleop-safety` (suggested name: `vr3pt-upstream-calibration-alignment`).
**Upstream reference:** NVlabs/GR00T-WholeBodyControl @ `4f5e118ae82e0448f2b2f441024d59fe74e0a4d9`

## Problem

Our controller-only VR_3PT teleop path (PICO headset + 2 controllers, no ankle trackers) diverges from upstream's `ThreePointPose` calibration math in three ways:

1. **Frame anchor**: wrists are expressed *head-relative* (subtracting `head_pos`) instead of pelvis-relative as upstream's `_process_3pt_pose` produces. Upstream's calibration formula assumes a stable pelvis-relative basis; our head-relative anchor moves with the operator's head, so the stored per-operator residual is correct on frame 1 and wrong on frame 2.
2. **Neck position**: our `_apply_calibration` reads the headset position and stores a `_calibration_neck_pos_offset`. Upstream synthesizes the third point's position purely from the kinematic chain `[0, 0, 0.05] + 0.35 * neck_z` — VR position is never read for the neck. Our deviation injects headset translation into the policy's torso target.
3. **Wrist orientation special case**: a `_calibration_wrist_orientation_uses_neck` flag was added that bypasses `calib_inv_rot` on `reset_with_measured_q`. Upstream always applies the neck-relative correction.

The user's stated goal: VR_3PT controls upper body naturally while Gear-Sonic's planner handles locomotion. This does not require ankle sensors. It does require the upper-body math to match what the policy was trained on.

## Goal

Realign the controller-only path to upstream's calibration contract by:

1. Inserting a *virtual pelvis* between raw XR poses and `ThreePointPose`, derived from headset position + headset yaw.
2. Reverting `ThreePointPose._capture_calibration` and `_apply_calibration` to their upstream behavior — strict pelvis-relative input, kinematic-chain neck position, unconditional neck-relative wrist correction.
3. Adding a SLERP-based low-pass on the headset orientation so quick head movements don't pump into the torso target.
4. Re-checking the `openxr_unitree` basis convention numerically and via a hardware-log gate.

Out of scope: SMPL teleop path (no intentional behavior change beyond restoring upstream `ThreePointPose` behavior — if a previous local diverge had altered the SMPL path's observable output, that change is reverted as a side-effect), upstream's locomotion policy, anything outside `gear_sonic/scripts/pico_manager_thread_server.py` and its tests. SMPL parity is verified by the same golden test using SMPL-shaped pelvis-relative fixtures (see Section 5.B).

## Architecture

```
PICO XR (OpenXR convention) → ControllerPoseReader → raw XR samples
                                  │
                                  ▼  (single place that converts full samples)
_process_controller_3pt_pose
  ├─ Q_openxr basis transform → robot-frame poses for L, R, head
  ├─ Apply legacy --left_controller_offset_rpy bias to L wrist orientation only (if nonzero)
  ├─ Apply legacy --right_controller_offset_rpy bias to R wrist orientation only (if nonzero)
  └─ Delegate to ControllerCalibState.process(...)
                                  │
                                  ▼
ControllerCalibState (owns: operator_drop_z, orn_filter, z_filter, last_pelvis_yaw_quat)
  ├─ head_quat_filt = orn_filter.update(head_rot_r.as_quat(scalar_first=True), dt)
  ├─ head_z_filt    = z_filter.update(head_pos_r[2], dt)
  ├─ pelvis_pos  = [head_pos_r[0], head_pos_r[1], head_z_filt - operator_drop_z]
  ├─ pelvis_quat = yaw_only(head_quat_filt, fallback=last_pelvis_yaw_quat)
  └─ Returns (3, 7) pelvis-relative rows for L wrist, R wrist, filtered head
                                  │
                                  ▼  (consumes upstream-shape pelvis-relative input)
ThreePointPose._capture_calibration / _apply_calibration   [reverted to upstream]
  ├─ neck quat anchor   inv(initial_filtered_neck_quat)
  ├─ wrist pos residual = lwrist_pos_corrected − g1_lwrist_FK_pos
  ├─ wrist orn residual = g1_lwrist_rot * lwrist_rot_corrected.inv()
  └─ neck position via kinematic chain [0, 0, 0.05] + 0.35 * neck_z
                                  │
                                  ▼
ZMQ publish → deploy → policy
```

### Key architectural decisions

- **Virtual pelvis is per-frame**, recomputed every iteration from the (filtered) headset pose. No tracker to estimate; substitution for SMPL's real pelvis.
- **`operator_drop_z` is captured by the startup calibration flow** (the lifecycle event the NVlabs docs call CALIB_FULL when `ThreePointPose` is fresh), not hardcoded. CLI default `--operator_drop_z 0.65` is only a fallback used until the first calibration. After the first capture, the captured value (from robot-frame `head_pos_r.z`) is used.
- **`ThreePointPose` is mode-agnostic**: both SMPL and controller-only paths produce upstream-shape pelvis-relative (3, 7) input. No `_calibration_neck_pos_offset` or `_calibration_wrist_orientation_uses_neck` flag exists on the class.
- **The SLERP-EMA on headset orientation** sits before pelvis-relative conversion. The same filtered headset drives both `pelvis_quat` (via `yaw_only`) and the third point's published orientation. This keeps the pelvis frame and the 3rd-point row in the same rotational reference at every frame, eliminating the warn-state filtered/unfiltered mismatch the critic flagged.
- **`openxr_unitree` stays as the basis convention** for both controllers and headset. It maps OpenXR rotation axes correctly so operator forearm twist appears on the G1 wrist's longitudinal axis. The static residual between PICO "identity" and G1 URDF wrist-yaw "identity" is absorbed by `calibration_lwrist_rot_offset` (upstream-style residual capture). `xrobotoolkit_unity` would require reapplying the SMPL per-joint OFFSETS, which are SMPL-frame artifacts that do not generalize to PICO hardware.

## Components

Single-file change in `gear_sonic/scripts/pico_manager_thread_server.py`. Tests, fixtures, and docs in new sibling files.

### New helper classes

**`HeadsetOrientationFilter`** (~30 LoC near `XRStalenessWatchdog`):

```python
class HeadsetOrientationFilter:
    """SLERP-based exponential moving average on a quaternion (w, x, y, z)."""

    DT_CLAMP_MAX_S = 0.05  # robustness cap on single-step alpha

    def __init__(self, tau_s: float = 1.0):
        self.tau_s = float(tau_s)
        self._state: np.ndarray | None = None

    def reset(self) -> None:
        self._state = None

    def update(self, quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
        if self._state is None:
            self._state = np.asarray(quat_wxyz, dtype=np.float64).copy()
            return self._state.copy()
        if dt <= 0.0:
            return self._state.copy()
        dt_eff = min(dt, self.DT_CLAMP_MAX_S)
        alpha = 1.0 - np.exp(-dt_eff / self.tau_s)
        # SLERP from self._state to quat_wxyz by alpha
        r_state = sRot.from_quat(self._state, scalar_first=True)
        r_new = sRot.from_quat(quat_wxyz, scalar_first=True)
        slerp = Slerp([0.0, 1.0], sRot.concatenate([r_state, r_new]))
        self._state = slerp(alpha).as_quat(scalar_first=True)
        return self._state.copy()
```

**`HeadsetZFilter`** (~20 LoC): same shape, scalar EMA.

### `ControllerCalibState` (new, ~110 LoC)

Owns `operator_drop_z`, both filters, and `_last_pelvis_yaw_quat`.

```python
class ControllerCalibState:
    IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

    def __init__(self, orn_tau_s: float = 1.0, z_tau_s: float = 0.3,
                 default_drop_z: float = 0.65,
                 drop_z_min: float = -0.5, drop_z_max: float = 3.0):
        self.default_drop_z = float(default_drop_z)
        self.operator_drop_z = float(default_drop_z)
        self.drop_z_min = float(drop_z_min)
        self.drop_z_max = float(drop_z_max)
        self.orn_filter = HeadsetOrientationFilter(tau_s=orn_tau_s)
        self.z_filter = HeadsetZFilter(tau_s=z_tau_s)
        self._last_pelvis_yaw_quat = self.IDENTITY_QUAT.copy()

    def capture(self, head_pos_r: np.ndarray) -> None:
        """Capture session operator drop-Z from robot-frame head position.

        Assumes G1 pelvis reference Z = 0 in the robot/policy frame.
        If a future robot has nonzero pelvis_ref_z, change to:
            operator_drop_z = head_pos_r[2] - g1_pelvis_ref_z
        """
        z = float(head_pos_r[2])
        if not np.isfinite(z) or z < self.drop_z_min or z > self.drop_z_max:
            print(f"[ControllerCalibState] WARNING: captured head_z={z:.2f} m "
                  f"outside [{self.drop_z_min}, {self.drop_z_max}]; falling back "
                  f"to default {self.default_drop_z:.2f} m.")
            self.operator_drop_z = self.default_drop_z
        else:
            self.operator_drop_z = z
            print(f"[ControllerCalibState] captured operator_drop_z = {z:.3f} m")
        self.reset_filters(preserve_yaw=False)

    def reset_filters(self, preserve_yaw: bool = True) -> None:
        """Reset filter state.

        preserve_yaw=True  (default): used by watchdog re-acquire and VR_3PT entry.
                                       Last good pelvis yaw is kept as singular-case fallback.
        preserve_yaw=False:            used by capture() (startup calibration flow).
                                       Pelvis yaw resets to identity — explicit re-anchor.
        """
        self.orn_filter.reset()
        self.z_filter.reset()
        if not preserve_yaw:
            self._last_pelvis_yaw_quat = self.IDENTITY_QUAT.copy()

    def process(self,
                l_ctrl_pos_r: np.ndarray, l_ctrl_rot_r: sRot,
                r_ctrl_pos_r: np.ndarray, r_ctrl_rot_r: sRot,
                head_pos_r:   np.ndarray, head_rot_r:   sRot,
                dt: float) -> np.ndarray:
        """Inputs are already in robot frame (openxr_unitree applied upstream).
        Each (pos, rot) pair matches the existing _pose_xr_to_robot(...) return
        shape: pos is np.ndarray (3,), rot is scipy.Rotation. No new dataclass
        needed; tuple unpacking happens in _process_controller_3pt_pose.

        Returns (3, 7) pelvis-relative row [x, y, z, qw, qx, qy, qz] per point.
        """
        for name, pos, rot in [
            ("l_ctrl", l_ctrl_pos_r, l_ctrl_rot_r),
            ("r_ctrl", r_ctrl_pos_r, r_ctrl_rot_r),
            ("head",   head_pos_r,   head_rot_r),
        ]:
            if pos.shape != (3,):
                raise ValueError(f"{name} pos malformed: shape={pos.shape}")
            if not np.all(np.isfinite(pos)):
                raise ValueError(f"{name} pos contains non-finite values: {pos}")
            # Duck-typing check: scipy.spatial.transform.Rotation is a Cython-backed
            # class and isinstance() can be brittle across imports / reloaded modules.
            # We require the .as_quat(scalar_first=...) method and verify the call
            # succeeds; that's both sufficient and stable.
            if not hasattr(rot, "as_quat"):
                raise ValueError(f"{name} rot must be a scipy.Rotation (or compatible); "
                                 f"got {type(rot)}")
            # scipy.Rotation internally stores a unit quaternion, but verify
            # the as_quat() output to catch any upstream construction that
            # bypassed normalization.
            q = rot.as_quat(scalar_first=True)
            if not np.all(np.isfinite(q)):
                raise ValueError(f"{name} quaternion contains non-finite values: {q}")
            q_norm = float(np.linalg.norm(q))
            if q_norm < 0.5 or q_norm > 2.0:
                raise ValueError(f"{name} quaternion norm = {q_norm} is far from 1; not a valid rotation")
        if dt < 0.0:
            raise ValueError(f"negative dt={dt}")

        head_quat_wxyz = head_rot_r.as_quat(scalar_first=True)
        head_quat_filt = self.orn_filter.update(head_quat_wxyz, dt)
        head_z_filt    = self.z_filter.update(float(head_pos_r[2]), dt)

        pelvis_pos = np.array([head_pos_r[0], head_pos_r[1],
                                head_z_filt - self.operator_drop_z], dtype=np.float64)
        pelvis_quat = yaw_only(head_quat_filt, fallback_quat=self._last_pelvis_yaw_quat)
        self._last_pelvis_yaw_quat = pelvis_quat

        row = np.zeros((3, 7), dtype=np.float32)
        r_pelvis_inv = sRot.from_quat(pelvis_quat, scalar_first=True).inv()
        l_quat = l_ctrl_rot_r.as_quat(scalar_first=True)
        r_quat = r_ctrl_rot_r.as_quat(scalar_first=True)
        for i, (p_pos, p_quat) in enumerate([
            (l_ctrl_pos_r, l_quat),
            (r_ctrl_pos_r, r_quat),
            (head_pos_r,   head_quat_filt),
        ]):
            row[i, :3] = r_pelvis_inv.apply(p_pos - pelvis_pos)
            row[i, 3:] = (r_pelvis_inv * sRot.from_quat(p_quat, scalar_first=True)).as_quat(scalar_first=True)
        return row
```

### `yaw_only` (new free function, ~15 LoC)

Singularity-safe yaw extraction with explicit fallback for straight-up/down headset:

```python
def yaw_only(quat_wxyz: np.ndarray, fallback_quat: np.ndarray | None = None) -> np.ndarray:
    r = sRot.from_quat(quat_wxyz, scalar_first=True)
    fwd = r.apply([1.0, 0.0, 0.0])
    xy_norm = float(np.linalg.norm(fwd[:2]))
    if xy_norm < 1e-6:
        if fallback_quat is not None:
            return np.asarray(fallback_quat, dtype=np.float64).copy()
        return np.array([1.0, 0.0, 0.0, 0.0])
    yaw = np.arctan2(fwd[1], fwd[0])
    return sRot.from_euler("z", yaw).as_quat(scalar_first=True)
```

### `_process_controller_3pt_pose` (modified)

Becomes thin wrapper around `ControllerCalibState.process`:

```python
def _process_controller_3pt_pose(
    l_ctrl_xr: np.ndarray, r_ctrl_xr: np.ndarray, head_xr: np.ndarray,
    dt: float, controller_calib: ControllerCalibState,
    left_controller_offset_rpy: tuple = (0, 0, 0),
    right_controller_offset_rpy: tuple = (0, 0, 0),
) -> np.ndarray:
    if controller_calib is None:
        raise ValueError("controller_calib required for PLANNER_VR_3PT mode")
    # _pose_xr_to_robot returns (pos: np.ndarray(3,), rot: scipy.Rotation).
    l_pos, l_rot = _pose_xr_to_robot(l_ctrl_xr, convention="openxr_unitree")
    r_pos, r_rot = _pose_xr_to_robot(r_ctrl_xr, convention="openxr_unitree")
    h_pos, h_rot = _pose_xr_to_robot(head_xr,   convention="openxr_unitree")
    # Legacy wrist orientation bias (orientation only; positions/headset untouched).
    if any(a != 0 for a in left_controller_offset_rpy):
        l_rot = l_rot * sRot.from_euler("xyz", left_controller_offset_rpy, degrees=True)
    if any(a != 0 for a in right_controller_offset_rpy):
        r_rot = r_rot * sRot.from_euler("xyz", right_controller_offset_rpy, degrees=True)
    return controller_calib.process(l_pos, l_rot, r_pos, r_rot, h_pos, h_rot, dt)
```

The wrist position subtraction-from-head logic from our current code — removed. The pelvis-relative transform inside `ControllerCalibState.process` replaces it.

### `ThreePointPose` — partial revert to upstream

| Field / method | Action | Upstream reference |
|---|---|---|
| `self._calibration_neck_pos_offset` | **Remove** field + all reads | not present upstream |
| `self._calibration_wrist_orientation_uses_neck` | **Remove** field + branches | not present upstream |
| `_capture_calibration` | Revert wrist residual logic to upstream's unconditional `calib_inv_rot * raw_wrist_rot`; remove neck pos offset capture | upstream `_capture_calibration` |
| `_apply_calibration` | Revert neck-position block to kinematic chain `[0, 0, 0.05] + 0.35 * neck_z`; remove the head-position-tracking branch | upstream `_apply_calibration` lines ~1149-1153 |
| `reset_with_measured_q` | Revert to upstream body (preserves neck, clears wrist offsets, sets `_calibration_pending = True`) | upstream `reset_with_measured_q` |
| `calibrate_vr_3pt_now`, `process_vr_3pt_pose` | Keep (calibrate_vr_3pt_now stays *synchronous capture*); argument always pelvis-relative (3, 7) row | n/a |

### Manager loop wiring

- Instantiate `controller_calib = ControllerCalibState(...)` next to `xr_watchdog`.
- Pass `controller_calib` reference into `PlannerStreamer` (or `_process_controller_3pt_pose` call sites).
- **Startup calibration flow** (`OFF` → `PLANNER` on `start_combo` rising edge):
  ```python
  # NOTE on naming: the NVlabs docs call this "CALIB_FULL" because the first
  # activation after a fresh ThreePointPose captures BOTH neck anchor and
  # wrist offsets. Subsequent activations preserve the existing neck anchor
  # (upstream `_capture_calibration` behavior at line ~1442) and only re-anchor
  # operator_drop_z + wrist offsets. We use "startup calibration flow" as the
  # lifecycle event name; "CALIB_FULL" applies precisely when ThreePointPose
  # is fresh (after construction or explicit ThreePointPose.reset()).
  raw = reader.get_latest()
  # Apply Q_openxr to the headset once, ONLY for drop-Z capture (separate from
  # the per-frame conversion inside _process_controller_3pt_pose).
  head_pos_r, _head_rot_r = _pose_xr_to_robot(raw.head_xr, convention="openxr_unitree")
  controller_calib.capture(head_pos_r)
  pelvis_relative = _process_controller_3pt_pose(raw.l_xr, raw.r_xr, raw.head_xr, dt=0.0,
                                                  controller_calib=controller_calib, ...)
  three_point.calibrate_vr_3pt_now(pelvis_relative)  # synchronous _capture_calibration
  ```
- **VR_3PT entry** (`left_axis_click`):
  ```python
  # CRITICAL ORDERING: filters reset BEFORE the gate evaluation so the gate
  # uses the same ControllerCalibState state that the first VR_3PT frame
  # will use. Otherwise the gate validates stale filter/yaw state and the
  # runtime publishes a different first target.
  #
  # Accepted side effect on gate refusal: filters are already reset by the
  # time the gate refuses. The operator's NEXT entry attempt starts from
  # those freshly-reset filters and the operator's then-current headset
  # orientation, not from the snapshot at the original click. This is
  # acceptable because (a) a refused gate already requires the operator to
  # adjust pose before retrying, and (b) the alternative (snapshot/restore
  # of filter state) adds complexity that does not match how an operator
  # actually behaves between attempts.
  controller_calib.reset_filters(preserve_yaw=True)   # filters only; drop_z preserved
  if not planner_streamer.check_vr3pt_entry_mismatch():
      new_mode = current_mode                         # stay in current mode
  else:
      planner_streamer.recalibrate_for_vr3pt()        # sets pending; next frame captures
      planner_streamer.start_vr3pt_ramp()             # existing gate→recal→ramp assertion
  ```
- **Watchdog warn → ok transition** (XR re-acquire): `controller_calib.reset_filters(preserve_yaw=True)`. No automatic recalibration (per official docs: wrist CALIB only on VR_3PT entry, not on tracking blips).
- **Watchdog estop**: terminal; manager exits. No recovery.

### CLI surface

```
NEW:
  --operator_drop_z 0.65              # fallback until startup calibration flow captures session operator height/drop estimate
  --operator_drop_z_min -0.5          # impossible-bound for capture sanity check
  --operator_drop_z_max 3.0           # impossible-bound for capture sanity check
  --headset_orn_lowpass_s 1.0         # SLERP-EMA τ — INITIAL default; final value set by τ sweep (Section 5.E)
  --headset_z_lowpass_s 0.3           # Z position EMA τ

LEGACY (still supported; used during validation phase):
  --left_controller_offset_rpy        # warn at startup if nonzero; applied as pre-calib bias
  --right_controller_offset_rpy       # same
```

Legacy-flag warning wording at startup when nonzero:

> `[Manager] Note: --left_controller_offset_rpy=(3.0, 0.0, -90.0) will be applied as a pre-calibration bias this run. Calibration absorbs the remainder of the controller-to-URDF wrist rotation automatically. Set to (0, 0, 0) to rely on calibration alone.`

## Data flow

Per-frame (controller-only mode, in `PLANNER_VR_3PT`):

```
PICO XR (OpenXR convention) → ControllerPoseReader (raw OpenXR samples)
        │
        ▼
_process_controller_3pt_pose(raw, dt, controller_calib):
  1) Q_openxr → (l_pos, l_rot), (r_pos, r_rot), (h_pos, h_rot)
  2) If --left_controller_offset_rpy nonzero:  l_rot = l_rot * R_left_bias
  3) If --right_controller_offset_rpy nonzero: r_rot = r_rot * R_right_bias
  4) Delegate to controller_calib.process(l_pos, l_rot, r_pos, r_rot, h_pos, h_rot, dt)
        │
        ▼
controller_calib.process:
  5) head_quat_filt = orn_filter.update(h_rot.as_quat(scalar_first=True), dt)
  6) head_z_filt    = z_filter.update(h_pos[2], dt)
  7) pelvis_pos  = [h_pos[0], h_pos[1], head_z_filt - operator_drop_z]
  8) pelvis_quat = yaw_only(head_quat_filt, fallback=_last_pelvis_yaw_quat)
  9) row[i] = pelvis_relative(point_i, pelvis_pos, pelvis_quat); row[2] uses head_quat_filt
 10) Return (3, 7)
        │
        ▼
ThreePointPose.process_vr_3pt_pose(row):
 11) If _calibration_pending: _capture_calibration(row)
 12) row_cal = _apply_calibration(row):
       row[2,3:]_cal = calib_inv * row[2,3:]
       row[0,3:]_cal = stored_L_rot_offset * (calib_inv * row[0,3:])
       row[1,3:]_cal = stored_R_rot_offset * (calib_inv * row[1,3:])
       row[0,:3]_cal = calib_inv.rotate(row[0,:3]) − stored_L_pos_offset
       row[1,:3]_cal = calib_inv.rotate(row[1,:3]) − stored_R_pos_offset
       neck_z        = row[2,3:]_cal.rotate([0,0,1])
       row[2,:3]_cal = [0, 0, 0.05] + 0.35 * neck_z      ← kinematic chain
        │
        ▼
ZMQ publish → deploy → policy
```

### Frame consistency

`pelvis_quat = yaw_only(filtered_head_quat, fallback=_last_pelvis_yaw_quat)` and `row[2, 3:] = inv(pelvis_quat) * filtered_head_quat` share the same filtered source. On fast head yaw, both lag together; published row is internally consistent.

**Hypothesis to validate on hardware**: a ~1 s lag on pelvis-frame motion is acceptable to the trained policy (whose SMPL-derived pelvis trajectory has some bandwidth we have not measured). If empirical sweep at τ ∈ {0.3, 0.5, 1.0} s shows instability at high τ, the default tightens. Not a proof — a tuning gate.

### Calibration lifecycle

| Event | Trigger | What happens |
|---|---|---|
| **Startup calibration flow — first activation (= CALIB_FULL in NVlabs docs)** | `A+B+X+Y` from `OFF`, `ThreePointPose` is fresh (no prior neck anchor) | `controller_calib.capture(head_pos_r)` sets `operator_drop_z` and `reset_filters(preserve_yaw=False)`. `three_point.calibrate_vr_3pt_now(pelvis_relative)` synchronously captures **both** neck quat anchor + wrist residuals vs FK(zero-q). This is the only activation that satisfies "full" semantics. |
| **STOP** | `A+B+X+Y` from any active mode | Exit to `OFF`. Per official docs. |
| **Startup calibration flow — subsequent activation** | `A+B+X+Y` from `OFF` (after STOP), `ThreePointPose` already has a neck anchor | Re-runs the startup flow. `operator_drop_z` and wrist offsets re-anchor. **Neck anchor is preserved** from the first activation (upstream `_capture_calibration` behavior at line ~1442). Documented as "wrist re-anchor against zero-q, preserved neck" — **not** CALIB_FULL despite using the same button. A true full re-calibration requires calling `ThreePointPose.reset()` first; no UI for this yet. |
| **VR_3PT entry** | `left_axis_click` from `PLANNER` / `PLANNER_FROZEN_UPPER_BODY` | (1) `controller_calib.reset_filters(preserve_yaw=True)` (filters only; drop-Z preserved) — **must run before the gate** so the gate evaluates with the same state the first VR_3PT frame will use. (2) Existing pose-mismatch gate. If refused: stay in current mode. (3) `planner_streamer.recalibrate_for_vr3pt()` (clears wrist offsets, preserves neck, sets pending — next frame captures wrist residuals vs measured-q). (4) `planner_streamer.start_vr3pt_ramp()` (existing gate→recal→ramp assertion). |
| **Watchdog warn (>50 ms)** | XR stale | `planner_streamer.freeze_vr3pt_target_once = True`. Re-publishes last good `_last_vr3pt_pose`. Filters NOT advanced (no `process()` call this iteration). |
| **Watchdog warn → ok** | XR re-acquired | `controller_calib.reset_filters(preserve_yaw=True)`. No automatic recalibration. |
| **Watchdog estop (>200 ms)** | Prolonged XR loss | Send stop command, manager exits. Terminal within session. |

### Numerical sanity for `openxr_unitree`

Pure right-controller forearm twist (rotation about controller-Z = controller longitudinal axis):
- `Q_openxr` maps `xr_Z → robot_-X`. Rotation axis lands on robot's X — which IS the G1 `right_wrist_yaw_link` URDF longitudinal axis. ✓
- `Q_xrobotoolkit_unity` would map `xr_Z → robot_+Y` (pitch axis) — wrong.

Static residual between PICO "identity" and G1 URDF wrist-yaw "identity" gets stored in `calibration_lwrist_rot_offset` / `calibration_rwrist_rot_offset`. Subsequent rotational motion translates correctly on each axis.

Empirical confirmation: hardware-log gate G2 below.

## Error handling and edge cases

### Filter first-call behavior
- `_state is None` → initialize from input, return input (any `dt`).
- `_state is set` and `dt <= 0.0` → return last state, do not snap.
- Otherwise → `alpha = 1 - exp(-min(dt, 0.05) / tau)`; SLERP for orientation, lerp for scalar.

### `yaw_only` singularity
- `xy_norm < 1e-6` (headset looking straight up/down) → return `fallback_quat` (caller's `_last_pelvis_yaw_quat`) or identity if no fallback.
- Operator briefly looks straight down → single-frame fallback to last yaw → no frame jump. Filter's ~1 s smoothing absorbs the glitch on its way in.

### `operator_drop_z` sanity
- Bounds `[drop_z_min=-0.5, drop_z_max=3.0]` configurable via CLI.
- Out-of-range or NaN capture → warning, fall back to `default_drop_z`. Operator re-CALIB.
- Hardware-log gate G1 measures the real `head_pos_r.z` distribution; defaults and bounds tighten after.

### `dt` clamp
- `DT_CLAMP_MAX_S = 0.05`. Bounds maximum single-step alpha during transient process stalls. Not pure time-based EMA in the [0.05 s, watchdog warn] band; acknowledged as a robustness cap. Long stalls trip the watchdog, which resets filters via `warn → ok` transition.

### Pre-startup-calibration operation
- `operator_drop_z = default_drop_z` (CLI fallback). `process()` works normally with the generic value. First `capture()` overwrites with session value.

### Watchdog `warn → freeze`
- Last good `_last_vr3pt_pose` republished by streamer. Filters not advanced. Frame consistency preserved because re-published row was the last filtered output.

### Startup calibration flow re-runs (subsequent activations after STOP)
- Second activation overwrites `operator_drop_z`, resets filters with `preserve_yaw=False` (explicit re-anchor). Wrist offsets re-captured. **Neck anchor preserved** (upstream behavior). Documented limitation.

### Input validation vs debug assertions
- Shape and finiteness checks in `process()`, `_process_controller_3pt_pose`, `capture()` use explicit `if ...: raise ValueError(...)`. Python `assert` is stripped under `-O` and is not safe for runtime validation in safety paths.
- `assert` reserved for internal invariants the path cannot violate at runtime (e.g. internal data structure shape after our own transforms).

### Legacy `--*_controller_offset_rpy` ordering
- Applied to controller wrist quaternions only, before the pelvis-relative transform inside `ControllerCalibState.process`. Not applied to positions, not applied to headset.
- Calibration absorbs the *remainder* of the controller-to-URDF rotation. Bias acts as a prior.
- Zero CLI values (no bias) is geometrically equivalent at steady state — calibration absorbs the full rotation in one shot. Bias is a tuning aid, not a correctness requirement.

## Testing

### Test layout

```
gear_sonic/tests/
├── test_headset_filters.py            (new)
├── test_yaw_only.py                   (new)
├── test_controller_calib_state.py     (new)
├── test_controller_calib_pelvis_anchoring.py   (new — head-relative regression guard)
├── test_three_point_pose_parity.py    (new — upstream golden)
├── fixtures/
│   ├── threepoint_golden.json         (new — pinned upstream SHA fixture)
│   └── regen_threepoint_golden.py     (new — regenerates fixture from a pinned SHA)
├── test_vr3pt_entry_gate.py           (existing — keep)
├── test_xr_staleness_watchdog.py      (existing — keep)
└── test_safety_smoke.py               (extend with new lifecycle drill items)
```

### A. Unit tests

**Filters (`test_headset_filters.py`)** — shared cases for both:
- First call after construction initializes state from input, returns input (any `dt`).
- First call after `reset()` re-initializes (any `dt`).
- `dt = 0.0` on a primed filter returns last state, does not snap.
- `dt = 0.06` (above clamp) yields same alpha as `dt = 0.05`.
- Long sequence with constant `dt`: exponential approach to target, ~63% at one τ.
- SLERP variant: feed quats 90° apart for 5τ; final state within 2° of target.

**`yaw_only` (`test_yaw_only.py`)**:
- Pure yaw input returns input.
- Yaw + pitch returns yaw-only quaternion.
- `fwd = [0, 0, 1]` (looking up) → returns `fallback_quat` (or identity).
- `fwd = [0, 0, -1]` (looking down) → same.
- `fwd = [0.01, 0.01, 0.99]` (just above 1e-6) → returns yaw from xy projection, not fallback.
- Singular + no fallback → identity (no exception).

**`ControllerCalibState` (`test_controller_calib_state.py`)**:
- Construction sets `operator_drop_z = default_drop_z`.
- `capture(valid_pos)` sets `operator_drop_z`, calls `reset_filters(preserve_yaw=False)` → `_last_pelvis_yaw_quat` == identity.
- `capture(out_of_range_pos)` falls back to default, prints warning, still resets filters.
- `capture(NaN_pos)` same as out-of-range.
- `process()` before `capture` uses default value; output shape (3, 7); no exception.
- `reset_filters(preserve_yaw=True)` clears filters; `_last_pelvis_yaw_quat` unchanged.
- `reset_filters(preserve_yaw=False)` resets `_last_pelvis_yaw_quat` to identity.
- Singular headset orientation in `process()` → `pelvis_quat` equals stored fallback, not identity.

**Head-relative regression guard (`test_controller_calib_pelvis_anchoring.py`)**:
- Two assertions to distinguish pelvis-relative from head-relative output (test bodies shown in Section 5 above).
- Catches the specific head-relative wrist subtraction bug regardless of `ThreePointPose` behavior.

### B. Upstream-parity golden test

**`test_three_point_pose_parity.py`** — verifies our `ThreePointPose._apply_calibration` produces numerical parity (`np.allclose(..., atol=1e-7)`) with upstream at the pinned SHA across both controller-derived and SMPL-derived pelvis-relative inputs.

Fixture metadata:
```json
{
  "_metadata": {
    "source_repo": "https://github.com/NVlabs/GR00T-WholeBodyControl",
    "source_commit": "4f5e118ae82e0448f2b2f441024d59fe74e0a4d9",
    "source_file": "gear_sonic/scripts/pico_manager_thread_server.py",
    "source_class": "ThreePointPose",
    "generated_at": "2026-05-27",
    "generator": "gear_sonic/tests/fixtures/regen_threepoint_golden.py"
  },
  "calib_input": [...],
  "frames": [
    {"label": "neutral",                    "input": [...], "expected_upstream": [...]},
    {"label": "mild_lean_forward",          "input": [...], "expected_upstream": [...]},
    {"label": "left_wrist_twist",           "input": [...], "expected_upstream": [...]},
    {"label": "right_wrist_lift",           "input": [...], "expected_upstream": [...]},
    {"label": "head_yaw_left",              "input": [...], "expected_upstream": [...]},
    {"label": "smpl_neutral_pelvis_rel",    "input": [...], "expected_upstream": [...]},
    {"label": "smpl_arms_raised_pelvis_rel","input": [...], "expected_upstream": [...]}
  ]
}
```

Regenerator script: checks out the pinned commit in a temporary worktree, imports upstream `ThreePointPose`, processes each input, writes the JSON. Updating the golden is a documented workflow that records the new SHA explicitly.

Test asserts `np.allclose(out, expected, atol=1e-7)`. Failure blocks merge.

### C. Hardware-log validation gates (merge-blocking)

**Gate G1: `head_pos_r.z` distribution.**
- 60 s session, operator standing normally, deploy + manager running.
- Log every `head_pos_r.z` post-`Q_openxr`.
- Required: median in a 10 cm band; all samples within `[drop_z_min, drop_z_max]`; captured `operator_drop_z` at the startup calibration flow within 10 cm of session median.
- Output: `docs/superpowers/specs/2026-05-27-hardware-validation/G1_drop_z_distribution.md` with histogram, median, bounds. Update default `--operator_drop_z` if needed.

**Gate G2: forearm-twist axis check.**
- 30 s session, operator twisting right controller about its longitudinal axis, starting from a held neutral pose.
- Log published `vr_3point_local_orn_target[1]` (right wrist quat after calibration).
- Method: take a held-neutral frame `q_neutral` and a twisted frame `q_after`. Compute the **relative** rotation `R_rel = R_after * R_neutral.inv()`. Decompose `R_rel` as rotation-vector → axis. Compare `axis` to the G1 `right_wrist_yaw_link` local X axis (expressed in the same pelvis-relative frame the policy sees, which is computed via FK on the held-neutral robot pose). Use absolute dot product to allow axis sign ambiguity:
  ```
  axis_alignment = abs(np.dot(axis, link_local_x))
  ```
- Required: `axis_alignment > cos(10°) ≈ 0.985` for the median of all twisted samples.
- Rationale for using relative rotation: raw published quats include the static `calibration_rwrist_rot_offset`, which is whatever residual was captured at the startup calibration flow. Comparing raw quat axes against the URDF X axis would conflate the static residual with the motion axis we actually care about.
- Output: `docs/superpowers/specs/2026-05-27-hardware-validation/G2_forearm_twist_axis.md`.
- Failure: pause spec, re-derive basis convention.

### D. Smoke test extensions

Append to `test_safety_smoke.py` (manual operator confirmation):

5. **Startup calibration re-run preserves neck**: trigger STOP, then START. Verify `operator_drop_z` re-anchored, neck anchor preserved (operator looks left after second START — robot torso target reflects original first-activation calibration's forward direction, not the new headset orientation).
6. **VR_3PT entry filter reset**: from `PLANNER`, click Left Stick. Verify console logs filter reset; watchdog reports `idle → ok`.
7. **XR re-acquire filter reset**: cover headset cameras ~80 ms then uncover. Console logs filter reset on `warn → ok`; next published target shows no single-frame jump.

### E. Manual τ sweep — tuning target, not pass/fail

- Run a fixed task in sim with τ ∈ {0.3, 0.5, 1.0} s, two operators, 0–5 subjective ratings.
- **Responsiveness**: sustained body lean held ≥2 s produces visible robot torso command in comfortable lag.
- **Quick-glance suppression**: brief head turn <0.5 s does not noticeably swing robot torso.
- **Slow-nod ambiguity is acknowledged**: slowly lowering chin will be misread as torso lean by all τ. Operators briefed that this is a fundamental limit of headset-only tracking. Spec does not promise to solve it.
- Winning τ → default written into final spec. No code change required.

### F. CI matrix

- Unit tests + parity test: every PR, headless.
- Hardware-log gates G1/G2: manually before merge, reports in PR description.
- Smoke test: before any real-hardware session per operator drill.

## Implementation summary

| Layer | Change scope | LoC est |
|---|---|---|
| `_pose_xr_to_robot` helper | unchanged | 0 |
| `_process_controller_3pt_pose` | rewritten as thin wrapper | ~40 |
| `HeadsetOrientationFilter` | new | ~30 |
| `HeadsetZFilter` | new | ~20 |
| `yaw_only` | new | ~15 |
| `ControllerCalibState` | new | ~110 |
| `ThreePointPose` | partial revert (remove 2 fields, restore 2 methods) | ~−40 net |
| Manager loop wiring | reset hooks at calibration triggers | ~20 |
| CLI parsing | 5 new flags; legacy warning | ~30 |
| Unit + parity tests | new | ~400 |
| Hardware-log validation reports | new | ~80 |
| Docs update | new (`real_robot_safety.md` standing-only constraint) | ~10 |

**Net Python LoC delta: ~+700, of which ~400 is tests.**

## Acceptance criteria

The spec is considered implemented when:

1. All unit tests pass in CI.
2. The upstream-parity golden test passes (`ThreePointPose._apply_calibration` matches NVlabs @ `4f5e118ae82e0448f2b2f441024d59fe74e0a4d9` to `atol=1e-7`).
3. Hardware-log Gate G1 (`head_pos_r.z` distribution) report committed; defaults updated if needed.
4. Hardware-log Gate G2 (forearm-twist axis check) report committed showing axis alignment within 10°.
5. Smoke test items 5/6/7 confirmed by operator drill.
6. τ sweep complete; final default chosen; documented in spec.

## Out-of-scope items (deferred)

- Full neck re-calibration (currently requires manual `ThreePointPose.reset()`; no UI). Add button combo or CLI command in a follow-up if operators request.
- SMPL teleop path — no intentional behavior change beyond restoring upstream `ThreePointPose` behavior. If a previous local diverge had altered the SMPL path's observable output (e.g. our `_calibration_neck_pos_offset` consumption in `_apply_calibration` affected SMPL frames too), that change is reverted as a side-effect of restoring upstream behavior. SMPL parity is verified by the same golden test using SMPL-shaped pelvis-relative fixtures (Section 5.B).
- `compute_controller_3pt_offsets.py` — kept as a utility but no longer required since calibration absorbs the controller-to-URDF rotation automatically. May be repurposed as a debug tool for inspecting captured offsets.
- LeRobot v2.1 data collection integration — covered by existing `gear_sonic/scripts/run_data_exporter.py`, untouched by this spec.

## Risks

- **`openxr_unitree` empirical validation fails (Gate G2).** Spec paused, basis convention re-derived. The numerical sanity in Section 3 suggests this won't fail, but hardware data is the only proof.
- **τ = 1.0 s is too sluggish for natural feel.** The sweep produces a smaller default. The slow-nod ambiguity remains regardless of τ.
- **`operator_drop_z` capture is noisy across sessions** (operator stance varies). Mitigation: the capture happens at each startup calibration flow; default fallback covers between-calibration sessions.
- **Upstream `ThreePointPose` evolves after our PR.** Parity test catches it; the golden fixture's pinned SHA documents what "matches upstream" meant when we shipped.
