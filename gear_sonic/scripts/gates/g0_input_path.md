# G0 Step 5: input path, streamed `smpl_pose` / `body_quat_w` to the policy observation

Configuration G0 runs: `bash deploy.sh --input-type zmq_manager sim` (POSE mode, streamed motion) with the
default observation config `policy/release/observation_config.yaml`
(`gear_sonic_deploy/deploy.sh:241` default, passed through at `:582`), streamer
`pico_manager_thread_server.py --manager`.

This is a code-reading record. Each hop is marked **verified by reading** (the code was read and the
behaviour follows from it) or **not verified** (not traceable by reading, or only inferred). It does not
change `robot_side`; it records what `chest_residual` spans.

## Headline finding (differs from the plan's assumption)

The plan says "the policy consumes `smpl_pose` directly". With this configuration it does **not**:

- The encoder mode selected for streamed motion is `smpl` (mode 2), which requires only
  `encoder_mode_4`, `smpl_joints_10frame_step1`, `smpl_anchor_orientation_10frame_step1` and
  `motion_joint_positions_wrists_10frame_step1` (`policy/release/observation_config.yaml`, `encoder_modes`).
- `smpl_pose*` observations exist in the registry (`g1_deploy_onnx_ref.cpp:1773-1776`) but none is enabled in
  `encoder_observations` or `observations`, so the `smpl_pose` that the deploy decodes and stores is never gathered
  into the policy input.
- Chest twist therefore reaches the policy only through (a) the root-local SMPL joint **positions**
  `smpl_joints` and (b) the root orientation (`body_quat_w`) via the anchor-orientation observation. G0's
  `ref_spine_twist` is computed from `smpl_pose` and is a streamer-side proxy for the spine rotation, not a
  quantity the policy reads. The BC calibration (`twist_chest`) is what tests whether the proxy tracks the measured waist.
  **Verified by reading** for the config file and the gather registry; **not verified** for what the ONNX encoder
  does with its inputs (the model is opaque to code reading).

## Hops

| # | Hop | file:line | Selected by | Transformations of spine / root rotations | Status |
|---|---|---|---|---|---|
| 1 | XRT body sample (24 joints x [x,y,z,qx,qy,qz,qw]) stamped `time.monotonic()` at poll | `pico_manager_thread_server.py:840-870` (stamp at `:856`, `:863`) | `--manager`, XRoboToolkit backend (default) | none (raw device sample) | verified by reading |
| 2 | Raw sample to SMPL (`smpl_pose` axis-angle 21x3, `global_orient`, joints) | `pico_manager_thread_server.py:586` (`process_smpl_joints`, defined `:450-487`) | streamer code path | root: axis-angle to quat, `smpl_root_ytoz_up` (90 deg about X, `gear_sonic/isaac_utils/rotations.py:712-719`), then `remove_smpl_base_rot` (conjugates the SMPL rest orientation [0.5,0.5,0.5,0.5], `rotations.py:704-708`) -> `global_orient_quat` (= streamed `body_quat_w`, wxyz). `smpl_joints_local` = joints rotated by the inverse root quat (so root yaw is removed from the joint positions). `smpl_pose` body pose is passed through unchanged | verified by reading (the body of the SMPL solver that produced `smpl_pose`/`global_orient` from the device poses is the model-based step; **not verified**) |
| 3 | Resample to `target_fps` (default 50) between two device samples | `pico_manager_thread_server.py:1377-1416` (`_interp_pose_axis_angle` `:784`, `_quat_lerp_normalized` `:769`, `use_joints` `:1409`) | `--target_fps` (default 50, `:2170`) | `smpl_joints`: linear; `smpl_pose`: axis-angle interpolation; `body_quat_w`: normalized quaternion lerp. Output frame lags the sample by up to 1/target_fps | verified by reading |
| 4 | Frame buffer and message | `pico_manager_thread_server.py:1490-1531`, packed at `:1541` (`pack_pose_message`, default protocol v3, `gear_sonic/utils/teleop/zmq/zmq_planner_sender.py:161`) | `--num_frames_to_send` (default 5, `:2168`) | none; wire fields `smpl_pose`, `smpl_joints`, `body_quat_w`, `joint_pos` (zeros except wrists, `:1419-1476`), `timestamp_monotonic` = the sample stamp of hop 1 | verified by reading |
| 5 | Deploy field decode | `zmq_endpoint_interface.hpp:648-670` (names), `:1063-1112` (body quat), `:1114-1170` (smpl_joints), `:1173+` (smpl_pose) | pose topic `pose` on 5556 via `ZMQManager` (`zmq_manager.hpp:103`, STREAMED_MOTION when `planner == false`, `:242`) | copy only. `body_quat_w`/`body_quat` accepted as the root quaternion; stored index order [0..3] and used as w,x,y,z downstream (the decode comment at `:1065` says xyzw; code copies indices unchanged) | verified by reading; quaternion order **not verified at runtime** (comment vs use mismatch) |
| 6 | Heading increment | `zmq_endpoint_interface.hpp:1483-1512` | `heading_increment` field (joystick yaw) | adds to `heading_state.delta_heading` | verified by reading |
| 7 | Merge into streamed motion | `streamed_motion_merger.hpp:485-510` | `StreamedMotionMerger::MergeIncomingData` | sliding window by `frame_index`; copies `body_quat`, `smpl_joints`, `smpl_pose` per frame. No filtering or interpolation in the merger or decoder (grep for interpol/filter/smooth/blend matches comments only) | verified by reading |
| 8 | Encoder mode | `zmq_endpoint_interface.hpp:1705-1710` | protocol v3 -> `SetEncodeMode(2)` (`smpl`); `zmq_manager.hpp:655-663` only switches to modes 0/1 for VR 3-point | mode 2 requires the four observations listed above | verified by reading; the streamer's protocol is the default v3 (`zmq_planner_sender.py:161`) |
| 9 | Heading state | `g1_deploy_onnx_ref.cpp:559-591` (`UpdateHeadingState`), `:593-606` (`ComputeApplyDeltaHeading`) | reinitialize flags (`zmq_endpoint_interface.hpp` `reinitialize_heading` sites), frame 0 | `apply_delta_heading = delta_quat(yaw) * init_heading(robot base_quat) * inv(heading(init_ref_data_root_rot))` | verified by reading |
| 10 | `smpl_anchor_orientation_10frame_step1` | registry `:1779`; `GatherMotionAnchorOrientationMutiFrame` `:616-695` (mode 0) | encoder mode 2 requirement | root: `new_ref_root_rot = apply_delta_heading * body_quat_w[frame]`; observation = first two columns of `R(conj(robot base_quat) * new_ref_root_rot)`, 6 values x 10 frames, `target_frame = current_frame_ + i` (step 1) | verified by reading |
| 11 | `smpl_joints_10frame_step1` | registry `:1769`; `GatherMotionSmplJointsMultiFrame` `:910-965` | encoder mode 2 requirement | none: all 24 root-local joint positions, 10 future frames, step 1, copied as x,y,z. **No spine rotation is read; chest twist is only present as the geometry of these positions** | verified by reading |
| 12 | `motion_joint_positions_wrists_10frame_step1` | registry `:1759`; `GatherMotionJointPositionsMultiFrame` `:740`; indices `policy_parameters.hpp:88` ({23..28}, wrists) | encoder mode 2 requirement | reads the streamed `joint_pos` wrist entries only (zeros elsewhere are not read) | verified by reading |
| 13 | Encoder input assembly | `g1_deploy_onnx_ref.cpp:1862-1895` (active functions in `encoder_observations` order), `:2025-2110` (mode filter) | `observation_config.yaml` | concatenation in config order; unused observations zero-filled | verified by reading |
| 14 | Policy ONNX/TensorRT encoder + decoder | `policy/release/model_encoder.onnx`, `model_decoder.onnx` (symlinks) | `deploy.sh` `--checkpoint` default | unknown (learned model) | **not verified** |
| 15 | Deploy debug targets (`g1_debug`) | `output_interface.hpp:150-260`, publish `zmq_output_handler.hpp` | always | `base_quat_target = apply_delta_heading * body_quat_w`; `body_q_target` = streamed `joint_pos` (zeros except wrists) | verified by reading |

## What `chest_residual` spans

The BC reference is `ref_chest_twist_joints`: the yaw (about the root-local up axis, +z) of the shoulder line
(SMPL joint 16 -> 17) relative to the hip line (1 -> 2), from the streamed root-local `smpl_joints`. This is the
quantity the policy can see (hop 11), unlike `smpl_pose`. The root-local frame is z-up because `smpl_joints_local`
is the z-up joints rotated by the inverse of the z-up root quat with the SMPL base rotation removed
(`pico_manager_thread_server.py:466-479`, hops 2); **verified by reading**, not empirically.
`ref_spine_twist` (from `smpl_pose`) is kept as a diagnostic with its own calibration (`BC_spine_diag`) and does not feed `robot_side`.

BC compares `ref_chest_twist_joints` (hops 2-4, 11) with `meas_waist_yaw`. In between are the anchor orientation
(hop 10) and the learned encoder/decoder (hop 14), which is not verifiable by reading, so a `chest_residual` result is an
end-to-end residual and says nothing about *why* the policy moved the waist. The shoulder/hip-line measure assumes a
bend about the lateral axis (it is unchanged by a forward bend); other bends are not verified.

## Not verified (summary)

- The SMPL fitting that turns device poses into `smpl_pose` / `global_orient` (hop 2, model-based).
- Quaternion component order on the deploy side at runtime (hop 5, comment vs code).
- Learned encoder/decoder behaviour (hop 14).
- The coordinate frame and handedness of the raw motion-tracker quaternion used by `operator_ref` (`g0_twist_log.py`):
  `g0_analyze.py` assumes the twist about +y of the tracker quaternion (xyzw); the `turn_body` / `twist_chest`
  calibration decides whether that mapping is valid (hop A `unmapped` otherwise).
