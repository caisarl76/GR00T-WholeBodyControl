# Camera configuration review — 2026-10-19

The reviewed RealSense path does **not** use the D435i RGB camera's full horizontal field of view. **30 FPS is not the hardware-wide maximum**, and the saved launch command actually selects a **15 FPS capture default**. The connected camera currently negotiates **USB 2**, which restricts the available resolutions and rates.

## Scope and evidence

- Local checkout: `gear_sonic/camera/composed_camera.py`, `gear_sonic/camera/drivers/realsense.py`, `gear_sonic/camera/sensor_server.py`, installer, service template, and `camera_command.txt`.
- Documented camera host: `unitree@192.168.0.223`, hostname `unitree-g1-nx`. The host-side main checkout has the same 640×480 / 15 FPS defaults. The saved command uses `/home/unitree/GR00T-WholeBodyControl-xr-video-bridge`; its RealSense driver also defaults to 640×480 / 15 FPS.
- Camera: RealSense D435I, serial `254322071593`, firmware `5.15.1.55`.
- Read-only SDK enumeration with `/home/unitree/GR00T-WholeBodyControl/.venv_camera/bin/python`: `query_devices()`, `get_stream_profiles()`, `get_intrinsics()`, and `rs2_fov()`; no pipeline was started and no camera settings were written.
- SDK `usb_type_descriptor` is `2.1`. `lsusb` identifies `8086:0b3a` on Bus 001 Device 003; `lsusb -t` and `/sys/bus/usb/devices/1-3/speed` confirm **480 Mb/s**. A separate USB3 hub is visible, but the camera is on the USB2 branch. This identifies the negotiated link; it does not identify whether the cable, connector, hub, or port causes the fallback.
- No camera publisher process or listening camera port was present; `composed_camera_server.service` was inactive/uninstalled and there was no tmux server. Therefore there is no active runtime FPS measurement in this review.

Official specifications and their caveats are collected in [d435i-hardware.md](d435i-hardware.md).

## Findings

### 1. VGA RGB crops the horizontal view

`RealSenseConfig.color_image_dim` is fixed to `(640, 480)` at `gear_sonic/camera/drivers/realsense.py:34`. `enable_stream()` uses these dimensions directly at line 79. The composed-camera CLI forwards FPS and depth enablement but does not expose image dimensions (`composed_camera.py:381`).

The connected device's calibration gives the following RGB profiles. Values below are SDK-advertised rates, not measured throughput; FOV is calculated from calibrated intrinsics, not measured using a scene target.

| RGB output | Advertised FPS on the current USB2 link | Calibrated horizontal × vertical FOV |
|---|---|---|
| 640×480 — server default | 6, 15, 30 | 55.621° × 43.135° |
| 424×240 | 6, 15, 30, 60 | 69.899° × 43.135° |
| 1280×720 | 6, 10, 15 | 70.238° × 43.135° |
| 1920×1080 | 8 | 70.238° × 43.136° |

The default loses about **14.6° of horizontal coverage** relative to the 1280×720 profile. Upscaling the VGA output cannot recover that coverage. This matches [RealSense's engineering confirmation that VGA RGB is cropped](https://github.com/realsenseai/librealsense/issues/7170).

No additional crop or resize occurs in the reviewed RealSense capture → composed publisher → JPEG serialization path. The crop already exists in the selected sensor profile.

### 2. Capture FPS and publisher FPS are separate

- `ComposedCameraConfig.fps = 30` (`composed_camera.py:79`) controls the publisher loop (`:482`).
- `ComposedCameraConfig.realsense_fps = 15` (`:103`) is passed to `RealSenseConfig.fps` (`:386`), which controls the actual sensor stream (`realsense.py:84`).
- The saved command in `camera_command.txt` supplies neither FPS option. With the reviewed defaults it requests **640×480 RGB at 15 FPS**, depth disabled, with a publisher scheduled at 30 Hz.

Setting only `--fps 60` does not increase sensor capture. `--realsense-fps` must also select a supported mode. At 640×480 on the current USB2 link, the highest enumerated RGB rate is 30 FPS; `--realsense-fps 60` is unsupported at those dimensions.

The client's `fps()` helper (`composed_camera.py:581`) uses the duration of a `read()` call, so its value is not evidence of sensor cadence. Measure distinct hardware frame numbers/timestamps or arrivals of new messages over elapsed time when validating a future configuration.

### 3. Wider and faster profiles require a resolution choice and, for useful higher resolution, USB3

On the **current USB2 connection**, the device advertises:

- Wide RGB at **1280×720 up to 15 FPS**.
- Approximately the same wide RGB coverage at **424×240 up to 60 FPS**.
- Cropped RGB at **640×480 up to 30 FPS**.

On **USB3**, the official D435/D435i table includes RGB **1280×720 at 30 FPS**, RGB **960×540 at 60 FPS**, depth **1280×720 at 30 FPS**, and depth **848×480 at 90 FPS**. Those are individual profile capabilities; exact FOV and stable simultaneous RGB-D throughput must be verified on the connected device after USB3 negotiation. See [datasheet Table 4-2, printed p.81](https://www.realsenseai.com/wp-content/uploads/2023/10/Intel-RealSense-D400-Series-Datasheet-September-2023.pdf#page=81).

### 4. Depth coverage is distinct from RGB coverage

Depth is disabled by default. When enabled in the current main implementation, `rs.align(rs.stream.color)` (`realsense.py:96`) maps depth into the narrower color image and publishes only that aligned result. Keeping a native depth image separately is necessary if the application needs depth outside the RGB view. The saved XR worktree's driver differs: it reads native depth without the main implementation's alignment step.

Selected native-depth calibrations and advertised rates on this USB2 link:

| Native depth output | Advertised FPS | Calibrated horizontal × vertical FOV |
|---|---|---|
| 640×480 — depth default | 6, 15, 30 | 78.699° × 63.177° |
| 640×360 | 30 | 89.072° × 57.923° |
| 848×480 | 6, 8, 10 | 89.072° × 58.228° |
| 480×270 | 6, 15, 30, 60 | 89.072° × 57.924° |
| 1280×720 | 6 | 89.072° × 57.923° |

These profiles trade horizontal and vertical coverage. The wider horizontal 16:9 modes do not preserve the VGA profile's full vertical coverage. Individual color/depth profile lists do not establish that every simultaneous combination works.

## Recommended changes for a follow-up

1. Restore a **USB3 connection** and confirm the camera itself reports a SuperSpeed link, rather than only checking the hub.
2. Expose RGB and depth dimensions in the composed-camera configuration. For RGB, select **1280×720 at 30 FPS** for resolution or evaluate **960×540 at 60 FPS** for cadence and wide coverage.
3. Set both capture and publisher FPS to the intended rate. If RGB and depth need different rates, the existing shared RealSense FPS setting must be split.
4. If depth outside RGB coverage is needed, retain native depth and its own intrinsics alongside any aligned depth image.
5. Validate selected intrinsics/FOV, sustained unique sensor frames, simultaneous streams, and delivered ZMQ rate. The connected RGB sensor currently reports auto exposure enabled and `auto_exposure_priority=0`; exposure-driven FPS reduction was not enabled at audit time.

This review changed documentation only. It did not alter source, services, USB configuration, or camera settings. USB3 streaming and sustained end-to-end FPS remain unmeasured.
