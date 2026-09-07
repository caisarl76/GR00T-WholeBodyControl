# G1 ACT → SONIC v1.1 validation

The confirmed simulation run executes a published 28D ACT model through native
SONIC v1.1 and MuJoCo G1 with two Dex3 hands. It uses recorded observations from
the first episode of Unitree's ToastedBread dataset. The result validates action
transport, standing startup, and measured arm/hand movement.

The follow-up [joint and palm comparison](../../reports/act-sonic-joint-pose-2026-09-07/README.md)
includes all 28 joint errors, hand position/orientation plots, the numerical
source evidence, and Ubuntu reproduction commands. It confirms exact reference
transport but does not establish identical ACT and measured actions. Raw ACT
versus measured palm error is 1.98 cm / 6.50° RMS on the left and
3.84 cm / 8.47° RMS on the right.

![Confirmed G1 and both Dex3 hands](../_static/vla_adapter/confirmed-action.png)

## Startup and reference execution

The simulator invokes keyboard **9 → Backspace**, disables the support band,
and resets the physical state before learned control. Native SONIC first runs a
nominal standing reference. ACT begins after 0.5 seconds of measured settling
with both feet contacting, low joint/base velocity, bounded waist angles, and an
upright base. The lower-body and root references remain nominal; arms and hands
enter from their measured grounded state.

Six 100-action predictions come from recorded observation frames 0, 100, 200,
300, 400, and 500. Each input contains four synchronized camera images and 28
joint states. The composer applies named joint mapping, 30→50 Hz resampling,
limits, and slew caps. It clips 641 of 16,800 values, adds a two-second entry and
three-second terminal settle, and pads 46 paired reference rows for native preview.

| Measurement | Confirmed result |
|---|---|
| Original ACT output | 600×28 absolute joint targets, 20 seconds |
| Native path | SONIC v1.1 encoder mode 0 → 64D token → decoder |
| Reference playback | 1,399 active ticks; all 1,249 distinct frames executed |
| Body/hand pairing | All selected references agree with paired source rows within 1e-6 |
| Invalid input handling | Duplicate, malformed, and expired chunks did not execute |
| Expiry | Last applied body/hand reference held; 67 ticks checked in replay summary |
| Reset → standing / ACT | 77.83 ms / 1.538 s |
| Observed duration after ACT start | 52.80 simulated seconds |
| Falls / automatic resets | 0 / 0 |
| Both feet contacting | 100% of recorded samples after ACT start |
| Minimum base height / maximum tilt | 0.7603 m / 3.510° |
| Maximum waist yaw / roll / pitch | 5.091° / 2.019° / 2.289° |
| Left/right arm tracking RMS | 0.1188 / 0.1263 rad |
| Left/right hand tracking RMS | 0.0565 / 0.1024 rad |

All seven joints in each arm and each hand moved. During seconds 10–20, after
the entry transition, right shoulder pitch traversed 1.025 rad, left elbow
0.578 rad, left index finger 1.005 rad, and right middle finger 1.672 rad.

The confirmed source and runtime identities, detailed joint excursions, source
hashes, and video hashes are in [validation.json](../_static/vla_adapter/validation.json).
The full native trace contains additional hold ticks after the replay summary.
The 30.37-second startup/action video and 22-second action video passed full
ffmpeg decoding, frame counts, codec, resolution, and 30 fps checks. The screenshot
above comes from the encoded action video. Raw videos, logs, checkpoints, and
recorded observations are run artifacts, rather than repository dependencies.

## Limits and reproduction

The ACT checkpoint is language-free. It predicts from recorded dataset images,
not live simulation cameras. The empty MuJoCo scene does not reproduce the
bread/toaster interaction, and the simulation does not directly replay recorded
ground-truth actions. These results establish neither manipulation-task success
nor general balance guarantees. Native reference leases use a shared monotonic
kernel clock, so the tested processes run on the same Linux Docker host.

Use the [Docker runbook](vla_adapter_h100.md) for pinned downloads and a fresh
replay. Model loading, native execution, physics, rendering, and native tests
were performed inside H100 Docker containers. The runtime-focused Python suite
contains 37 tests; native validation contains three adapter tests and the
existing Dex3 decoder test. Neither physical robot execution nor host software
installation is part of this workflow.
