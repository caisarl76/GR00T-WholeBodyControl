# PICO optical fingers with full-body tracking

The host implementation supports Dex3 first and Inspire FTP next. Full-body
and arm targets continue to come from the PICO fused body skeleton; only
finger articulation uses optical landmarks. A missing optical side holds its
last bounded command while body and opposite-hand output continue.

## Source and installation prerequisites

The original XRoboToolkit APK is supported without a headset rebuild. Enable
`Tracking - Hand` and `Send` in the headset app. Disable `Switch w/ A Button`
so the A button used by manager chords does not also toggle the headset stream.
All 26 poses, joint tracking flags, radii and hand activity are required.
Zero radii are accepted as unavailable metadata: original-APK active captures
contain tracked finger positions with all radii zero. Negative or nonfinite
radii still reject; retargeting uses positions, not radii.

When per-hand timestamps are absent, the PC binding uses the host monotonic
time of each change in that side's pose or tracking metadata. Repeated identical
packets do not refresh it, even while the body or other hand moves. After
100 ms without a change the hand holds; five advancing valid samples allow
recovery. An exactly stationary hand can also hold until movement resumes.
This detects unchanged cached data, not sensor acquisition freshness. The
existing joint limits, rate limits, measured-feedback checks and independent
hand watchdogs still apply.

Dex3 startup holds use the robot's physical thumb-1 range (left upper/right
lower ±1.04719755 rad). The optimizer retains its narrower ±0.92 rad target
bound. This lets measured startup poses converge at the configured rate instead
of rejecting valid feedback or abruptly clipping the initial command.

For a straight, spread Dex3 thumb, the retargeter follows the measured distal
thumb direction in the anatomical hand frame. It adjusts thumb base rotation
and distal flexion within the audited joint limits, retaining the positional
solution for proximal flexion and the other fingers. A straight human thumb
cannot be represented by simply setting the Dex3 distal motor to zero: its
mechanical base orientation would leave the thumb pointing into the palm.
The correction fades out as the thumb bends or approaches the index/middle
fingertips, preserving the original DexPilot pinch behavior. This prioritizes
open-thumb direction over exact fingertip position; downstream speed limits
and feedback checks still apply.

The manager announces `original APK compatibility` for each admitted legacy
source. Snapshots/captures carry `timestamp_source`: 0 means device timestamp,
1 means host content-change time. Exported episodes also carry per-hand
`teleop.*_hand_timestamp_source`; -1 means unavailable, including older
diagnostics without this field. `source_timestamp_ns` is the effective clock
in that declared domain. A clock-domain change requires readmission. Captures
using host timing are not evidence of device-timestamp qualification.

The public XRoboToolkit Unity Client at commit
`cdc53166b0bf412efae71046c6a225eb5091605f` omits those two timestamps.
`external_dependencies/pico_optical_hand_source.patch` adds a timestamp only
when that side's `GetJointLocations` call succeeds. Cached sides retain their
old timestamp. This patch is optional for stronger freshness evidence. Apply it to a checkout of that exact client revision,
build its PICO APK using the upstream Unity instructions, and install it on
the headset for device-timestamp qualification. This patch timestamps successful
SDK reads; the native hand API does not expose a sensor acquisition timestamp.
It detects cached side objects and lost successful reads, not an undetectable
failure inside a native SDK that falsely reports successful fresh samples.

```bash
git -C /path/to/XRoboToolkit-Unity-Client apply --check \
  /path/to/GR00T-WholeBodyControl/external_dependencies/pico_optical_hand_source.patch
git -C /path/to/XRoboToolkit-Unity-Client apply \
  /path/to/GR00T-WholeBodyControl/external_dependencies/pico_optical_hand_source.patch
```

Rebuild the binding in
`external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64` into the
teleoperation environment using that directory's CMake/setuptools workflow.
The new SDK must expose atomic left/right hand and controller snapshots.
Install `gear_sonic[teleop]` into that environment; it pins official
`dex-retargeting==0.4.6`. Vendored YAML/URDF hashes and the upstream license
are checked before loading a retargeter.

Rebuild the C++ deployment executable as well: optical Dex3 arming requires
its new per-side DDS feedback age/validity fields. Fresh ZMQ messages with
old cached hand positions do not qualify. Both local message age and physical
Dex3 sample age must be below 100 ms.

The PICO application must keep full-body tracking enabled when controllers
are put down. Disable its optional **Switch w/ A Button** data-send toggle so
the recording/tracking chords do not inadvertently stop XR data transmission.

## Dex3 launch and operation

Run the manager, simulator and deploy launcher from this same worktree.
The Python entry points select their own checkout, even when a shared virtual
environment has an editable installation pointing at another checkout.

Start the normal deployment, camera server and exporter using the existing
repository workflows, with the rebuilt binding/deploy executable. Pass the
same hand profile to manager and exporter:

```bash
python gear_sonic/scripts/pico_manager_thread_server.py --manager \
  --hand-input optical --hand-profile dex3 --hand-max-rate 0.5
python gear_sonic/scripts/run_data_exporter.py --hand-profile dex3 \
  --task-prompt "optical hand qualification"
```

`--hand-max-rate 0.5` is the first Dex3 qualification rate, in rad/s. It limits
speed only; it does not restrict Dex3 travel to the Stage A 20% range. Enforce
that range in the qualification setup before attached-hand tests. Normal
production is 2.0 rad/s only after the staged checks in the design pass.
The initial hands must have fresh measured feedback; optical frames can become
valid after the controllers are put down. Five advancing valid hand samples
admit each side, followed by a bounded transition to live finger targets.
Do not pass `--disable-dex3-hands` to deployment when using optical Dex3:
that removes the hand state/command channels required for arming.

| Input | Action |
| --- | --- |
| PICO A+B+X+Y | Start or stop SONIC |
| PICO A+X or manager T | Toggle full-body tracking |
| PICO left grip+A | Toggle recording |
| Manager C | Start recording |
| Manager S | Stop and save recording |
| PICO left grip+B | Discard active recording |
| Deploy-terminal O | Independent SONIC stop |

The manager has no O stop key. Its Ctrl+C requests a recording save and waits
at most two seconds for IDLE, then exits without sending a SONIC-stop command.
A global controller stop remains immediate and discards active capture.

Wait for the recorder's RECORDING acknowledgement before putting controllers
down. Pick them up to stop/save, wait for durable IDLE, and repeat. Stop
tracking before hanging the robot on the crane and stopping SONIC. A normal
tracking exit is refused while recording or saving remains uncertain.

Manager controls and hand watchdogs run at 50 Hz even when the body source
has no new frame. Controller chords require 40 ms stable input, have a 200 ms
combination window, and require neutral release before repeating. The full
four-button chord suppresses the subset tracking/recording commands.

For A+X, release all buttons and both grips, then hold A and X together for
at least 0.3 seconds. The manager prints `Controller action: tracking` when
recognized. Holding the left grip can select recording instead, and a short
tap may finish before the combination window. Both controller snapshots must
be fresh.

Dex3 admission reads the DDS `left_hand_q` / `right_hand_q` fields from
`g1_debug`. The similarly named `*_hand_q_measured` visualization fields can
contain commanded positions and must not supply the startup baseline. Measured
Dex3 positions within 1e-4 rad of a joint limit are clamped to that limit.
For measured right `index_0` only, a lower excursion up to 0.001 rad is
admitted and projected to zero; raw capture retains the original measurement.
This narrowly scoped allowance addresses recorded near-zero feedback rejection
and does not widen commanded joint limits. Larger excursions still block admission. An arming refusal now prints the
specific transport, schema, DDS or joint-limit failure.

## Simulation test commands on this host

The startup fixes initialize DDS once and create command locks before any
subscriber callback. The shared `gear_sonic_teleop` environment now has the
rebuilt snapshot binding and official dex-retargeting 0.4.6; existing NumPy,
Pinocchio, SciPy and Torch versions were preserved. The worktree deployment
binary was rebuilt with fresh hand-feedback fields. Pinned local LFS payloads
were restored by SHA-256, and existing ONNX/TRT assets were copied locally.
These host checks do not establish that the headset APK emits side timestamps.

Close any older simulator/deployment before this test. Check
`lsof -nP -iTCP:5557 -sTCP:LISTEN`; the port must be free. Keep all processes
below in this checkout even though the Python environments are shared.
Only one simulator may publish on DDS domain 0; a free ZMQ port alone does
not exclude an old simulator. The temporary Inspire body-startup diagnostic
manager uses controller inputs, so it cannot test optical Dex3 finger motion.

Terminal 1 — simulator:

```bash
cd ~/work/GR00T-WholeBodyControl
source ~/work/GR00T-WholeBodyControl/.venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

Terminal 2 — deployment with Dex3 enabled:

```bash
cd ~/work/GR00T-WholeBodyControl/gear_sonic_deploy
./target/release/g1_deploy_onnx_ref lo \
  policy/release/model_decoder.onnx reference/example/ \
  --obs-config policy/release/observation_config.yaml \
  --encoder-file policy/release/model_encoder.onnx \
  --planner-file planner/target_vel/V2/planner_sonic.onnx \
  --input-type zmq_manager --output-type all --zmq-host localhost \
  --disable-crc-check --live-pose-playback
```

`--live-pose-playback` is opt-in and skips accumulated streamed history while
retaining the encoder's future observation window. It bounds extra playback
backlog; it does not eliminate the observation window's latency. Planner and
loaded-motion playback are unchanged. Keep Dex3 enabled for measured feedback
and finger commands.

Terminal 3 — optional recording test, using deliberately black camera frames:

```bash
cd ~/work/GR00T-WholeBodyControl
source ~/work/GR00T-WholeBodyControl/.venv_data_collection/bin/activate
python gear_sonic/scripts/run_data_exporter.py \
  --hand-profile dex3 --use-dummy-camera \
  --dataset-name "pico_hand_sim_$(date +%Y%m%d_%H%M%S)" \
  --task-prompt "simulation hand-tracking test"
```

This exporter checks recording controls without requiring a camera server;
its black images are not useful visual training data. Skip it for a pure
hand-motion test. Raw hand capture below does not require the exporter.

Terminal 4 — manager:

```bash
cd ~/work/GR00T-WholeBodyControl
source ~/work/GR00T-WholeBodyControl/.venv_teleop/bin/activate
PICO_CAPTURE_DIR="outputs/pico_hands_$(date +%Y%m%d_%H%M%S)"
python gear_sonic/scripts/pico_manager_thread_server.py --manager \
  --hand-input optical --hand-profile dex3 --hand-max-rate 0.5 \
  --hand-log-dir "$PICO_CAPTURE_DIR"
```

Keep the fused body stream active. Start SONIC with ABXY, release the buttons,
then enable tracking with AX or manager T. With recording enabled, press C
and wait for the exporter acknowledgement. Put the controllers down and
check thumb, index and middle motion; occlude one hand and verify it holds
while the other continues. Use S to save, wait for IDLE, T to exit tracking,
and O in the deployment terminal to stop SONIC. Ctrl+C in the manager flushes
raw capture; it does not stop SONIC.

After the manager exits, replay in the same terminal:

```bash
python -m gear_sonic.scripts.replay_pico_hands --profile dex3 \
  --max-rate 0.5 --input "$PICO_CAPTURE_DIR"/capture_*.npz \
  --output-dir "${PICO_CAPTURE_DIR}_replay"
cat "${PICO_CAPTURE_DIR}_replay/report.json"
```

Require valid optical samples and zero emitted bound/rate/hold violations;
inspect `reason_codes`, `counts`, and transition events when a hand remains
WAITING/HOLDING. `d0_qualified=false` is expected until the complete real-source
qualification and live-host timing evidence have been reviewed.

## Inspire FTP

### Local MuJoCo optical test

Stop the previous Dex3 simulator, deployment and manager first. Run all three
processes from this checkout, using the shared Python environments. PC2 and the
physical Inspire bridge are not part of this test.

Terminal 1:

```bash
cd ~/work/GR00T-WholeBodyControl
source ~/work/GR00T-WholeBodyControl/.venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py --hand-profile inspire_ftp \
  --hand-command-source optical
```

Terminal 2:

```bash
cd ~/work/GR00T-WholeBodyControl/gear_sonic_deploy
./deploy.sh --input-type zmq_manager --disable-dex3-hands --live-pose-playback sim
```

Terminal 3:

```bash
cd ~/work/GR00T-WholeBodyControl
source ~/work/GR00T-WholeBodyControl/.venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager \
  --hand-input optical --hand-profile inspire_ftp --hand-max-rate 1.0 \
  --inspire-status-host 127.0.0.1 \
  --hand-log-dir "outputs/pico_inspire_sim_$(date +%Y%m%d_%H%M%S)"
```

Enable Hand and Send in XRoboToolkit. Start SONIC with A+B+X+Y, release all
buttons and grips, then hold A+X for about 0.5 seconds. Put the controllers down.
Check little, ring, middle, index, thumb bend and thumb rotation on both hands.
Inspire rate is a normalized fraction of travel per second (maximum 1.0),
not the Dex3 rad/s setting. MuJoCo supplies measured six-motor feedback locally
on port 5563; the optical manager supplies direct six-motor targets on 5556.
This verifies the simulated mechanism, not physical Inspire driver timing.
The simulator defaults to `--hand-command-source controller` for the existing
controller-driven Inspire path. Select `optical` explicitly for this workflow;
only one subscriber owns the simulated hand commands.

### Physical bridge

The manager sends direct six-motor normalized commands, in order little,
ring, middle, index, thumb bend, thumb rotation. It never projects them through
Dex3's three fingers. PC2 retains the sole DDS command writer; the existing
headless driver retains sole Modbus ownership.

```bash
python gear_sonic/scripts/pico_manager_thread_server.py --manager \
  --hand-input optical --hand-profile inspire_ftp --inspire-status-host PC2_IP
python gear_sonic/scripts/run_data_exporter.py --hand-profile inspire_ftp \
  --inspire-status-host PC2_IP --task-prompt "optical Inspire qualification"
python gear_sonic/scripts/run_pico_inspire_bridge.py --help
```

The PC2 bridge defaults to monitor-only and requires the existing PC2 DDS SDK
and driver environment. Publishing retains the reviewed Stage A envelope
800–1000 counts and maximum 5 counts per 100 ms tick. Later hardware ranges
remain separate qualification stages. Use its CLI help for existing DDS
interface/domain/source options. Monitor-only feedback is not a recordable
applied action. Recording requires fresh ACTIVE feedback and a successfully
applied command sequence under the current manager provenance.

Ports: manager commands/state 5556, robot feedback 5557, local exporter
recording status 5562, PC2 Inspire physical status 5563. Exporter/manager profile
mismatches prevent recording. Dex3 episodes have 43 joints; Inspire episodes
have 41. Manager intent, physical observation and actually applied Inspire
commands remain separate dataset fields. Missing data drops frames instead of
creating zero hand commands.

## Verification and qualification

Software tests live in `gear_sonic/tests/test_pico_*.py`. Binding tests compile
a test-only extension with SDK stubs; they never initialize XR or robot hardware.
The retarget tests require the pinned real dependency. Run pytest with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` in the matching environment to avoid
unrelated ROS pytest plugins.

If deployment reports `Address already in use`, inspect the feedback publisher
before starting another process:

```bash
lsof -nP -iTCP:5557 -sTCP:LISTEN
```

Stop the previous deployment in its own terminal if replacing that session.
The launcher now detects this conflict before building. Changing only the ZMQ
port does not isolate two simulations that share the same DDS domain/topics.
The `lo is not multicast-capable` message itself is informational; the simulator
uses loopback for local operation. DDS initialization now occurs once, and a
real initialization failure stops startup instead of continuing into callbacks.

To capture the exact manager inputs and outputs, add
`--hand-log-dir outputs/pico_capture_run1` to the optical manager command. Use a
new directory for each run. It writes bounded asynchronous `capture_*.npz`
shards and a `transitions.jsonl` event log. Each tick includes both atomic
snapshots, body wrists, measured positions, emitted commands, rejection/state
codes and publication provenance. Missing measurements remain unavailable.
A disk/queue error disables logging and prints the error while motion continues.

Replay those shards in an environment with the pinned retargeter:

```bash
python -m gear_sonic.scripts.replay_pico_hands --profile dex3 \
  --input outputs/pico_capture_run1/capture_*.npz \
  --output-dir outputs/pico_replay_run1
```

Use `--profile inspire_ftp` for its six-motor captures. The result contains
`replay.npz`, `transitions.jsonl` and `report.json`. Automatic checks cover
source loss, recovery and bounded commands; absent pose labels or real live
evidence remain explicit gaps. A synthetic fixture cannot establish D0.
If capture used a reduced rate, pass the same `--max-rate` to replay so the
reproduced commands use the original limiter setting.

Replay, simulation and attached-hand results must be recorded against the
measurable gates in
`docs/superpowers/specs/2026-09-04-pico-optical-hand-tracking-fullbody-design.md`.
Passing software tests does not mark those gates complete. Collect a real
PICO source fixture and exercise unilateral/bilateral occlusion, controller
pickup, source restarts and full recording cycles before hardware rollout.

## Coordinate provenance

Raw joint 0 is palm and joint 1 is wrist; retarget input is `raw[1:26,:3]`.
The source sends native OpenXR hand coordinates without the Unity rendering
reflection. Retargeting derives anatomical axes from wrist/index/middle
landmarks so moving the wrist in space does not change finger articulation.
Body positions also reach the wire in the native frame: the PICO body SDK's
Z reflection is undone by `TrackingData.GetBodyTracking`. Alignment compares
optical wrist 1 and body wrists 20/21 before robot-coordinate transforms.

Sources: [pinned client serializer](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/blob/cdc53166b0bf412efae71046c6a225eb5091605f/Assets/Scripts/TrackingData.cs),
[pinned hand rendering conversion](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/blob/cdc53166b0bf412efae71046c6a225eb5091605f/PICO%20Unity%20Integration%20SDK/Runtime/Scripts/Features/PXR_HandTracking.cs),
[pinned native SDK wrapper](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/blob/cdc53166b0bf412efae71046c6a225eb5091605f/PICO%20Unity%20Integration%20SDK/Runtime/Scripts/PXR_Plugin.cs).

### Inspect PICO input and recorded Dex3 actions

The read-only browser viewer displays PICO finger landmarks/flags, the manager's
recorded target, the command after smoothing/slew limiting, and measured robot
joints. Robot skeletons use forward kinematics of those exact recorded joints;
the viewer does not rerun retargeting or connect to XR/DDS/control sockets.

```bash
cd ~/work/GR00T-WholeBodyControl
source ~/work/GR00T-WholeBodyControl/.venv_teleop/bin/activate
python gear_sonic/scripts/view_pico_hands.py --capture-dir "$CAPTURE/hands" --port 8766
```

Open http://127.0.0.1:8766. Choose a capture, scrub/play frames, and drag the
canvases to rotate the anatomical view. Orange is target, blue is command, green
is measured. The raw table lists native PICO joint names, positions in meters,
quaternions (xyzw), and original flags. The joint action table uses radians.

`Follow latest` reads newly completed shards from the selected directory;
250-frame logging normally adds about five seconds of buffering. It is not
instantaneous live telemetry. Restart the viewer with the new directory when
the manager starts a new capture. Missing PICO geometry/feedback stays missing;
held targets can still be visible, so inspect active/state/reason/sent together.
"Unchanged sample" measures time since a timestamp change observed within this
shard, not device acquisition freshness. The first age is unknown. The viewer
currently supports Dex3 only.


## Remaining work and verification scope

See [post-merge evidence and qualification steps](pico_hand_followups.md) for
the resolved audit items, capture measurements, and current handoffs.

User-operated MuJoCo tests confirmed Dex3 and Inspire finger movement. Real
G1 + Dex3 tests on PC2 with SONIC v1.1 confirmed both hands, pinch, wrist
rotation, and open/close at an increased `--hand-max-rate`. These observations
are functional checks, not completion of the design's formal D0/D1 and
attached-hand qualification gates.

- **Open-thumb mapping remains unresolved and is deferred to a separate
  session.** The direction correction described above is implemented, but the
  user still observed an inward thumb with spread fingers. Its regression
  tests do not establish that the reported visual mismatch is fixed.
- **Intermittent left articulation loss recovered without a confirmed root
  cause.** During one captured failure, raw accepted left landmarks remained
  nearly straight while the headset visualization reportedly bent correctly.
  Check raw landmarks, flags, target, command and measured joints together;
  do not infer working finger data from body FPS or hand activity alone.
- **Near-limit feedback rejection:** offline diagnosis identified right
  `index_0` excursions below zero. A measurement-only allowance is under
  validation; physical zero characterization and real confirmation remain
  pending. See the linked evidence record.
- **Full fist closure is not gesture calibration.** DexPilot matches landmark
  geometry within robot joint limits. `--hand-max-rate` changes speed, not the
  final pose. Full-travel fist calibration remains separate work.
- Complete labelled source-loss/restart replay, sustained bilateral operation,
  controller pickup, recovery and durable recording checks. Physical Inspire
  qualification is separate from the confirmed MuJoCo Inspire result.

Keep follow-up results with the capture identifiers and exact revisions. The
read-only viewer above can compare source articulation with robot actions
without commanding a robot.
