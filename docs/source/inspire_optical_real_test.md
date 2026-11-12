# Real G1 + Inspire optical qualification

Target: Inspire PC2 `192.168.0.222` (`pc2_222`), workstation `192.168.0.62`.
The user confirmed bilateral optical fingers and A+X POSE/PLANNER switching in
MuJoCo. Intermittent left source articulation recovered without an established
root cause; that remains a qualification item.

## Deployment decision: retain the original SONIC

Use the existing executable:

`/home/unitree/GR00T-WholeBodyControl/gear_sonic_deploy/target/release/g1_deploy_onnx_ref`

The checkout is `087f9ac`; the previously inspected executable SHA256 is
`8d4d1228e456a85682317de4dbce93f1642978c061d660493e084264a221a9f6`.
The user verified that both source and executable lack the custom one-second
teleoperation motor-output ramp. That ramp originated in fork commit `7877831`
and remains in the different, older installation
`/home/unitree/gear_sonic_inspire_deploy_dcf5e72`. Do not conflate those paths.

The ramp blended all 29 motor targets from measured posture to policy output
while action history retained the full policy requests. Attenuating leg balance
corrections this way was a plausible contributor to transient balance loss,
not a proven explanation for every switching issue.

**No SONIC rebuild is required solely for ramp removal, and the selected real
workflow does not deploy the local adapted handoff changes.** Those changes
were tested locally but are not active or physically qualified on PC2. Any
remaining switch instability in the original deployment needs separate diagnosis.
Do not add a one-second switch delay or motor-output blend. Preserve the
separate standing-startup smoothing and Inspire measured initialization.

Keep the user's existing SONIC v1.1 launch command, model/config selection and
planner. Artifact size differences do not by themselves establish model version
or a need to replace working models. The previously prepared C++/model
qualification archive is superseded for this workflow and must not be used as
an instruction to replace SONIC.

## Optical bridge package — user operated

The selected payload contains seven Python files for native optical q6 input,
DDS publishing and physical status. It contains no SONIC executable, model,
CUDA, TensorRT or dependency installer. It runs in the existing `inspire_ws`
environment. The driver alone cannot translate PICO commands: a publishing
Inspire bridge must run alongside SONIC and the installed headless driver.

Bundle: `/tmp/gr00t-inspire-optical-integration/outputs/pc2-inspire-optical-20260921/pc2-inspire-optical-20260921.tar.gz`

SHA256: `9d75e23d27b7577729cb00958d1a906709d2e3422e0ccb5067836adae09c4505`

The September 21 package was copied to PC2 .222 and its checksums and imports
were verified. The following commands document that installation; do not
overwrite an active installation. Keep the previous bridge as a rollback option.

Workstation:

```bash
scp /tmp/gr00t-inspire-optical-integration/outputs/pc2-inspire-optical-20260921/pc2-inspire-optical-20260921.tar.gz \
  pc2_222:/home/unitree/pc2-inspire-optical-20260921.tar.gz
```

PC2:

```bash
export QUAL=/home/unitree/inspire-optical-bridge-20260921
mkdir "$QUAL"
tar -xzf /home/unitree/pc2-inspire-optical-20260921.tar.gz -C "$QUAL"
cd "$QUAL"
sha256sum -c SHA256SUMS
```

There is no CMake/build step for this Python bridge.

## Monitor hand feedback first

Keep real SONIC control stopped for this check. Use the already working driver
in `inspire_ws`; do not start a second driver. Stop the controller bridge
`run_inspire_ftp_pc2_bridge.py` before publishing with the optical bridge.
The optical bridge owns both DDS command topics, and the existing headless
driver remains the sole Modbus owner. The state proxy on 5558 does not replace
optical status on 5563.

PC2 bridge terminal:

```bash
source /home/unitree/miniconda3/etc/profile.d/conda.sh
conda activate inspire_ws
export QUAL=/home/unitree/inspire-optical-bridge-20260921
cd "$QUAL"
export PYTHONPATH="$PWD"
python -c 'import numpy, zmq, cyclonedds, inspire_sdkpy, unitree_sdk2py'
RUN=$(mktemp -d "$QUAL/monitor-XXXXXXXX")
python -u gear_sonic/scripts/run_pico_inspire_bridge.py \
  --host 192.168.0.62 --port 5556 --network-interface enP8p1s0 \
  --status-port 5563 --evidence-log "$RUN/bridge.jsonl"
```

Monitor-only creates no command writers. Check both measured hands update,
ages stay below 500 ms, errors are zero, and discovery identifies exactly one
state writer per hand. Missing source data is expected before the manager runs.
Finish monitor-only with Ctrl+C before starting the publishing bridge.

## First attached-hand and body test

Run only with the operator/spotter ready at the supported robot and the stop
procedure reviewed in [real robot safety](user_guide/real_robot_safety.md).
Keyboard `O` in the SONIC terminal is the immediate software stop; use the
established physical stop procedure if software does not respond. Stop local
MuJoCo/deployment/manager processes before connecting the real setup.

PC2 bridge terminal — this command moves hands during initialization:

```bash
RUN=$(mktemp -d "$QUAL/optical-XXXXXXXX")
python -u gear_sonic/scripts/run_pico_inspire_bridge.py \
  --host 192.168.0.62 --port 5556 --network-interface enP8p1s0 \
  --status-port 5563 --publish --initialize-hands \
  --evidence-log "$RUN/bridge.jsonl"
```

Wait for `hand_initialized` and READY. This first pass intentionally uses the
800–1000 count envelope and five-count steps per 100 ms, so finger movement is
small and slow. After confirming motor order, polarity, feedback and stops,
restart the bridge with `--full-position-range` for 0–1000 motion. Keep active
slew limiting for the first full-range test. `--hand-max-rate` alone cannot
raise the bridge's five-count/tick limit; `--no-active-slew-limit` is a separate,
later physical speed qualification, not part of these initial commands.

The workstation manager also accepts `--hand-max-rate 0.0` for Inspire. This
disables its rate cap during recovery and tracking while retaining 60 ms
smoothing and feedback/source checks. It does not change the PC2 bridge's
five-count/tick limit, position envelope, initialization, or fault response.
No PC2 installation or rebuild is required for this manager option.

PC2 SONIC terminal: run the user's established SONIC v1.1 command from
`/home/unitree/GR00T-WholeBodyControl/gear_sonic_deploy`, using the original
`./target/release/g1_deploy_onnx_ref`. Keep real CRC checks enabled and Dex3
actuation disabled for Inspire. Use `zmq_manager` with workstation
`192.168.0.62:5556` and the established body feedback endpoint on PC2 port 5557.
Do not add local-fork-only flags or switch to the older ramp-bearing deployment.
The guide intentionally does not replace the user's working model/launcher
arguments with unverified defaults.

Workstation manager terminal:

```bash
cd /tmp/gr00t-inspire-optical-integration
export PYTHONPATH="$PWD"
mkdir -p outputs
CAPTURE=$(mktemp -d "$PWD/outputs/inspire-real-XXXXXXXX")
set -o pipefail
/home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python -u \
  gear_sonic/scripts/pico_manager_thread_server.py --manager \
  --hand-input optical --hand-profile inspire_ftp --hand-max-rate 0.5 \
  --zmq_feedback_host 192.168.0.222 --inspire-status-host 192.168.0.222 \
  --hand-log-dir "$CAPTURE/hands" --record_dir "$CAPTURE/body" \
  2>&1 | tee "$CAPTURE/manager.log"
```

With XRoboToolkit body/hand tracking and Send enabled, start SONIC using
A+B+X+Y. Once stable, release all buttons/grips and hold A+X for at least 0.3 s.
Check both bare hands and every motor. Exercise POSE→PLANNER→POSE, held partial
closure, independent left/right occlusion, recovery and the stop path. If the
left fingers stop again, preserve raw capture and compare headset visualization
before restarting the service.

Inspire hand readiness does not gate A+X body-mode switching. Missing, stale,
or fault-latched hand feedback can leave fingers waiting while body POSE runs.
The hand runtime still requires admitted measured feedback, and the PC2 bridge
still rejects optical finger control while fault-latched. This separation does
not clear a hand fault or qualify finger tracking; verify body switching and
finger response separately. Restart the workstation manager to load this change;
no PC2 deployment rebuild or bridge update is required.

The user confirmed attached-hand operation alongside the original SONIC and
recorded full-range hand actions with PLANNER intervals. The latest inspected
episode contains turning, but no commanded translational locomotion. The
Left Grip+A recording timing fix is **needed to be tested** on the physical
controllers; see the remaining hardware TODO in the integration plan. Local
handoff test results do not independently qualify the original PC2 deployment.

## Full range and recording controls

For the tested full-range configuration, add `--full-position-range` and
`--no-active-slew-limit` to the publishing bridge command, and use
`--hand-max-rate 0` on the manager. Initialization/shutdown ramps and hardware
limits remain; manager smoothing also remains. These are explicit opt-in
settings.

Left Grip+A toggles recording; Left Grip+B aborts. Release all controls before
the first chord and between actions. Hold the complete chord for at least
0.3 seconds. Holding grip before A is supported by the updated manager. Wait
for the exporter acknowledgement; keyboard C starts and S stops/saves in the
manager terminal. The controller timing fix awaits physical verification.
