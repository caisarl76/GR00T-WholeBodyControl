# Inspire FTP Hands in GEAR-SONIC MuJoCo

This runbook exercises the two Inspire FTP hands entirely in simulation. The
SONIC policy still controls only the G1's 29 body DoFs; normalized six-motor
hand commands bypass SONIC and are consumed by the Python MuJoCo adapter.

## Safety boundary

This phase does **not** SSH to PC2, import the Inspire FTP SDK, publish
`rt/inspire_hand/ctrl/l`, or publish `rt/inspire_hand/ctrl/r`. It also launches
the C++ process with Dex3 DDS disabled. Do not substitute `real` for `sim` in
these commands.

The normalized hand order is:

```text
[pinky, ring, middle, index, thumb_bend, thumb_rotation]
```

`1.0` is open and `0.0` is closed. PICO trigger closes the first five motors;
PICO grip independently closes thumb rotation. Both `pose` and `planner`
messages must contain `left_hand_joints` and `right_hand_joints`, each with
shape `(6,)`.

## Automated headless verification

Run from the repository root with no viewer, PICO, DDS, or hardware:

```bash
/home/jihun/work/GR00T-WholeBodyControl/.venv_sim/bin/python \
  gear_sonic/scripts/build_inspire_ftp_mujoco_model.py --check

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python \
  -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_inspire_ftp_mujoco.py \
  gear_sonic/tests/test_inspire_ftp_zmq.py \
  gear_sonic/tests/test_inspire_ftp_bridge.py \
  gear_sonic/tests/test_inspire_ftp_launch.py
```

Run the isolated twelve-motor sweep. It disables contact only to measure
actuator isolation and settled mimic kinematics:

```bash
PYTHONPATH="$PWD" \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python \
  gear_sonic/scripts/verify_inspire_ftp_mujoco.py \
  --motor-sweep --duration 2 \
  --json-output /tmp/inspire_ftp_mujoco_report.json
```

Run the separate contact-enabled open/close/open cycle:

```bash
PYTHONPATH="$PWD" \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python \
  gear_sonic/scripts/verify_inspire_ftp_mujoco.py \
  --contact-cycle --duration 15 --open-hold 5 \
  --json-output /tmp/inspire_ftp_contact_cycle.json
```

The verified model has 54 joints, 41 actuators, 12 equality constraints, and
two palm/thumb contact exclusions. A successful report has twelve passing
motors, finite state, mimic error below `2e-3` rad, and joint-limit error no
greater than `1e-6` rad.

## Full headless SONIC regression

This path needs loopback CycloneDDS and the built TensorRT deployment binary.
It never opens a viewer. Use three terminals.

Terminal 1 — MuJoCo with the Inspire profile:

```bash
PYTHONPATH="$PWD" \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_sim/bin/python \
  gear_sonic/scripts/run_sim_loop.py \
  --hand-profile inspire_ftp \
  --no-enable-onscreen --no-enable-offscreen
```

Terminal 2 — SONIC body process with no Dex3 DDS endpoints:

```bash
cd gear_sonic_deploy
./deploy.sh --disable-dex3-hands sim
```

Confirm the printed final command contains both `--disable-crc-check` and
`--disable-dex3-hands`. The shell forwarding can be checked without building
or launching:

```bash
cd gear_sonic_deploy
./deploy.sh --dry-run --disable-dex3-hands sim
```

Terminal 3 — bounded local hand/planner publisher:

```bash
PYTHONPATH="$PWD" \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python \
  gear_sonic/scripts/verify_inspire_ftp_mujoco.py \
  --scripted-publisher --duration 15 --open-hold 5 \
  --json-output /tmp/inspire_ftp_scripted_publisher.json
```

Expected C++ evidence is `Dex3 hands disabled`, `Init Done`, a transition to
`CONTROL`, periodic finite loop timing, and a normal exit after the scripted
stop. Stop the simulator with `Ctrl-C` only after the publisher has returned
the hands to open.

## PICO-to-MuJoCo launch

After the automated checks pass, the all-in-one launcher propagates the same
profile to MuJoCo, PICO, and C++:

```bash
/home/jihun/work/GR00T-WholeBodyControl/.venv_data_collection/bin/python \
  gear_sonic/scripts/launch_data_collection.py \
  --sim --hand-profile inspire_ftp
```

This command depends on live XRoboToolkit/PICO state and may open configured
visual components. Use it only when the headset and trackers are ready. The
launcher intentionally rejects `hand_profile=inspire_ftp` without `--sim` in
this phase.

## Watchdog behavior

Before the first valid hand pair, both hands command open. A valid pair remains
active for 250 ms. When messages become stale, each motor ramps toward `1.0`
using the configured normalized rate rather than jumping. Invalid, partial,
non-finite, or out-of-range pairs are rejected atomically. To shut down safely,
stop the PICO/scripted source, allow the stale watchdog enough time to open the
hands, then stop MuJoCo.

## Hardware promotion gate

PC2 and the real FTP DDS services remain a separate integration step. Before
enabling them, capture live PICO trigger/grip ranges, verify the same six-motor
order, implement a PC2-side fail-open bridge that is not a second Modbus owner,
and perform the first attached-hand test at low rate with stationary G1 arms.
