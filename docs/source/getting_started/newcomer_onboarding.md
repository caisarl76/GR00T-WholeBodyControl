# Newcomer Onboarding: PC2 + Workstation Data Collection

This command-first page uses the lab topology in which **PC2** runs the C++
GEAR-SONIC deployment and camera server, while the laptop/workstation runs the
PICO manager, data exporter, and camera viewer. It overrides conflicting
placement of both C++ deployment and the camera server in the existing [Data
Collection](../tutorials/data_collection.md) tutorial. That tutorial remains
authoritative only where it does not conflict with this PC2/workstation
topology.

## Runtime Topology

```text
Laptop/workstation                              PC2
------------------                              ---
PICO manager          -- planner commands -->   GEAR-SONIC deploy
<WORKSTATION_IP>:5556                           state publisher :5557

PICO manager          <-- measured state -----  GEAR-SONIC deploy
feedback=<PC2_IP>:5557

PICO manager          -- pose stream ------->   Data exporter
localhost:5556 (workstation-local)

Data exporter         <-- robot state -------   GEAR-SONIC deploy
                      <-- camera frames ------   camera server :<CAMERA_PORT>

Camera viewer         <-- camera frames ------   camera server :<CAMERA_PORT>
```

`localhost:5556` is the workstation-local PICO-manager-to-exporter pose stream:
the exporter subscribes to the manager. `localhost` is correct only for this
manager-to-exporter traffic on the workstation. Cross-machine manager,
feedback, state, and camera endpoints must use the configured machine IPs.

## Before You Begin

This is a hard reader contract for the real-robot sections. Repository cloning
and non-actuating setup or reading may proceed before every gate below is
complete, but do not launch or actuate the real robot until they all pass.
Complete the [MuJoCo Quick Start](quickstart.md) with the intended input
sequence, drill the stop paths in simulation from [Real Robot Teleoperation
Safety](../user_guide/real_robot_safety.md), and complete [VR Teleop
Setup](vr_teleop_setup.md), including XRoboToolkit, networking, and calibration.

You must also have authorized G1 access; a clear 3 m zone; a protective
harness/frame; and named assignments for the robot owner, VR operator, spotter,
and safety operator.

```{danger}
The robot owner must supply `<LAB_APPROVED_HARDWARE_ESTOP_PROCEDURE>` with the
exact mechanism, location, and actions. This independent hardware-stop or
physical power-cut procedure must not depend on PC2, the deploy process or
input thread, ZMQ, the workstation, the network, or the Unitree wireless
remote. Rehearse it with the robot supported, and assign a dedicated safety
operator to keep it continuously available from before launch until actuation
stops.

Unitree remote damping combinations are not an E-stop during low-level deploy.
If the procedure is unknown, untrained, or unavailable, stop and do not run the
real-robot sections. Loss of power makes the robot dead weight, so the
harness/frame must support it.
```

## Configuration

Replace every angle-bracket value below with the lab's configuration. These are
intentional configuration tokens, not unfinished documentation.

| Shell token | Description |
| --- | --- |
| `<PC2_IP>` | PC2's reachable IP address for deployment state and camera traffic. |
| `<PC2_USER>` | SSH user authorized to access PC2. |
| `<PC2_REPO_DIR>` | Absolute checkout path on PC2. |
| `<WORKSTATION_IP>` | Workstation address reachable from PC2 for PICO-manager planner commands. |
| `<WORKSTATION_REPO_DIR>` | Absolute checkout path on the workstation. |
| `<ROBOT_NETWORK_INTERFACE>` | PC2 network interface connected to the robot. |
| `<REPO_REVISION>` | The same explicit 40-character Git commit to check out on both machines. |
| `<TENSORRT_ROOT>` | PC2 TensorRT installation root used by the C++ deployment build. |
| `<EGO_CAMERA_TYPE>` | Camera backend: only `oak`, `oak_mono`, or `realsense`. |
| `<EGO_CAMERA_DEVICE_ID>` | Camera device identifier; use `''` when the selected backend does not need one. |
| `<CAMERA_PORT>` | TCP port exposed by the PC2 camera server. |
| `<TASK_PROMPT>` | Task prompt recorded with the collected data. |
| `<DATASET_NAME>` | Destination dataset name for the data exporter. |

`<LAB_APPROVED_HARDWARE_ESTOP_PROCEDURE>` is a non-shell human checklist: the
robot owner supplies the exact mechanism, location, and actions. Never execute
it as a shell command.

**Workstation — every new workstation terminal, any working directory**

```bash
export PC2_IP='<PC2_IP>'
export PC2_USER='<PC2_USER>'
export PC2_REPO_DIR='<PC2_REPO_DIR>'
export WORKSTATION_IP='<WORKSTATION_IP>'
export WORKSTATION_REPO_DIR='<WORKSTATION_REPO_DIR>'
export REPO_REVISION='<REPO_REVISION>'
export CAMERA_PORT='<CAMERA_PORT>'
export TASK_PROMPT='<TASK_PROMPT>'
export DATASET_NAME='<DATASET_NAME>'
```

**PC2 — every new PC2 terminal, any working directory**

```bash
export PC2_IP='<PC2_IP>'
export PC2_REPO_DIR='<PC2_REPO_DIR>'
export WORKSTATION_IP='<WORKSTATION_IP>'
export REPO_REVISION='<REPO_REVISION>'
export ROBOT_NETWORK_INTERFACE='<ROBOT_NETWORK_INTERFACE>'
export TENSORRT_ROOT='<TENSORRT_ROOT>'
export EGO_CAMERA_TYPE='<EGO_CAMERA_TYPE>'
export EGO_CAMERA_DEVICE_ID='<EGO_CAMERA_DEVICE_ID>'
export CAMERA_PORT='<CAMERA_PORT>'
readonly GEAR_SONIC_HF_REV='9c0ff22b4ffec27c5392e8e284eb2f2df7a5b4e2'
```

The quotes make the placeholders safe to paste without accidental shell
redirection, but you must replace them before use. The hardware-stop value is a
human checklist, never a command.

## 0. Clone the GR00T Repository

On **both machines — any working directory**, install the prerequisites first:

```bash
set -euo pipefail
sudo apt-get update
sudo apt-get install -y git git-lfs iproute2 netcat-openbsd
git lfs install
```

**Workstation — parent directory of `$WORKSTATION_REPO_DIR`**

```bash
set -euo pipefail
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git "$WORKSTATION_REPO_DIR"
cd "$WORKSTATION_REPO_DIR"
git fetch origin "$REPO_REVISION"
git checkout --detach "$REPO_REVISION"
test "$(git rev-parse HEAD)" = "$REPO_REVISION"
git submodule update --init --recursive
git lfs pull
git rev-parse HEAD
```

**PC2 — parent directory of `$PC2_REPO_DIR`**

```bash
set -euo pipefail
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git "$PC2_REPO_DIR"
cd "$PC2_REPO_DIR"
git fetch origin "$REPO_REVISION"
git checkout --detach "$REPO_REVISION"
test "$(git rev-parse HEAD)" = "$REPO_REVISION"
git submodule update --init --recursive
git lfs pull
git rev-parse HEAD
```

The package and clone commands must exit 0. Each checkout explicitly fails
unless `git rev-parse HEAD` exactly equals `$REPO_REVISION`; the final
`git rev-parse HEAD` visibly prints the resulting 40-character commit. Both
final outputs must be the same 40-character commit; stop on a mismatch because
the Python and C++ components share a ZMQ wire format. `git lfs pull` does not
download ignored deployment ONNX files; Section 1 handles those files.
