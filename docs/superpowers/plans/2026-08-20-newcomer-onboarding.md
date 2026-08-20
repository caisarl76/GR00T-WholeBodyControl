# Newcomer Onboarding Runbook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a command-first, safety-gated onboarding runbook for the lab's PC2/workstation GEAR-SONIC data-collection topology.

**Architecture:** One MyST Markdown page owns the ordered two-machine procedure and links existing detailed setup and safety guides. A separate checksum manifest pins the four deployment artifacts, while the existing Sphinx index provides discoverability. The implementation changes documentation only; validation builds the docs and performs offline content/checksum checks without starting robot, PICO, or camera hardware.

**Tech Stack:** MyST Markdown, Sphinx, Bash command examples, Python 3.10 probes, ZeroMQ, Hugging Face CLI, SHA-256

---

## File Map

- Create `docs/source/getting_started/newcomer_onboarding.md`: the complete newcomer runbook, topology, commands, safety gates, probes, and shutdown procedure.
- Create `docs/source/getting_started/gear_sonic_deployment_9c0ff22.sha256`: the immutable four-file deployment checksum manifest.
- Modify `docs/source/index.rst`: add the runbook to the Getting Started toctree.

The angle-bracket strings in this plan are intentional user-facing configuration tokens required by the approved design. They are not missing implementation details. Do not edit runtime Python/C++ code and do not start physical hardware while executing this plan.

### Task 1: Add the Pinned Deployment Checksum Manifest

**Files:**
- Create: `docs/source/getting_started/gear_sonic_deployment_9c0ff22.sha256`

- [ ] **Step 1: Verify the manifest does not already exist**

Run:

```bash
test ! -e docs/source/getting_started/gear_sonic_deployment_9c0ff22.sha256
```

Expected: exit status 0. If the file exists, inspect it and reconcile it with the approved design before continuing.

- [ ] **Step 2: Create the manifest with the approved hashes**

Create `docs/source/getting_started/gear_sonic_deployment_9c0ff22.sha256` with exactly:

```text
013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3  gear_sonic_deploy/policy/release/model_encoder.onnx
c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed  gear_sonic_deploy/policy/release/model_decoder.onnx
466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b  gear_sonic_deploy/policy/release/observation_config.yaml
39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea  gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx
```

- [ ] **Step 3: Validate the tracked manifest structure**

Run:

```bash
python - <<'PY'
from pathlib import Path
import re

path = Path("docs/source/getting_started/gear_sonic_deployment_9c0ff22.sha256")
lines = path.read_text().splitlines()
expected_paths = [
    "gear_sonic_deploy/policy/release/model_encoder.onnx",
    "gear_sonic_deploy/policy/release/model_decoder.onnx",
    "gear_sonic_deploy/policy/release/observation_config.yaml",
    "gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx",
]
assert len(lines) == 4, f"expected 4 entries, found {len(lines)}"
actual_paths = []
for line in lines:
    match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
    assert match, f"malformed checksum line: {line!r}"
    actual_paths.append(match.group(2))
assert actual_paths == expected_paths, actual_paths
print("PASS: checksum manifest structure")
PY
```

Expected: `PASS: checksum manifest structure`. The artifact-content check runs after the runbook's pinned download commands are documented and again in final verification.

- [ ] **Step 4: Commit the manifest**

```bash
git add docs/source/getting_started/gear_sonic_deployment_9c0ff22.sha256
git commit -m "docs: pin GEAR-SONIC deployment artifacts"
```

### Task 2: Create the Runbook Foundation, Safety Contract, and Clone Workflow

**Files:**
- Create: `docs/source/getting_started/newcomer_onboarding.md`
- Reference: `docs/superpowers/specs/2026-08-19-newcomer-onboarding-design.md`
- Reference: `docs/source/user_guide/real_robot_safety.md`
- Reference: `docs/source/getting_started/vr_teleop_setup.md`

- [ ] **Step 1: Verify the user-facing page does not already exist**

Run:

```bash
test ! -e docs/source/getting_started/newcomer_onboarding.md
```

Expected: exit status 0.

- [ ] **Step 2: Create the page title, scope, and topology**

Start the page with the title `# Newcomer Onboarding: PC2 + Workstation Data Collection`. State that this page overrides the existing data-collection tutorial wherever that tutorial places the C++ deployment on the workstation.

Add this topology verbatim:

```text
Laptop/workstation                              PC2
------------------                              ---
PICO manager          -- planner commands -->   GEAR-SONIC deploy
<WORKSTATION_IP>:5556                           state publisher :5557

PICO manager          <-- measured state -----  GEAR-SONIC deploy
feedback=<PC2_IP>:5557

Data exporter         <-- robot state -------   GEAR-SONIC deploy
                      <-- camera frames ------   camera server :<CAMERA_PORT>

Camera viewer         <-- camera frames ------   camera server :<CAMERA_PORT>
```

Explain that `localhost` is correct only for exporter-to-manager traffic on the workstation. Cross-machine manager, feedback, state, and camera endpoints must use the configured machine IPs.

- [ ] **Step 3: Add the hard reader contract and independent-stop gate**

Link these prerequisites:

- `[MuJoCo Quick Start](quickstart.md)` completed with the intended input sequence.
- `[Real Robot Teleoperation Safety](../user_guide/real_robot_safety.md)` stop paths drilled in simulation.
- `[VR Teleop Setup](vr_teleop_setup.md)` completed, including XRoboToolkit, networking, and calibration.
- Authorized G1 access, a clear 3 m zone, protective harness/frame, robot owner, VR operator, spotter, and safety operator assigned.

Add a danger admonition with these requirements:

1. `<LAB_APPROVED_HARDWARE_ESTOP_PROCEDURE>` is supplied by the robot owner with the exact mechanism, location, and actions.
2. The mechanism is independent of PC2, the deploy process, its input thread, ZMQ, workstation, network, and Unitree wireless remote.
3. The safety operator rehearses it with the robot supported and keeps it continuously available from before launch until actuation stops.
4. Unitree remote damping combinations are not an E-stop during low-level deploy.
5. If the procedure is unknown, untrained, or unavailable, stop here and do not run the real-robot sections.

- [ ] **Step 4: Add the configuration table and copyable shell blocks**

The table must define these shell tokens:

```text
<PC2_IP>
<PC2_USER>
<PC2_REPO_DIR>
<WORKSTATION_IP>
<WORKSTATION_REPO_DIR>
<ROBOT_NETWORK_INTERFACE>
<REPO_REVISION>
<TENSORRT_ROOT>
<EGO_CAMERA_TYPE>
<EGO_CAMERA_DEVICE_ID>
<CAMERA_PORT>
<TASK_PROMPT>
<DATASET_NAME>
```

Define `<LAB_APPROVED_HARDWARE_ESTOP_PROCEDURE>` separately as a non-shell checklist value. State that only `oak`, `oak_mono`, and `realsense` are accepted for `<EGO_CAMERA_TYPE>`, and that an unused device ID is entered as an empty string.

Add this workstation block and tell readers to paste it into every new workstation terminal:

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

Add this PC2 block and tell readers to paste it into every new PC2 terminal:

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

Warn that quoted placeholders prevent accidental shell redirection but must still be replaced before execution.

- [ ] **Step 5: Add Section 0 with separate clone commands for both machines**

Use the heading `## 0. Clone the GR00T Repository`. Label every block with the machine and working directory.

On both machines, install clone and diagnostic prerequisites:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs iproute2 netcat-openbsd
git lfs install
```

Workstation clone:

```bash
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git "$WORKSTATION_REPO_DIR"
cd "$WORKSTATION_REPO_DIR"
git fetch origin "$REPO_REVISION"
git checkout --detach "$REPO_REVISION"
git submodule update --init --recursive
git lfs pull
git rev-parse HEAD
```

PC2 clone:

```bash
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git "$PC2_REPO_DIR"
cd "$PC2_REPO_DIR"
git fetch origin "$REPO_REVISION"
git checkout --detach "$REPO_REVISION"
git submodule update --init --recursive
git lfs pull
git rev-parse HEAD
```

Expected: both `git rev-parse HEAD` commands print the same 40-character commit. Stop on a mismatch because Python and C++ share a ZMQ wire format.

- [ ] **Step 6: Check the foundation and commit it**

Run:

```bash
rg -n '^## 0\. Clone|LAB_APPROVED_HARDWARE_ESTOP_PROCEDURE|Runtime Topology|REPO_REVISION' docs/source/getting_started/newcomer_onboarding.md
git diff --check -- docs/source/getting_started/newcomer_onboarding.md
```

Expected: all four required concepts are found and the whitespace check exits 0.

Commit:

```bash
git add docs/source/getting_started/newcomer_onboarding.md
git commit -m "docs: add newcomer onboarding foundation"
```

### Task 3: Document Environment Installation and Immutable Deployment Inputs

**Files:**
- Modify: `docs/source/getting_started/newcomer_onboarding.md`
- Reference: `install_scripts/install_pico.sh`
- Reference: `install_scripts/install_data_collection.sh`
- Reference: `install_scripts/install_camera_server.sh`
- Reference: `docs/source/getting_started/installation_deploy.md`
- Reference: `download_from_hf.py`
- Reference: `gear_sonic_deploy/reference/convert_motions.py`

- [ ] **Step 1: Add the separate-environment warning**

Use the heading `## 1. Install the Required Environments`. State that the three install scripts delete and recreate their target environments, so they must not be rerun casually when local packages or changes matter. Do not invoke the repository-wide `check_environment.py` because its default checks unrelated training, Isaac Lab, CUDA, and TensorRT requirements.

- [ ] **Step 2: Add workstation environment commands and focused checks**

From `$WORKSTATION_REPO_DIR`:

```bash
cd "$WORKSTATION_REPO_DIR"
bash install_scripts/install_pico.sh
bash install_scripts/install_data_collection.sh

.venv_teleop/bin/python --version
.venv_teleop/bin/python -c 'import msgpack, numpy, zmq; print("PASS: teleop imports")'
.venv_teleop/bin/python gear_sonic/scripts/pico_manager_thread_server.py --help

.venv_data_collection/bin/python --version
.venv_data_collection/bin/python -c 'import cv2, lerobot, msgpack_numpy, zmq; print("PASS: data imports")'
.venv_data_collection/bin/python gear_sonic/scripts/run_data_exporter.py --help
.venv_data_collection/bin/python gear_sonic/scripts/run_camera_viewer.py --help
```

Expected: both Python commands report 3.10, both import commands print `PASS`, and every help command exits 0.

- [ ] **Step 3: Add the PC2 camera environment commands**

From `$PC2_REPO_DIR`:

```bash
cd "$PC2_REPO_DIR"
bash install_scripts/install_camera_server.sh
```

Tell the reader to answer `n` when prompted to install the systemd service. Then run:

```bash
.venv_camera/bin/python --version
.venv_camera/bin/python -c 'import depthai, msgpack_numpy, zmq; print("PASS: camera imports")'
.venv_camera/bin/python -m gear_sonic.camera.composed_camera --help
```

Expected: Python 3.10, the import check prints `PASS`, and help exits 0.

- [ ] **Step 4: Add PC2 platform and TensorRT gates before native build**

Document this exact gate:

```bash
cd "$PC2_REPO_DIR/gear_sonic_deploy"
ARCH="$(uname -m)"
printf 'PC2 architecture: %s\n' "$ARCH"
test -d "$TensorRT_ROOT"
test -x "$TensorRT_ROOT/bin/trtexec"
TRT_VERSION="$($TensorRT_ROOT/bin/trtexec --version 2>&1)"
printf '%s\n' "$TRT_VERSION"

case "$ARCH" in
  x86_64)
    printf '%s\n' "$TRT_VERSION" | grep -Eq 'TensorRT.*10\.13([. ]|$)'
    ;;
  aarch64)
    grep -q 'R36' /etc/nv_tegra_release
    if dpkg-query -W -f='${Version}\n' nvidia-jetpack 2>/dev/null; then
      dpkg-query -W -f='${Version}\n' nvidia-jetpack | grep -Eq '^6([.+~-]|$)'
    fi
    printf '%s\n' "$TRT_VERSION" | grep -Eq 'TensorRT.*10\.7([. ]|$)'
    ;;
  *)
    printf 'Unsupported PC2 architecture: %s\n' "$ARCH" >&2
    exit 1
    ;;
esac
```

Expected: every command exits 0. x86_64 must report TensorRT 10.13; aarch64 must report L4T R36.x, JetPack 6 when its metapackage is installed, and TensorRT 10.7. Explain that a version mismatch is a hard stop because incorrect TensorRT versions can produce unsafe planner inference.

- [ ] **Step 5: Add native dependency and build commands**

From `$PC2_REPO_DIR/gear_sonic_deploy`:

```bash
bash scripts/install_deps.sh
source scripts/setup_env.sh
just build
test -x target/release/g1_deploy_onnx_ref
```

Expected: `setup_env.sh` prints `TensorRT environment configured`, `just build` exits 0, and the executable check exits 0.

- [ ] **Step 6: Add immutable model, config, planner, and reference-motion preparation**

From `$PC2_REPO_DIR`, install the CLI into the existing camera environment:

```bash
uv pip install --python .venv_camera/bin/python huggingface_hub
```

Download the four deploy files from the immutable revision:

```bash
.venv_camera/bin/hf download nvidia/GEAR-SONIC model_encoder.onnx \
  --revision "$GEAR_SONIC_HF_REV" \
  --local-dir gear_sonic_deploy/policy/release
.venv_camera/bin/hf download nvidia/GEAR-SONIC model_decoder.onnx \
  --revision "$GEAR_SONIC_HF_REV" \
  --local-dir gear_sonic_deploy/policy/release
.venv_camera/bin/hf download nvidia/GEAR-SONIC observation_config.yaml \
  --revision "$GEAR_SONIC_HF_REV" \
  --local-dir gear_sonic_deploy/policy/release
.venv_camera/bin/hf download nvidia/GEAR-SONIC planner_sonic.onnx \
  --revision "$GEAR_SONIC_HF_REV" \
  --local-dir gear_sonic_deploy/planner/target_vel/V2
```

Do not use `download_from_hf.py`; it resolves mutable `main` and does not accept a Hub revision.

Prepare one immutable example reference motion from the same revision:

```bash
.venv_camera/bin/hf download nvidia/GEAR-SONIC \
  sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl \
  --revision "$GEAR_SONIC_HF_REV" \
  --local-dir .hf_reference_source

.venv_camera/bin/python gear_sonic_deploy/reference/convert_motions.py \
  .hf_reference_source/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl \
  gear_sonic_deploy/reference/example
```

Explain that a fresh clone does not populate `reference/example/`; the conversion creates the CSV directory structure consumed by the C++ reader.

- [ ] **Step 7: Add exact artifact and reference checks**

From `$PC2_REPO_DIR`:

```bash
test -s gear_sonic_deploy/policy/release/model_encoder.onnx
test -s gear_sonic_deploy/policy/release/model_decoder.onnx
test -s gear_sonic_deploy/policy/release/observation_config.yaml
test -s gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx
sha256sum --check docs/source/getting_started/gear_sonic_deployment_9c0ff22.sha256
find gear_sonic_deploy/reference/example -type f -name joint_pos.csv -size +0c -print -quit | grep -q .
```

Expected: all four hashes report `OK`, every file check exits 0, and at least one non-empty `joint_pos.csv` exists below `reference/example/`.

- [ ] **Step 8: Check the installation section and commit it**

Run:

```bash
rg -n '^## 1\.|TensorRT.*10\.13|TensorRT.*10\.7|9c0ff22b4ffec27c5392e8e284eb2f2df7a5b4e2|convert_motions.py|sha256sum --check' docs/source/getting_started/newcomer_onboarding.md
git diff --check -- docs/source/getting_started/newcomer_onboarding.md
```

Expected: each gate is present and the whitespace check exits 0.

Commit:

```bash
git add docs/source/getting_started/newcomer_onboarding.md
git commit -m "docs: document newcomer installation workflow"
```

### Task 4: Document PC2 Deployment, Blank PICO Section, and Foreground Camera

**Files:**
- Modify: `docs/source/getting_started/newcomer_onboarding.md`
- Reference: `gear_sonic_deploy/scripts/preflight.sh`
- Reference: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`
- Reference: `gear_sonic/camera/composed_camera.py`
- Reference: `install_scripts/install_camera_server.sh`

- [ ] **Step 1: Add the PC2 deployment heading and no-hands hardware gate**

Use `## 2. Deploy GEAR-SONIC on PC2`. State that this lab robot is documented without Dex3/Inspire hands, but the operator must physically confirm that configuration before continuing. Explain that `deploy.sh` cannot forward `--disable-dex3-hands`, so this runbook uses the validated direct invocation.

- [ ] **Step 2: Add the independent-stop and actuated-initialization warning immediately before launch**

Require the reader to fill and verbally confirm `<LAB_APPROVED_HARDWARE_ESTOP_PROCEDURE>` with the robot owner. Then add this danger text immediately before the deployment command:

```text
Pressing Enter starts an actuated initialization immediately. The deploy process drives all joints toward the default standing pose over three seconds with nonzero gains while low-level commands are published at 500 Hz. PICO engagement starts policy CONTROL, but it is not the first robot motion. Before pressing Enter, the protective harness/frame, clear 3 m zone, spotter, and independent hardware E-stop or physical power-cut procedure must already be ready.
```

- [ ] **Step 3: Add preflight and the direct no-Dex3 command**

From `$PC2_REPO_DIR/gear_sonic_deploy`:

```bash
cd "$PC2_REPO_DIR/gear_sonic_deploy"
source scripts/setup_env.sh
bash scripts/preflight.sh
```

Expected: every prompt is confirmed and the script prints `Real-robot preflight confirmed.`

Then launch in the same focused terminal:

```bash
just run g1_deploy_onnx_ref \
  "$ROBOT_NETWORK_INTERFACE" \
  policy/release/model_decoder.onnx \
  reference/example/ \
  --obs-config policy/release/observation_config.yaml \
  --encoder-file policy/release/model_encoder.onnx \
  --planner-file planner/target_vel/V2/planner_sonic.onnx \
  --input-type zmq_manager \
  --output-type zmq \
  --zmq-host "$WORKSTATION_IP" \
  --disable-dex3-hands
```

Expected startup evidence:

- input type is `zmq_manager`;
- output type is ZMQ and port 5557 binds;
- the configured workstation host is shown;
- `[INFO] Dex3 hands disabled` appears;
- transient LowState-unavailable messages stop;
- `Init Done` appears, proving transition to `WAIT_FOR_CONTROL`;
- no CRC or safety error appears.

State explicitly that policy `CONTROL` is not engaged until the later PICO start command, but the three-second initialization ramp already actuated the robot.

- [ ] **Step 4: Add the intentionally empty PICO section**

Add `## 3. Set Up PICO Teleoperation` followed immediately by `## 4. Run the Camera Server on PC2`, with no body content between the headings. The hard prerequisite near the top of the page already links `vr_teleop_setup.md`.

- [ ] **Step 5: Add the foreground-only camera gate and systemd collision check**

In Section 4, explain that Section 1 already created `.venv_camera` and the reader must not rerun the destructive installer. Use a new PC2 terminal and check:

```bash
cd "$PC2_REPO_DIR"
if systemctl is-active --quiet composed_camera_server.service; then
  echo 'ERROR: composed_camera_server.service is already active; do not start a foreground server.' >&2
  exit 1
fi
echo 'PASS: no active camera systemd service'
```

Expected: the command prints the `PASS` line. If it exits 1, stop and resolve which camera startup method is authoritative.

- [ ] **Step 6: Add type-specific camera discovery commands**

For `oak` or `oak_mono`:

```bash
.venv_camera/bin/python - "$EGO_CAMERA_DEVICE_ID" <<'PY'
import sys
import depthai as dai

expected = sys.argv[1]
devices = dai.Device.getAllAvailableDevices()
ids = []
for device in devices:
    getter = getattr(device, "getMxId", None)
    ids.append(getter() if callable(getter) else str(getattr(device, "mxid", device)))
print("Detected OAK IDs:", ids)
if not ids:
    raise SystemExit("FAIL: no OAK camera detected")
if expected and expected not in ids:
    raise SystemExit(f"FAIL: configured OAK ID {expected!r} not found")
print("PASS: OAK camera detected")
PY
```

For `realsense`, first install the separate driver and require an explicit serial:

```bash
uv pip install --python .venv_camera/bin/python pyrealsense2
.venv_camera/bin/python - "$EGO_CAMERA_DEVICE_ID" <<'PY'
import sys
import pyrealsense2 as rs

expected = sys.argv[1]
if not expected:
    raise SystemExit("FAIL: set EGO_CAMERA_DEVICE_ID to a RealSense serial")
serials = [
    device.get_info(rs.camera_info.serial_number)
    for device in rs.context().query_devices()
]
print("Detected RealSense serials:", serials)
if expected not in serials:
    raise SystemExit(f"FAIL: configured RealSense serial {expected!r} not found")
print("PASS: RealSense camera detected")
PY
```

Reject every other camera type:

```bash
case "$EGO_CAMERA_TYPE" in
  oak|oak_mono|realsense) ;;
  *) echo "Unsupported EGO_CAMERA_TYPE: $EGO_CAMERA_TYPE" >&2; exit 1 ;;
esac
```

- [ ] **Step 7: Add the foreground camera command with optional device-ID omission**

Use a Bash array so an empty device ID does not become an empty CLI argument:

```bash
CAMERA_DEVICE_ARGS=()
if [[ -n "$EGO_CAMERA_DEVICE_ID" ]]; then
  CAMERA_DEVICE_ARGS=(--ego-view-device-id "$EGO_CAMERA_DEVICE_ID")
fi

.venv_camera/bin/python -m gear_sonic.camera.composed_camera \
  --ego-view-camera "$EGO_CAMERA_TYPE" \
  "${CAMERA_DEVICE_ARGS[@]}" \
  --port "$CAMERA_PORT"
```

Expected: the camera initializes, the server binds the configured port, frames are published, and no frame-timeout/reconnect error repeats. Keep this terminal running.

- [ ] **Step 8: Check Sections 2-4 and commit them**

Run:

```bash
rg -n '^## [234]\.|500 Hz|--disable-dex3-hands|Init Done|composed_camera_server.service|pyrealsense2|CAMERA_DEVICE_ARGS' docs/source/getting_started/newcomer_onboarding.md
git diff --check -- docs/source/getting_started/newcomer_onboarding.md
```

Expected: all safety/runtime markers are found and the whitespace check exits 0.

Commit:

```bash
git add docs/source/getting_started/newcomer_onboarding.md
git commit -m "docs: document PC2 runtime startup"
```

### Task 5: Document Bidirectional Checks, Engagement, Workstation Processes, and Shutdown

**Files:**
- Modify: `docs/source/getting_started/newcomer_onboarding.md`
- Reference: `gear_sonic/scripts/pico_manager_thread_server.py`
- Reference: `gear_sonic/scripts/run_data_exporter.py`
- Reference: `gear_sonic/scripts/run_camera_viewer.py`
- Reference: `gear_sonic/utils/data_collection/zmq_state_subscriber.py`

- [ ] **Step 1: Add Section 5 and start only the PICO manager**

Use `## 5. Run Workstation Processes`. In workstation Terminal 1:

```bash
cd "$WORKSTATION_REPO_DIR"
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py \
  --manager \
  --port 5556 \
  --zmq_feedback_host "$PC2_IP" \
  --zmq_feedback_port 5557
```

Expected: the manager starts in interactive manager mode and listens on port 5556. Do not start the exporter or viewer yet.

- [ ] **Step 2: Add exact pre-engagement directional port checks**

Workstation Terminal 2:

```bash
ss -ltnp | grep ':5556'
nc -zvw 3 "$PC2_IP" 5557
nc -zvw 3 "$PC2_IP" "$CAMERA_PORT"
```

PC2 Terminal 2:

```bash
ss -ltnp | grep ':5557'
ss -ltnp | grep ":$CAMERA_PORT"
nc -zvw 3 "$WORKSTATION_IP" 5556
```

Expected: the manager `ss` line shows `*:5556` or `0.0.0.0:5556`, the deploy line shows `*:5557` or `0.0.0.0:5557`, the camera line shows the configured port on a non-loopback listener, and every `nc` command reports success with exit status 0. These checks prove both manager-command and feedback/camera directions; they do not prove message content.

- [ ] **Step 3: Add the bounded pre-engagement `robot_config` content probe**

Run in workstation Terminal 2 from `$WORKSTATION_REPO_DIR`:

```bash
.venv_data_collection/bin/python - "$PC2_IP" <<'PY'
import sys

from gear_sonic.utils.data_collection.zmq_state_subscriber import poll_robot_config_zmq

host = sys.argv[1]
expected = {
    "model_path": "policy/release/model_decoder.onnx",
    "reference_motion_path": "reference/example/",
    "planner_path": "planner/target_vel/V2/planner_sonic.onnx",
    "obs_config_path": "policy/release/observation_config.yaml",
    "encoder_file": "policy/release/model_encoder.onnx",
    "control_frequency": 50,
    "planner_frequency": 10,
    "is_using_encoder": True,
}
config = poll_robot_config_zmq(host, 5557, timeout_sec=10)
mismatches = {
    key: {"expected": value, "actual": config.get(key)}
    for key, value in expected.items()
    if config.get(key) != value
}
if mismatches:
    raise SystemExit(f"FAIL: robot_config mismatch: {mismatches}")
print("PASS: robot_config matches the pinned deployment")
PY
```

Expected: `PASS: robot_config matches the pinned deployment` and exit status 0. Also require the PC2 deploy terminal to show `Init Done` with no continuing LowState-unavailable message and no CRC/safety error. Do not subscribe to `g1_debug` before engagement because it is not published in `INIT` or `WAIT_FOR_CONTROL`.

- [ ] **Step 4: Add the PICO engagement action and startup stop warning**

Require the VR operator to assume the documented CALIB_FULL pose, then press PICO `A+B+X+Y` to enter planner mode. State that `]` is not an engagement key for `zmq_manager`.

Add a danger admonition explaining:

- after the start message sets `operator_state.start`, the ZMQ manager input thread may block for up to five seconds waiting for planner initialization;
- keyboard `O` and a later PICO stop are not guaranteed immediate during that interval;
- the safety operator keeps the deploy terminal focused for normal `O` handling but continuously holds the independent hardware E-stop/power-cut control;
- neither software input qualifies as the independent startup E-stop.

- [ ] **Step 5: Add the bounded post-start `g1_debug` probe**

Immediately after PICO start, in workstation Terminal 2:

```bash
.venv_data_collection/bin/python - "$PC2_IP" <<'PY'
import sys
import time

import numpy as np

from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber

subscriber = ZMQStateSubscriber(host=sys.argv[1], port=5557)
deadline = time.monotonic() + 10.0
try:
    while time.monotonic() < deadline:
        message = subscriber.get_msg()
        if message is None:
            time.sleep(0.02)
            continue
        measured = np.asarray(message.get("body_q_measured"))
        if measured.ndim != 1 or measured.size < 29:
            raise SystemExit(f"FAIL: body_q_measured shape is {measured.shape}")
        if not np.isfinite(measured).all():
            raise SystemExit("FAIL: body_q_measured contains non-finite values")
        print("PASS: finite g1_debug body_q_measured received")
        break
    else:
        raise SystemExit("FAIL: no g1_debug message within 10 seconds")
finally:
    subscriber.close()
PY
```

Expected: the `PASS` line and exit status 0. Run this before VR_3PT, exporter startup, viewer startup, or recording. On any nonzero exit, malformed message, or non-finite state, the safety operator immediately uses `<LAB_APPROVED_HARDWARE_ESTOP_PROCEDURE>` without waiting for `O`; do not continue.

- [ ] **Step 6: Add exporter and viewer commands after the post-start pass**

Workstation Terminal 2, exporter:

```bash
cd "$WORKSTATION_REPO_DIR"
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_data_exporter.py \
  --dataset-name "$DATASET_NAME" \
  --task-prompt "$TASK_PROMPT" \
  --camera-host "$PC2_IP" \
  --camera-port "$CAMERA_PORT" \
  --sonic-zmq-host localhost \
  --sonic-zmq-port 5556 \
  --state-zmq-host "$PC2_IP" \
  --state-zmq-port 5557 \
  --robot-config-timeout 10
```

Workstation Terminal 3, camera viewer:

```bash
cd "$WORKSTATION_REPO_DIR"
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_camera_viewer.py \
  --camera-host "$PC2_IP" \
  --camera-port "$CAMERA_PORT"
```

Expected: the exporter receives robot config/state, manager pose, and camera frames without timeout; the viewer prints detected streams and displays live frames.

Document recording controls:

- `Left Grip + A`: start an episode; press again to save it.
- `Left Grip + B`: discard the active episode.

- [ ] **Step 7: Add ordered fault handling, normal shutdown, and troubleshooting**

Normal shutdown order:

1. Press keyboard `O` in the focused PC2 deploy terminal and wait for deploy stop confirmation. If response is uncertain, use the independent hardware stop/power cut and expect the harness/frame to carry dead weight.
2. If an exporter episode is active, press `Left Grip + A` and wait for `Finished saving episode` plus idle confirmation, or press `Left Grip + B` and wait for `Discarded episode`.
3. Only after exporter idle, press `Ctrl+C` in the exporter terminal. Warn that interrupting with buffered frames marks the episode discarded.
4. Press `Q` in the camera viewer.
5. Stop the PICO manager with `Ctrl+C`.
6. Stop the foreground PC2 camera server with `Ctrl+C`.

Add a compact troubleshooting table covering:

- Git LFS pointer/missing artifact versus the pinned Hub download and hash check.
- Wrong active virtual environment.
- Cross-machine endpoint accidentally set to `localhost`.
- Ports 5556, 5557, or camera port missing/blocked.
- `robot_config` mismatch.
- no post-start `g1_debug` or non-finite measured state.
- camera detection, frame timeout, or active systemd collision.
- PC2 and workstation revisions differing.

- [ ] **Step 8: Check Section 5 and commit it**

Run:

```bash
rg -n '^## 5\.|--zmq_feedback_host|robot_config matches|finite g1_debug|body_q_measured|--state-zmq-host|Left Grip \+ A|Left Grip \+ B|Finished saving episode|Discarded episode' docs/source/getting_started/newcomer_onboarding.md
git diff --check -- docs/source/getting_started/newcomer_onboarding.md
```

Expected: all lifecycle markers are found and the whitespace check exits 0.

Commit:

```bash
git add docs/source/getting_started/newcomer_onboarding.md
git commit -m "docs: document workstation data collection"
```

### Task 6: Publish in the Toctree and Run Documentation Verification

**Files:**
- Modify: `docs/source/index.rst`
- Verify: `docs/source/getting_started/newcomer_onboarding.md`
- Verify: `docs/source/getting_started/gear_sonic_deployment_9c0ff22.sha256`

- [ ] **Step 1: Confirm the new page is not yet in the Getting Started toctree**

Run:

```bash
test "$(rg -c '^   getting_started/newcomer_onboarding$' docs/source/index.rst)" -eq 0
```

Expected: exit status 0.

- [ ] **Step 2: Add the page to the Getting Started toctree**

Insert this line immediately after `getting_started/vr_teleop_setup`:

```rst
   getting_started/newcomer_onboarding
```

- [ ] **Step 3: Verify required headings and the intentionally blank Section 3**

Run:

```bash
rg -n '^## [0-5]\.' docs/source/getting_started/newcomer_onboarding.md
```

Expected exactly these six ordered headings:

```text
## 0. Clone the GR00T Repository
## 1. Install the Required Environments
## 2. Deploy GEAR-SONIC on PC2
## 3. Set Up PICO Teleoperation
## 4. Run the Camera Server on PC2
## 5. Run Workstation Processes
```

Run this structural check:

```bash
python - <<'PY'
from pathlib import Path

text = Path("docs/source/getting_started/newcomer_onboarding.md").read_text()
between = text.split("## 3. Set Up PICO Teleoperation", 1)[1].split(
    "## 4. Run the Camera Server on PC2", 1
)[0]
assert not between.strip(), f"Section 3 must be blank, found: {between!r}"
print("PASS: Section 3 is intentionally blank")
PY
```

Expected: the `PASS` line.

- [ ] **Step 4: Verify the exact user-facing token inventory**

Run:

```bash
python - <<'PY'
from pathlib import Path
import re

text = Path("docs/source/getting_started/newcomer_onboarding.md").read_text()
actual = set(re.findall(r"<[A-Z][A-Z0-9_]+>", text))
expected = {
    "<PC2_IP>",
    "<PC2_USER>",
    "<PC2_REPO_DIR>",
    "<WORKSTATION_IP>",
    "<WORKSTATION_REPO_DIR>",
    "<ROBOT_NETWORK_INTERFACE>",
    "<REPO_REVISION>",
    "<TENSORRT_ROOT>",
    "<EGO_CAMERA_TYPE>",
    "<EGO_CAMERA_DEVICE_ID>",
    "<CAMERA_PORT>",
    "<TASK_PROMPT>",
    "<DATASET_NAME>",
    "<LAB_APPROVED_HARDWARE_ESTOP_PROCEDURE>",
}
assert actual == expected, f"token mismatch: missing={expected - actual}, extra={actual - expected}"
print("PASS: configuration token inventory matches")
PY
```

Expected: the `PASS` line.

- [ ] **Step 5: Recheck pinned artifacts and scan for accidental lab values**

Run:

```bash
sha256sum --check docs/source/getting_started/gear_sonic_deployment_9c0ff22.sha256
test -s docs/source/getting_started/gear_sonic_deployment_9c0ff22.sha256
if rg -n '/home/|192\.168\.|10\.[0-9]+\.|172\.(1[6-9]|2[0-9]|3[01])\.' docs/source/getting_started/newcomer_onboarding.md; then
  echo 'FAIL: possible machine-specific path or private IP found' >&2
  exit 1
fi
echo 'PASS: no obvious machine-specific path or private IP'
```

Expected: all four artifacts report `OK`, the manifest is non-empty, and the final command prints `PASS`.

- [ ] **Step 6: Perform the command-block labeling and outcome review**

Read the completed page from top to bottom. For every shell block, confirm the immediately preceding text names one of `Workstation`, `PC2`, or `Documentation verification` and states the working directory. Confirm every command or grouped probe names an expected success message or exit status. Confirm no command uses a cross-machine `localhost` value except the exporter-to-manager endpoint on the workstation.

Expected: no unlabeled command block, no implicit working directory, no outcome-free probe, and no incorrect cross-machine `localhost` remains.

- [ ] **Step 7: Build the Sphinx documentation**

Run:

```bash
make -C docs html
test -s docs/build/html/getting_started/newcomer_onboarding.html
```

Expected: Sphinx exits 0 and the generated HTML file is non-empty.

- [ ] **Step 8: Review the final diff and commit publication**

Run:

```bash
git diff --check
git status --short
git diff -- docs/source/index.rst docs/source/getting_started/newcomer_onboarding.md docs/source/getting_started/gear_sonic_deployment_9c0ff22.sha256
```

Expected: no whitespace errors; only the three planned documentation files are part of this implementation. Preserve every unrelated pre-existing worktree change.

Commit:

```bash
git add docs/source/index.rst docs/source/getting_started/newcomer_onboarding.md docs/source/getting_started/gear_sonic_deployment_9c0ff22.sha256
git commit -m "docs: publish newcomer onboarding guide"
```

Do not start the real robot, PICO hardware, deploy binary, camera server, manager, exporter, or viewer as part of documentation verification.
