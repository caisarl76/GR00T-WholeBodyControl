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
export TensorRT_ROOT='<TENSORRT_ROOT>'
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
sudo apt-get install -y git git-lfs iproute2 netcat-openbsd ripgrep
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

## 1. Install the Required Environments

The environment installers are destructive with respect to their target virtual
environments: `install_scripts/install_pico.sh` deletes and recreates
`.venv_teleop`, `install_scripts/install_data_collection.sh` deletes and
recreates `.venv_data_collection`, and
`install_scripts/install_camera_server.sh` deletes and recreates
`.venv_camera`. Do not rerun them casually when locally installed packages or
other changes inside those environments matter.

This workflow does not use unqualified `python check_environment.py`. Its
default all-mode includes unrelated training, Isaac Lab, CUDA, and TensorRT
checks. Use the focused checks below instead.

**Workstation — `$WORKSTATION_REPO_DIR`**

```bash
(
  set -euo pipefail
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
)
```

Expected: both installers exit 0, both Python version commands report Python
3.10, both import commands print their `PASS` line, and every help command exits
0.

**PC2 — `$PC2_REPO_DIR`**

```bash
(
  set -euo pipefail
  cd "$PC2_REPO_DIR"
  bash install_scripts/install_camera_server.sh

  .venv_camera/bin/python --version
  .venv_camera/bin/python -c 'import depthai, msgpack_numpy, zmq; print("PASS: camera imports")'
  .venv_camera/bin/python -m gear_sonic.camera.composed_camera --help
)
```

When prompted to install the systemd service, answer `n`. This prevents this
installer run from creating or starting the service so foreground mode can be
used later, but it does not prove that a service left by an older installation
is inactive. Section 4 checks that condition before foreground launch.

Expected: the installer exits 0, the version command reports Python 3.10, the
import command prints `PASS: camera imports`, and the help command exits 0.
These checks validate the common camera environment only; they do not establish
that a RealSense camera is ready. The camera-type-specific checks come in
Section 4.

### PC2 Native Dependencies, Platform Gate, and Build

`scripts/install_deps.sh` may use `sudo` and install system packages, including
JetPack or CUDA packages where appropriate. It therefore runs before the
platform and TensorRT gate, which must pass before the build.

Sourcing `scripts/setup_env.sh` also has host effects. On a bare Jetson it may
run `sudo jetson_clocks`, create DLA symlinks under
`/usr/lib/aarch64-linux-gnu/nvidia`, and append the DLA library path to
`~/.bashrc`. Obtain operator and administrator authorization for these package,
privilege, performance-mode, system-library, and shell-profile changes before
running the block. Stop here if that authorization is not granted.

**PC2 — `$PC2_REPO_DIR/gear_sonic_deploy`**

```bash
(
  set -euo pipefail
  cd "$PC2_REPO_DIR/gear_sonic_deploy"
  bash scripts/install_deps.sh

  ARCH="$(uname -m)"
  printf 'PC2 architecture: %s\n' "$ARCH"
  test -d "$TensorRT_ROOT"
  test -x "$TensorRT_ROOT/bin/trtexec"
  TRT_VERSION="$("$TensorRT_ROOT/bin/trtexec" --version 2>&1)"
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

  VALIDATED_TENSORRT_ROOT="$TensorRT_ROOT"
  set +u
  source scripts/setup_env.sh
  set -u
  test "$TensorRT_ROOT" = "$VALIDATED_TENSORRT_ROOT"

  CANONICAL_TENSORRT_ROOT="$(realpath -e "$TensorRT_ROOT")"
  test "$CANONICAL_TENSORRT_ROOT" != /
  mkdir -p build
  BUILD_DIR="$(mktemp -d "$PWD/build/onboarding.XXXXXX")"
  printf 'Fresh build directory: %s\n' "$BUILD_DIR"
  cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DBUILD_ROS2=OFF \
    -DBUILD_DEPLOY_TESTS=OFF \
    -DTensorRT_ROOT="$CANONICAL_TENSORRT_ROOT"

  mapfile -t TRT_INCLUDE_PATHS < <(
    sed -n 's/^TensorRT_INCLUDE_DIR:[^=]*=//p' "$BUILD_DIR/CMakeCache.txt"
  )
  mapfile -t TRT_LIBRARY_PATHS < <(
    sed -nE 's/^TensorRT_[A-Za-z0-9_]+_LIBRARY:[^=]*=(.*)$/\1/p' "$BUILD_DIR/CMakeCache.txt"
  )
  test "${#TRT_INCLUDE_PATHS[@]}" -eq 1
  TRT_INCLUDE_DIR="${TRT_INCLUDE_PATHS[0]}"
  test -n "$TRT_INCLUDE_DIR"
  test "${#TRT_LIBRARY_PATHS[@]}" -eq 4

  for TRT_PATH in "$TRT_INCLUDE_DIR" "${TRT_LIBRARY_PATHS[@]}"; do
    CANONICAL_TRT_PATH="$(realpath -e "$TRT_PATH")"
    case "$CANONICAL_TRT_PATH" in
      "$CANONICAL_TENSORRT_ROOT"/*) ;;
      *)
        printf 'TensorRT cache path escaped validated root: %s\n' \
          "$CANONICAL_TRT_PATH" >&2
        exit 1
        ;;
    esac
  done

  TRT_VERSION_HEADER="$TRT_INCLUDE_DIR/NvInferVersion.h"
  grep -Eq '^[[:space:]]*#[[:space:]]*define[[:space:]]+NV_TENSORRT_MAJOR[[:space:]]+10([[:space:]]|$)' \
    "$TRT_VERSION_HEADER"
  case "$ARCH" in
    x86_64) TRT_EXPECTED_MINOR=13 ;;
    aarch64) TRT_EXPECTED_MINOR=7 ;;
  esac
  grep -Eq "^[[:space:]]*#[[:space:]]*define[[:space:]]+NV_TENSORRT_MINOR[[:space:]]+${TRT_EXPECTED_MINOR}([[:space:]]|$)" \
    "$TRT_VERSION_HEADER"

  cmake --build "$BUILD_DIR" -j"$(nproc)"
  test -x target/release/g1_deploy_onnx_ref
)
```

This is one fail-fast subshell: a failed dependency installation or gate cannot
be hidden by a later successful command, and the build must consume the same
TensorRT root that passed validation. Nounset is deliberately disabled only
while sourcing the upstream `setup_env.sh` because it reads variables that are
normally unset; `set -u` restores it immediately afterward.

The direct CMake configure mirrors the current `.justfile` recipe flags but
replaces `just build` because that recipe reuses `build/`; its cache could
resolve stale or default TensorRT paths. `mktemp` instead creates a unique fresh
binary directory under the ignored `build/` directory. It does not delete or
overwrite an existing build, so the isolated directory remains available for
inspection or later cleanup. The root test rejects `/`: `TensorRT_ROOT` must
name a specific TensorRT installation so the quoted containment check is
unambiguous.

Expected: every top-level command in the subshell exits 0. An `x86_64` PC2 must
use TensorRT 10.13. An `aarch64` PC2 must use L4T R36.x, JetPack 6 when the
`nvidia-jetpack` metapackage is installed, and TensorRT 10.7. Environment setup
prints `TensorRT environment configured`, and the validated root comparison
exits 0. The fresh CMake cache must contain exactly one include path and four
component-library paths, all resolving canonically beneath the validated root;
the cached header must report TensorRT major 10 and platform-specific minor 13
or 7. Only then does the build run; it and the executable check must exit 0.

```{danger}
This gate is a hard stop. A platform or TensorRT mismatch can produce unsafe
planner inference. Do not bypass, weaken, or ignore a failed check; correct the
PC2 platform and TensorRT installation before continuing to the native build.
```

### Immutable Deployment Artifacts

Install the Hugging Face CLI into the existing camera environment rather than
creating another environment.

**PC2 — `$PC2_REPO_DIR`**

```bash
(
  set -euo pipefail
  cd "$PC2_REPO_DIR"
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  command -v uv
  uv pip install --python .venv_camera/bin/python huggingface_hub
  test -x .venv_camera/bin/hf
)
```

The camera installer may place `uv` in one of these user-local directories, but
changes made to its child-shell `PATH` do not propagate to this terminal.
Expected: `command -v uv` succeeds, installation exits 0, and the executable
check confirms `.venv_camera/bin/hf`.

Use the pinned shell constant from the PC2 configuration for every download.

**PC2 — `$PC2_REPO_DIR`**

```bash
(
  set -euo pipefail
  cd "$PC2_REPO_DIR"
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
)
```

Expected: all four downloads exit 0 and populate the named paths. Do not use
`python download_from_hf.py`: it resolves the mutable `main` branch and accepts
no Hugging Face Hub revision.

The repository already tracks `gear_sonic_deploy/reference/example` through Git
LFS. Its contents are pinned by `$REPO_REVISION` and populated by the successful
Section 0 `git lfs pull`; do not replace them with a mutable download or runtime
conversion. Later deployment uses `gear_sonic_deploy/reference/example/`
directly.

### Artifact Integrity Checks

**PC2 — `$PC2_REPO_DIR`**

```bash
(
  set -euo pipefail
  cd "$PC2_REPO_DIR"
  command -v rg
  test -s gear_sonic_deploy/policy/release/model_encoder.onnx
  test -s gear_sonic_deploy/policy/release/model_decoder.onnx
  test -s gear_sonic_deploy/policy/release/observation_config.yaml
  test -s gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx
  sha256sum --check docs/source/getting_started/gear_sonic_deployment_9c0ff22.sha256
  if rg -l '^version https://git-lfs.github.com/spec/v1$' gear_sonic_deploy/reference/example; then
    printf 'Unresolved Git LFS pointers found under reference/example\n' >&2
    exit 1
  else
    RG_STATUS=$?
    if [ "$RG_STATUS" -ne 1 ]; then
      printf 'Reference-tree scan failed with status %s\n' "$RG_STATUS" >&2
      exit "$RG_STATUS"
    fi
  fi
  find gear_sonic_deploy/reference/example -type f -name joint_pos.csv -size +0c \
    -exec awk -F, 'NR == 1 { exit !($1 == "joint_0") }' {} \; -print -quit | grep -q .
)
```

Expected: `command -v rg` succeeds, all file checks exit 0, all four hashes
report `OK`, no unresolved Git LFS pointer exists under `reference/example`, and
only `rg` status 1 (no matches) is accepted. The final check finds at least one
nonempty real joint CSV whose first record begins with the field `joint_0`.

## 2. Deploy GEAR-SONIC on PC2

This lab runbook is only for the physically confirmed configuration without
Dex3 or Inspire hands. The operator and robot owner must physically inspect the
robot together and verbally confirm that no hands are attached before
continuing. If the hardware differs or either person is uncertain, stop. The
`deploy.sh` wrapper cannot forward `--disable-dex3-hands`, so this configuration
uses the validated direct `just run` invocation below.

Use one focused deployment terminal so setup, preflight, and launch share the
same current environment. Nounset is disabled only while sourcing the upstream
`setup_env.sh`, which reads normally unset variables, and is restored
immediately afterward. The preflight is interactive: confirm every prompt and
require `Real-robot preflight confirmed.` before launch. After preflight, the
operator must type exactly `ACTUATE` at the distinct actuation prompt. Any other
input or EOF aborts before launch, and any failed command stops the entire
block.

Before running it, fill in `<LAB_APPROVED_HARDWARE_ESTOP_PROCEDURE>` with the
exact lab procedure and verbally confirm the completed procedure with the robot
owner.

**PC2 focused deployment terminal — `$PC2_REPO_DIR/gear_sonic_deploy`**

```{danger}
After preflight, typing `ACTUATE` and pressing Enter starts an actuated initialization immediately. The deploy process drives all joints toward the default standing pose over three seconds with nonzero gains while low-level commands are published at 500 Hz. PICO engagement starts policy CONTROL, but it is not the first robot motion. Before typing `ACTUATE` and pressing Enter, the protective harness/frame, clear 3 m zone, spotter, and independent hardware E-stop or physical power-cut procedure must already be ready.
```

```bash
(
  set -euo pipefail
  cd "$PC2_REPO_DIR/gear_sonic_deploy"
  set +u
  source scripts/setup_env.sh
  set -u
  bash scripts/preflight.sh
  IFS= read -r -p 'Type ACTUATE to begin the three-second initialization ramp: ' ACTUATION_CONFIRMATION
  test "$ACTUATION_CONFIRMATION" = ACTUATE
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
)
```

Expected sequence: preflight prints `Real-robot preflight confirmed.`, the
operator types exactly `ACTUATE` at the confirmation prompt, and only then does
deployment produce the startup evidence below. Any other input or EOF aborts
the block before `just run`.

Expected startup evidence:

- Input type is `zmq_manager`.
- Output is ZMQ and port 5557 binds successfully.
- The configured workstation host is displayed.
- `[INFO] Dex3 hands disabled` is printed.
- Transient LowState-unavailable messages stop.
- `Init Done` proves the process reached WAIT_FOR_CONTROL.
- No CRC or safety error is reported.

Policy CONTROL waits for the later PICO start, but the three-second
initialization has already actuated the robot. Keep this deployment terminal
running and focused for the safety operator; do not reuse it for other work.

## 3. Set Up PICO Teleoperation
## 4. Run the Camera Server on PC2

Section 1 created `.venv_camera`; do not rerun the destructive camera
installer. Open a new PC2 terminal, load the PC2 configuration variables, and
work from `$PC2_REPO_DIR`.

First, establish mutual exclusion with the systemd launch path. Foreground mode
requires that no camera service unit is installed.

**PC2 new camera terminal — `$PC2_REPO_DIR`**

```bash
(
  set -euo pipefail
  cd "$PC2_REPO_DIR"
  if ! CAMERA_SERVICE_LOAD_STATE="$(systemctl show --property=LoadState --value composed_camera_server.service 2>/dev/null)"; then
    echo 'ERROR: could not query camera service LoadState' >&2
    exit 1
  fi
  case "$CAMERA_SERVICE_LOAD_STATE" in
    not-found) ;;
    '')
      echo 'ERROR: camera service LoadState is empty; cannot prove foreground exclusivity' >&2
      exit 1
      ;;
    *)
      echo "ERROR: installed camera service (LoadState=$CAMERA_SERVICE_LOAD_STATE) is an alternate launch path; use it or have the robot owner remove it before foreground mode" >&2
      exit 1
      ;;
  esac
  echo 'PASS: camera service LoadState=not-found; no unit is installed'
)
```

Expected: `PASS: camera service LoadState=not-found; no unit is installed` and
exit 0. A query failure, empty state, or any installed unit state is a hard
stop. If a unit is installed, use the systemd launch branch instead or have the
robot owner remove it before continuing with foreground mode. This foreground
branch never treats an installed inactive, failed, masked, or transitional unit
as safe; the repository unit uses `Restart=on-failure`.

Before camera-specific discovery, reject every unsupported configured type.
Run this command directly so its failure cannot be masked by a later command.

**PC2 new camera terminal — `$PC2_REPO_DIR`**

```bash
case "$EGO_CAMERA_TYPE" in
  oak|oak_mono|realsense) ;;
  *) echo "Unsupported EGO_CAMERA_TYPE: $EGO_CAMERA_TYPE" >&2; exit 1 ;;
esac
```

Expected: exit 0 only for `oak`, `oak_mono`, or `realsense`. Any other value
prints the unsupported type and exits 1; stop there.

Run only the discovery branch matching the configured camera type. For `oak`
or `oak_mono`, use the existing DepthAI dependency. The probe requires
`DeviceInfo.getDeviceId()` and validates exactly the identifier API that the
runtime OAK driver consumes. An SDK lacking that API is incompatible and must
not pass discovery.

**PC2 new camera terminal — `$PC2_REPO_DIR`**

```bash
(
  set -euo pipefail
  cd "$PC2_REPO_DIR"
  .venv_camera/bin/python - "$EGO_CAMERA_DEVICE_ID" <<'PY'
import sys
import depthai as dai

expected = sys.argv[1]
devices = dai.Device.getAllAvailableDevices()
ids = []
for device in devices:
    runtime_getter = getattr(device, "getDeviceId", None)
    if not callable(runtime_getter):
        raise SystemExit("FAIL: installed DepthAI lacks DeviceInfo.getDeviceId required by runtime")
    ids.append(runtime_getter())
print("Detected OAK IDs:", ids)
if not ids:
    raise SystemExit("FAIL: no OAK camera detected")
if expected and expected not in ids:
    raise SystemExit(f"FAIL: configured OAK ID {expected!r} not found")
print("PASS: OAK camera detected")
PY
)
```

Expected: the detected OAK IDs are printed, followed by
`PASS: OAK camera detected`, and the probe exits 0. An empty configured device
ID is allowed when the selected backend does not need one.

For `realsense`, install its separate driver into the existing camera
environment first. This is a new terminal, so make the user-local `uv`
locations discoverable explicitly.

**PC2 new camera terminal — `$PC2_REPO_DIR`**

```bash
(
  set -euo pipefail
  cd "$PC2_REPO_DIR"
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  command -v uv
  uv pip install --python .venv_camera/bin/python pyrealsense2
)
```

Expected: `command -v uv` prints its executable path, the `pyrealsense2`
installation exits 0, and no new virtual environment is created. Then probe
the configured serial in the same PC2 repo context; RealSense requires an
explicit nonempty serial.

**PC2 new camera terminal — `$PC2_REPO_DIR`**

```bash
(
  set -euo pipefail
  cd "$PC2_REPO_DIR"
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
)
```

Expected: the detected serial list is printed, followed by
`PASS: RealSense camera detected`, and the probe exits 0. A missing or
unmatched configured serial is a hard stop.

After the matching discovery probe passes, start the foreground camera server
in that same new PC2 terminal. The Bash array omits the device-ID option when
the configured ID is empty instead of passing an empty CLI argument.

**PC2 new camera terminal — `$PC2_REPO_DIR`**

```bash
(
  set -euo pipefail
  cd "$PC2_REPO_DIR"
  CAMERA_DEVICE_ARGS=()
  if [[ -n "$EGO_CAMERA_DEVICE_ID" ]]; then
    CAMERA_DEVICE_ARGS=(--ego-view-device-id "$EGO_CAMERA_DEVICE_ID")
  fi

  if ! CAMERA_SERVICE_LOAD_STATE="$(systemctl show --property=LoadState --value composed_camera_server.service 2>/dev/null)"; then
    echo 'ERROR: could not query camera service LoadState' >&2
    exit 1
  fi
  case "$CAMERA_SERVICE_LOAD_STATE" in
    not-found) ;;
    '')
      echo 'ERROR: camera service LoadState is empty; cannot prove foreground exclusivity' >&2
      exit 1
      ;;
    *)
      echo "ERROR: installed camera service (LoadState=$CAMERA_SERVICE_LOAD_STATE) is an alternate launch path; use it or have the robot owner remove it before foreground mode" >&2
      exit 1
      ;;
  esac
  echo 'PASS: camera service LoadState=not-found; no unit is installed'

  .venv_camera/bin/python -m gear_sonic.camera.composed_camera \
    --ego-view-camera "$EGO_CAMERA_TYPE" \
    "${CAMERA_DEVICE_ARGS[@]}" \
    --port "$CAMERA_PORT"
)
```

Expected: the repeated fail-closed classifier prints its second `PASS`
immediately before starting the foreground Python process, the configured
backend initializes, the server binds the configured port, frames are
published, and no repeated timeout or reconnect messages appear. Keep this
foreground terminal running.

## 5. Run Workstation Processes

Start only the PICO manager first. Do not start the exporter or viewer yet;
their startup is gated on the directional network checks and the pre- and
post-engagement probes below.

**Workstation Terminal 1 — `$WORKSTATION_REPO_DIR`**

```bash
(
  set -euo pipefail
  cd "$WORKSTATION_REPO_DIR"
  .venv_teleop/bin/python gear_sonic/scripts/pico_manager_thread_server.py \
    --manager \
    --port 5556 \
    --zmq_feedback_host "$PC2_IP" \
    --zmq_feedback_port 5557
)
```

Expected: the process enters interactive manager mode, listens on workstation
port 5556, and connects its feedback input to PC2 port 5557. Keep this
foreground process running in Workstation Terminal 1. **Do not start the
exporter or viewer yet.**

### Verify Both Network Directions Before Engagement

These commands prove TCP reachability only. They do not prove that a peer is
publishing valid robot state, planner commands, or camera content. Run each
block directly so that `set -e` stops at the first failed check and a later
success cannot mask it.

**Workstation Terminal 2 — any working directory**

```bash
(
  set -euo pipefail
  MANAGER_LISTENER="$(ss -H -ltnp 'sport = :5556')"
  test -n "$MANAGER_LISTENER"
  printf '%s\n' "$MANAGER_LISTENER"
  nc -zvw 3 "$PC2_IP" 5557
  nc -zvw 3 "$PC2_IP" "$CAMERA_PORT"
)
```

Expected: the exact socket filter returns a nonempty listener on port 5556,
with output showing the expected wildcard address and the manager process when
`ss -p` permissions expose it. The filter alone proves that something listens
on the exact port, not its identity. If its address or visible process
contradicts Workstation Terminal 1, stop. Both `nc` commands must report success
and exit 0; they check the workstation-to-PC2 paths for deployment state and
camera traffic.

**PC2 Terminal 2 — any working directory**

```bash
(
  set -euo pipefail
  DEPLOY_LISTENER="$(ss -H -ltnp 'sport = :5557')"
  CAMERA_LISTENER="$(ss -H -ltnp "sport = :$CAMERA_PORT")"
  test -n "$DEPLOY_LISTENER"
  test -n "$CAMERA_LISTENER"
  printf '%s\n' "$DEPLOY_LISTENER"
  printf '%s\n' "$CAMERA_LISTENER"
  nc -zvw 3 "$WORKSTATION_IP" 5556
)
```

Expected: both exact socket filters return nonempty results. Their output must
show deployment on port 5557 using the expected wildcard address and the
camera server on its configured port using a non-loopback address; expected
process names should also appear when `ss -p` permissions expose them. The
filters alone prove only that something listens on each exact port, not the
listener identities. Stop if an address or visible process contradicts the
focused deployment or camera terminal. `nc` must reach the workstation manager
on port 5556 and exit 0. The later content probes supply protocol-level
evidence; stop on any failure here because all three cross-machine paths must
work in the documented directions.

### Probe the Pinned Configuration Before Engagement

Before engaging the robot, request one bounded `robot_config` response. This
checks the current exact ten-field schema and every advertised value against
the pinned configuration; it does not engage CONTROL.

**Workstation Terminal 2 — `$WORKSTATION_REPO_DIR`**

```bash
(
  set -euo pipefail
  cd "$WORKSTATION_REPO_DIR"
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
    "policy_fp16": False,
    "planner_fp16": False,
}
config = poll_robot_config_zmq(host, 5557, timeout_sec=10)
missing = sorted(set(expected) - set(config))
unexpected = sorted(set(config) - set(expected))
if missing or unexpected:
    raise SystemExit(f"FAIL: robot_config schema mismatch: missing={missing}, unexpected={unexpected}")
mismatches = {
    key: {"expected": value, "actual": config.get(key)}
    for key, value in expected.items()
    if config.get(key) != value
}
if mismatches:
    raise SystemExit(f"FAIL: robot_config mismatch: {mismatches}")
print("PASS: robot_config matches the pinned deployment")
PY
)
```

Expected: the final line is exactly
`PASS: robot_config matches the pinned deployment`, and the probe exits 0. At
the same time, the focused PC2 deployment terminal must show `Init Done`, must
not continue reporting LowState-unavailable messages, and must show no CRC or
safety error. Stop if any of these checks fail.

The manager's FeedbackReader is already subscribed to `g1_debug`, but
deployment publishes no payload on that topic while it is in INIT or
WAIT_FOR_CONTROL. Do **not** run a standalone `g1_debug` probe or treat absence
as a failure until CONTROL has entered the startup-ready state below.

### Engage Planner Mode

The VR operator assumes the `CALIB_FULL` pose documented in the VR teleoperation
setup, then presses the PICO `A+B+X+Y` combination to enter planner mode. The
keyboard `]` key is **not** the engagement key for the `zmq_manager` input mode
used by this runbook. Begin the startup gate immediately when the combination
is pressed.

```{danger}
After PICO start sets `operator_state.start`, the ZMQ manager input thread may
block for up to five seconds while it waits for planner initialization. During
that interval, keyboard uppercase `O` and a later PICO stop are not guaranteed
to take effect immediately.

The safety operator must watch the focused PC2 deployment terminal while
continuously holding the independent hardware E-stop or power-cut described by
`<LAB_APPROVED_HARDWARE_ESTOP_PROCEDURE>`. Within five seconds of PICO start,
that terminal must print exactly
`[ZMQManager] motion name is planner_motion`, with neither
`Planner initialization timeout` nor
`Planner failed to initialize. Stopping control.`

The independent hardware-only fault-stop rule remains active until the exact
ready marker appears. If it does not appear within five seconds, or if either
error appears, immediately use `<LAB_APPROVED_HARDWARE_ESTOP_PROCEDURE>` without
waiting for uppercase `O` or PICO stop. Neither software input is an
independent startup E-stop. Only after the ready marker appears with no error
may the measured-state probe below be run or accepted.
```

### Complete the Startup Gate with a Measured-State Probe

As soon as the startup-ready marker appears with no initialization error, run
this bounded `g1_debug` probe. Both the ready marker and this probe's PASS are
required **before** starting VR_3PT, the exporter, the viewer, or recording.

**Workstation Terminal 2 — `$WORKSTATION_REPO_DIR`**

```bash
(
  set -euo pipefail
  cd "$WORKSTATION_REPO_DIR"
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
        if "body_q_measured" not in message:
            raise SystemExit("FAIL: g1_debug is missing body_q_measured")
        measured = np.asarray(message["body_q_measured"])
        if measured.shape != (29,):
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
)
```

Expected: the final line is exactly
`PASS: finite g1_debug body_q_measured received`, and the probe exits 0. Any
nonzero exit, malformed shape, or non-finite value requires the safety operator
to **immediately** use `<LAB_APPROVED_HARDWARE_ESTOP_PROCEDURE>` without waiting
for uppercase `O`. A PASS establishes that the sample contains exactly 29
NumPy-convertible numeric, finite, double-equivalent measured-joint values. Do
not continue on failure.

### Start Exporter and Viewer Only After the Probe Passes

Only after the focused PC2 terminal prints the exact startup-ready marker with
no initialization error **and** the post-start `g1_debug` probe prints its
`PASS` line may you start the exporter.

**Workstation Terminal 2 — `$WORKSTATION_REPO_DIR`**

```bash
(
  set -euo pipefail
  cd "$WORKSTATION_REPO_DIR"
  .venv_data_collection/bin/python gear_sonic/scripts/run_data_exporter.py \
    --dataset-name "$DATASET_NAME" \
    --task-prompt "$TASK_PROMPT" \
    --camera-host "$PC2_IP" \
    --camera-port "$CAMERA_PORT" \
    --sonic-zmq-host localhost \
    --sonic-zmq-port 5556 \
    --state-zmq-host "$PC2_IP" \
    --state-zmq-port 5557 \
    --robot-config-timeout 10
)
```

Expected: the exporter receives the pinned robot configuration and state from
PC2, manager pose data from workstation-local port 5556, and camera frames
from PC2 without a timeout. Keep it running.

Then start the viewer in a third workstation terminal.

**Workstation Terminal 3 — `$WORKSTATION_REPO_DIR`**

```bash
(
  set -euo pipefail
  cd "$WORKSTATION_REPO_DIR"
  .venv_data_collection/bin/python gear_sonic/scripts/run_camera_viewer.py \
    --camera-host "$PC2_IP" \
    --camera-port "$CAMERA_PORT"
)
```

Expected: the configured streams appear and live frames continue updating.
Keep the viewer running.

The recording controls are exact:

- **Left Grip + A** starts an episode; pressing **Left Grip + A** again saves
  it.
- **Left Grip + B** saves the active episode to disk marked discarded in
  `discarded_episode_indices` and returns the exporter to idle. It does not
  delete the episode.

### Normal Shutdown: Stop the Robot First

Normal shutdown must follow this exact order. A fault during the five-second
startup interval or any failed or bad state probe always bypasses this normal
sequence and uses the independent hardware stop immediately.

1. Press uppercase `O` in the focused PC2 deployment terminal. Require it to
   print `Stop` after the damping-only LowCommandWriter, then
   `[DEBUG] Program exiting normally...`, and require the `just run` command to
   return to the shell. If these markers and terminal return do not occur
   promptly, or if the result is uncertain, use the independent hardware stop
   or power-cut; the harness must carry the robot's dead weight.
2. If an episode is active, press **Left Grip + A** and wait for
   `Finished saving episode` (the code transitions to idle in the same loop),
   **or** press **Left Grip + B** and wait for `Discarded episode` (which saves
   it to disk marked in `discarded_episode_indices` and returns the exporter to
   idle; it does not delete the episode). Do not proceed without the chosen
   confirmation.
3. Only after the exporter is idle, press `Ctrl+C` in Workstation Terminal 2.
   Never interrupt during `save_episode()`; wait for
   `Finished saving episode` and idle. `Ctrl+C` attempts to mark a still-intact
   unsaved buffer discarded only when its size is greater than zero. It cannot
   guarantee recovery or marking if interruption occurs inside
   `save_episode()` after the buffer has already been consumed or popped.
4. Press lowercase `q` in the focused viewer window.
5. Press `Ctrl+C` in Workstation Terminal 1 to stop the manager.
6. Press `Ctrl+C` in the PC2 foreground camera terminal to stop the camera
   server.

The robot always stops first. Neither exporter cleanup nor viewer, manager, or
camera shutdown is a substitute for confirmed robot shutdown.

### Troubleshooting

| Symptom | Required action |
| --- | --- |
| An ONNX or other deployment artifact is a Git LFS pointer or is missing | Stop. Restore the pinned Hugging Face revision and verify the documented SHA-256 hashes from Section 1; do not substitute an unpinned artifact. |
| Imports or CLI options are missing | Confirm the command uses `.venv_teleop` for the manager, `.venv_data_collection` for exporter/viewer/probes, and `.venv_camera` for the PC2 camera server. Rerun the matching focused Section 1 checks. |
| A cross-machine command uses `localhost` | Replace it with the configured PC2 or workstation IP. Only the workstation-local manager-to-exporter pose path uses `localhost:5556`. |
| Port 5556, 5557, or the configured camera port is missing, unexpected, or blocked | Stop before engagement. Repeat both exact-filter directional `ss`/`nc` blocks, compare visible listener address/process with the launched terminals, correct binding, routing, or firewall policy, and require every command to exit 0. |
| `robot_config` schema or a value differs from the pinned contract | Stop before engagement. Check deploy arguments, artifacts, and revisions on both machines; restart with the exact pinned configuration and rerun the bounded ten-field probe. |
| The planner-ready marker is absent, an initialization error appears, or `g1_debug` is absent, malformed, or non-finite after ready | Immediately use `<LAB_APPROVED_HARDWARE_ESTOP_PROCEDURE>` without waiting for uppercase `O` or PICO stop. Inspect deploy startup and safety logs only after the robot is independently stopped; do not continue collection. |
| Camera discovery fails, frames time out, or foreground launch collides with systemd | Stop collection. Re-run the matching Section 4 camera probe, verify device ID and non-loopback listener, and resolve the installed service versus foreground launch path before retrying. |
| The two repository revisions differ | Stop. Check out the same explicit 40-character `$REPO_REVISION` on both machines, update submodules and LFS, and repeat the artifact and environment checks. |
