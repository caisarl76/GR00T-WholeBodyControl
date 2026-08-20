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
