# Newcomer Onboarding Runbook Design

Date: 2026-08-19

## Purpose

Create a command-first onboarding page that lets a new lab member prepare and
run the GR00T Whole-Body Control data-collection stack across two machines:

- **PC2** runs the real-robot GEAR-SONIC deployment and camera server.
- **Laptop/workstation** runs the PICO manager, data exporter, and camera
  viewer.

The guide must not contain the lab's current IP addresses, usernames, or
machine-specific repository paths. Readers fill in an explicit configuration
table before running any commands. This page is authoritative for the requested
PC2/workstation split and explicitly overrides the existing data-collection
tutorial's topology where that tutorial places the C++ deployment on the
workstation.

## Destination and Discoverability

Add the user-facing guide at:

```text
docs/source/getting_started/newcomer_onboarding.md
```

Add it to the **Getting Started** toctree in `docs/source/index.rst`. Keep the
existing installation, VR setup, deployment, and data-collection pages as the
detailed references; the new page is the single ordered runbook for the lab's
PC2/workstation topology.

## Reader Contract

The runbook targets a newcomer who can use a Linux shell but does not yet know
which GR00T component belongs on which machine. It assumes authorized access to
PC2 and the real Unitree G1. Every command block is labeled with its execution
machine and working directory.

Before the real-robot procedure, the reader must have:

- completed the repository's MuJoCo quick start with the same intended input
  sequence;
- validated the keyboard, PICO, XR-loss, and mismatched-entry stop paths from
  `docs/source/user_guide/real_robot_safety.md` in simulation;
- completed `docs/source/getting_started/vr_teleop_setup.md`, including
  XRoboToolkit PC service installation, headset networking, and the required
  tracker/controller calibration;
- identified a safety operator and satisfied the repository's real-robot
  preflight requirements.

The onboarding page does not reteach simulation or PICO hardware setup, but it
must present these as hard readiness gates rather than optional references.

Commands are runnable after the reader replaces the documented configuration
tokens. The token table includes:

- `<PC2_IP>`
- `<PC2_USER>`
- `<PC2_REPO_DIR>`
- `<WORKSTATION_IP>`
- `<WORKSTATION_REPO_DIR>`
- `<ROBOT_NETWORK_INTERFACE>`
- `<REPO_REVISION>`
- `<EGO_CAMERA_TYPE>`
- `<EGO_CAMERA_DEVICE_ID>`
- `<CAMERA_PORT>`
- `<TASK_PROMPT>`
- `<DATASET_NAME>`

Angle-bracket values are explicitly described as user-supplied configuration,
not unfinished documentation. The page begins with copyable shell-variable
blocks for each machine; readers replace the angle-bracket values once and the
later commands use the named variables so the shell does not interpret angle
brackets as redirection.

## Runtime Topology

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

The deploy subscribes to the PICO manager at `<WORKSTATION_IP>:5556`. The PICO
manager and data exporter both subscribe to PC2's measured robot state at
`<PC2_IP>:5557`. The data exporter additionally connects to its local PICO
manager and PC2's camera server. The guide calls out both network directions
because `localhost` is correct only for the exporter-to-manager connection on
the workstation; using it for either cross-machine path breaks teleoperation or
measured-state calibration.

## Document Structure

### 0. Clone the GR00T Repository

Show Git LFS prerequisites and clone commands for both the workstation and
PC2. Include `git lfs pull`. Require both clones to check out the same explicit
`<REPO_REVISION>`, and compare `git rev-parse HEAD` output before any processes
are launched because the Python and C++ components share a ZMQ wire format.
The guide does not embed credentials or assume identical home-directory paths.

Do not use an unqualified `python check_environment.py` in this workflow. Its
default mode also checks training and Isaac Lab requirements that are unrelated
to the Python 3.10 teleop/data environments. Use focused install and CLI checks
for each component instead.

### 1. Install the Virtual Environments

Explain that the repository intentionally uses separate environments. Install
only the environments needed by this workflow:

- Workstation: `.venv_teleop` via `install_scripts/install_pico.sh`.
- Workstation: `.venv_data_collection` via
  `install_scripts/install_data_collection.sh`.
- PC2: `.venv_camera` via `install_scripts/install_camera_server.sh`.
- PC2: C++ deployment dependencies and build setup under
  `gear_sonic_deploy`; this is not a Python virtual environment.

Ensure `iproute2` (`ss`) and `netcat-openbsd` (`nc`) are installed on both
machines because the pre-engagement checks depend on them. The page must show
the package-install command before using either diagnostic.

Provide lightweight import or `--help` checks after installation. Warn that the
install scripts recreate their target virtual environments, so readers should
not rerun them casually when an environment contains local packages.

On PC2, use the already-created camera environment to install
`huggingface_hub`, then run the repository downloader from the repository root:

```sh
uv pip install --python .venv_camera/bin/python huggingface_hub
.venv_camera/bin/python download_from_hf.py
```

Git LFS does not provide the ignored deployment ONNX files. Before building or
launching, require exact successful `test -s` checks for:

```text
gear_sonic_deploy/policy/release/model_encoder.onnx
gear_sonic_deploy/policy/release/model_decoder.onnx
gear_sonic_deploy/policy/release/observation_config.yaml
gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx
```

The expected outcome is exit status zero for every check and a non-empty
`gear_sonic_deploy/reference/example/` motion-data directory.

### 2. Deploy GEAR-SONIC on PC2

Use the repository-supported deployment wrapper with explicit cross-machine
input settings:

1. SSH to PC2 and enter `gear_sonic_deploy`.
2. Source `scripts/setup_env.sh` and build with `just build` during one-time
   setup.
3. Start the real-robot process without engaging the policy:

   ```sh
   bash deploy.sh \
     --input-type zmq_manager \
     --output-type zmq \
     --zmq-host <WORKSTATION_IP> \
     <ROBOT_NETWORK_INTERFACE>
   ```

4. Confirm that the printed configuration says `Input Type: zmq_manager`,
   `Output Type: zmq`, and `ZMQ Host: <WORKSTATION_IP>`.
5. Complete the deploy preflight, but do not press the policy-engagement key
   until the Section 5 bidirectional network checks pass.
6. Confirm that PC2 listens on TCP port 5557 after the deploy process starts.

The guide does not recommend disabling CRC, VR_3PT safety filters, preflight
checks, or other safety mechanisms. Do not promise arbitrary binary flags such
as `--disable-dex3-hands` through `deploy.sh`: the current wrapper accepts only
its documented options and treats unknown flags as interface values. Robot-
specific binary flags require a separately validated direct invocation or a
future wrapper change and are outside this docs-only task.

### 3. Set Up PICO Teleoperation

Keep this section body intentionally empty for the first version, as requested.
The heading remains in the ordered workflow so future PICO hardware setup can
be added without renumbering the runbook. The Reader Contract carries the
mandatory PICO setup prerequisite and links the full existing setup guide.

### 4. Run the Camera Server on PC2

Define two mutually exclusive camera startup branches. The reader chooses one
while running `install_scripts/install_camera_server.sh`:

- **Foreground:** answer `n` to the systemd prompt, confirm the service is not
  active, then run `python -m gear_sonic.camera.composed_camera` with
  `<EGO_CAMERA_TYPE>`, optional `<EGO_CAMERA_DEVICE_ID>`, and `<CAMERA_PORT>`.
- **systemd:** answer `y`, supply the camera configuration and port to the
  installer, then use the generated service only. Do not also run the foreground
  command.

For an empty device-ID value, omit the entire device-ID option rather than
passing an empty shell argument.

Verification covers:

- camera discovery before startup;
- server logs without frame timeout errors;
- a listening TCP port on PC2;
- reachability from the workstation.

Camera-specific flags must come from the supported command's `--help` output.
The onboarding page must not claim that an unvalidated camera model is generally
supported.

### 5. Run Workstation Processes

Use three labeled terminals, all launched from `<WORKSTATION_REPO_DIR>`:

1. `.venv_teleop`: run `pico_manager_thread_server.py` in manager/controller
   mode on local port 5556 and explicitly subscribe to measured state with
   `--zmq_feedback_host <PC2_IP> --zmq_feedback_port 5557`.
2. `.venv_data_collection`: run `run_data_exporter.py` with the local manager
   endpoint, `<PC2_IP>:5557` state endpoint, PC2 camera endpoint, task prompt,
   and dataset name.
3. `.venv_data_collection`: run `run_camera_viewer.py` against the same PC2
   camera endpoint.

Document the exporter recording controls already implemented by the manager:
start or save with `Left Grip + A`, and discard with `Left Grip + B`.

Before policy engagement, require exact directional checks:

1. On the workstation, `ss` must show the PICO manager listening on `*:5556`.
2. From PC2, a bounded `nc -zvw 3 <WORKSTATION_IP> 5556` check must succeed.
3. On PC2, `ss` must show the deploy feedback publisher listening on `*:5557`.
4. From the workstation, `nc -zvw 3 <PC2_IP> 5557` must succeed.
5. From the workstation, `nc -zvw 3 <PC2_IP> <CAMERA_PORT>` must succeed.
6. The manager must report received robot feedback rather than operating only
   from unset/frozen measured targets.

Only after these outcomes, the safety operator may press `]` in the deploy
terminal to engage the policy. The runbook must name the expected success text
or exit code for each command and must not suggest policy engagement when any
check fails. It must keep the deploy terminal focused and remind the safety
operator that `O` is the primary software stop.

## Failure Handling and Safe Shutdown

Add a compact troubleshooting table for the highest-probability failures:

- Git LFS pointer files instead of model assets.
- Wrong virtual environment active.
- PC2 endpoint replaced with `localhost` on the workstation.
- Ports 5556, 5557, or `<CAMERA_PORT>` not listening or blocked.
- Camera not detected or frame timeouts.
- PICO manager and deploy using incompatible ZMQ host or port values.

Shutdown guidance prioritizes robot safety: stop or emergency-stop robot control
using the documented deploy controls. Before interrupting the exporter, either
save the active episode with `Left Grip + A` and wait for `Finished saving
episode`/idle confirmation, or discard it with `Left Grip + B` and wait for
`Discarded episode`. An interrupt with buffered frames marks that episode as
discarded. After the exporter is idle, stop camera viewing, PICO management, and
the foreground camera server as appropriate. A systemd camera service is
managed with `systemctl` rather than killed as an arbitrary process.

## Verification

Before considering the page complete:

1. Run `make -C docs html` and require exit status zero.
2. Require
   `docs/build/html/getting_started/newcomer_onboarding.html` to be a non-empty
   file and check that the source page is present in the Getting Started
   toctree.
3. Search for accidental real IP addresses, usernames, home-directory paths,
   and secrets.
4. Confirm every configuration token used by a command appears in the token
   table.
5. Confirm every command block names its machine and working directory.
6. Compare CLI flags against the current scripts or their `--help` output.
7. Confirm the exact deployment model files with non-empty-file checks.
8. Confirm the runbook contains both directional port checks before policy
   engagement.
9. Review the page for a safe startup order, episode-aware shutdown, and an
   explicit process shutdown path.

Hardware execution is outside documentation verification: do not start the real
robot, PICO hardware, or camera merely to validate a docs-only change.

## Scope Boundaries

This version does not cover PICO hardware installation or calibration, SONIC
training, checkpoint export, simulation, VLA fine-tuning, robot network
provisioning, or camera-driver development. It links to existing detailed pages
where useful without reproducing them.
