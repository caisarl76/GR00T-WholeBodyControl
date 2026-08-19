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
table before running any commands.

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

Commands are runnable after the reader replaces the documented configuration
tokens. The token table includes:

- `<PC2_IP>`
- `<PC2_USER>`
- `<PC2_REPO_DIR>`
- `<WORKSTATION_REPO_DIR>`
- `<ROBOT_NETWORK_INTERFACE>`
- `<CAMERA_PORT>`
- `<TASK_PROMPT>`
- `<DATASET_NAME>`

Angle-bracket values are explicitly described as user-supplied configuration,
not unfinished documentation.

## Runtime Topology

```text
Laptop/workstation                              PC2
------------------                              ---
PICO manager          -- planner commands -->   GEAR-SONIC deploy
localhost:5556                                  state publisher :5557

Data exporter         <-- robot state -------   GEAR-SONIC deploy
                      <-- camera frames ------   camera server :<CAMERA_PORT>

Camera viewer         <-- camera frames ------   camera server :<CAMERA_PORT>
```

The data exporter connects to the local PICO manager, PC2's robot-state
publisher, and PC2's camera server. The guide calls out these three endpoints
because mixing up `localhost` and `<PC2_IP>` is a likely newcomer error.

## Document Structure

### 0. Clone the GR00T Repository

Show Git LFS prerequisites and clone commands for both the workstation and
PC2. Include `git lfs pull` and `python check_environment.py` where applicable.
The guide does not embed credentials or assume identical home-directory paths.

### 1. Install the Virtual Environments

Explain that the repository intentionally uses separate environments. Install
only the environments needed by this workflow:

- Workstation: `.venv_teleop` via `install_scripts/install_pico.sh`.
- Workstation: `.venv_data_collection` via
  `install_scripts/install_data_collection.sh`.
- PC2: `.venv_camera` via `install_scripts/install_camera_server.sh`.
- PC2: C++ deployment dependencies and build setup under
  `gear_sonic_deploy`; this is not a Python virtual environment.

Provide lightweight import or `--help` checks after installation. Warn that the
install scripts recreate their target virtual environments, so readers should
not rerun them casually when an environment contains local packages.

### 2. Deploy GEAR-SONIC on PC2

Use the repository-supported deployment flow rather than duplicating all model
paths in the onboarding guide:

1. SSH to PC2 and enter `gear_sonic_deploy`.
2. Source `scripts/setup_env.sh` and build with `just build` during one-time
   setup.
3. Start the real-robot flow with `bash deploy.sh real` or the configured
   network interface supported by the script.
4. Confirm the deploy preflight, operator safety zone, start-control key, and
   emergency-stop key.
5. Verify that robot state is published on TCP port 5557.

The guide does not recommend disabling CRC, VR_3PT safety filters, preflight
checks, or other safety mechanisms. If the lab's hand configuration requires a
deployment option such as `--disable-dex3-hands`, the reader is directed to use
only the configuration approved for that robot instead of assuming it globally.

### 3. Set Up PICO Teleoperation

Keep this section body intentionally empty for the first version, as requested.
The heading remains in the ordered workflow so future PICO hardware setup can
be added without renumbering the runbook.

### 4. Run the Camera Server on PC2

Provide a foreground command based on
`python -m gear_sonic.camera.composed_camera`, using the configured camera type,
device identifier when required, and `<CAMERA_PORT>`. Also describe the
repository's optional systemd installation path for persistent startup.

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
   mode on local port 5556.
2. `.venv_data_collection`: run `run_data_exporter.py` with the local manager
   endpoint, `<PC2_IP>:5557` state endpoint, PC2 camera endpoint, task prompt,
   and dataset name.
3. `.venv_data_collection`: run `run_camera_viewer.py` against the same PC2
   camera endpoint.

Document the exporter recording controls already implemented by the manager:
start or save with `Left Grip + A`, and discard with `Left Grip + B`.

## Failure Handling and Safe Shutdown

Add a compact troubleshooting table for the highest-probability failures:

- Git LFS pointer files instead of model assets.
- Wrong virtual environment active.
- PC2 endpoint replaced with `localhost` on the workstation.
- Ports 5556, 5557, or `<CAMERA_PORT>` not listening or blocked.
- Camera not detected or frame timeouts.
- PICO manager and deploy using incompatible ZMQ host or port values.

Shutdown guidance prioritizes robot safety: stop or emergency-stop robot control
using the documented deploy controls, then stop data export, camera viewing,
PICO management, and the foreground camera server as appropriate. A systemd
camera service is managed with `systemctl` rather than killed as an arbitrary
process.

## Verification

Before considering the page complete:

1. Build the Sphinx documentation using the repository's documented docs build
   command.
2. Check that the new page is present in the Getting Started toctree.
3. Search for accidental real IP addresses, usernames, home-directory paths,
   and secrets.
4. Confirm every configuration token used by a command appears in the token
   table.
5. Confirm every command block names its machine and working directory.
6. Compare CLI flags against the current scripts or their `--help` output.
7. Review the page for a safe startup order and an explicit shutdown path.

Hardware execution is outside documentation verification: do not start the real
robot, PICO hardware, or camera merely to validate a docs-only change.

## Scope Boundaries

This version does not cover PICO hardware installation or calibration, SONIC
training, checkpoint export, simulation, VLA fine-tuning, robot network
provisioning, or camera-driver development. It links to existing detailed pages
where useful without reproducing them.
