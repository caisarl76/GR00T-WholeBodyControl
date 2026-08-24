# Newcomer Onboarding Teammate Handover

Date: 2026-08-24

## Outcome and Machine Topology

The newcomer workflow is ready for teammate review and supervised lab use. The
operational instructions live in
`docs/source/getting_started/newcomer_onboarding.md`; do not execute this
handover as a second runbook.

The final topology is:

- PC2 runs the GEAR-SONIC C++ deployment and camera server.
- The laptop/workstation runs the PICO manager, data exporter, and camera
  viewer.
- Planner commands travel from workstation port 5556 to PC2.
- Robot configuration and measured state travel from PC2 port 5557 to the
  workstation.
- Camera frames travel from PC2 port `<CAMERA_PORT>` to the workstation.

The final runbook intentionally overrides the older data-collection page where
that page places the C++ deployment on the workstation.

## Operational Authority and Superseded Instructions

The source-of-truth order is:

1. `docs/source/getting_started/newcomer_onboarding.md` governs all real
   execution.
2. `docs/source/getting_started/gear_sonic_deployment_9c0ff22.sha256` governs
   the four deployment-artifact digests.
3. `docs/superpowers/specs/2026-08-19-newcomer-onboarding-design.md` preserves
   approved rationale but does not override the runbook.
4. `docs/superpowers/plans/2026-08-20-newcomer-onboarding.md` is historical
   implementation intent and is superseded wherever it conflicts with the
   final runbook.

These safety-critical old-plan instructions are explicitly superseded:

| Topic | Superseded plan behavior | Governing final behavior |
| --- | --- | --- |
| `ACTUATE` readiness | The numbered plan can be read as starting deployment before camera and manager readiness. | Complete Sections 0 and 1, require the Section 4 live-frame content `PASS`, start and probe the Section 5 manager from PC2, then return to Section 2 and type `ACTUATE`. |
| Episode close | The historical normal shutdown stops deployment before resolving an active episode. | While all streams are healthy, finish or discard the active episode and require exporter idle; deployment is then the first process stopped. |
| Deployment stop | The historical plan can be read as treating a hardware stop as sufficient process shutdown. | After any hardware stop, keep it secured, terminate deploy, and prove the deploy process and TCP port 5557 are absent before remediation, restart, or robot-power restoration. |
| Remediation order | The historical plan uses generic stop-and-resolve wording. | Post-`ACTUATE`, pre-engagement failure requires confirmed uppercase-`O` shutdown or full independent-stop cleanup. Planner-start or measured-state failure requires the independent hardware stop and the same cleanup proof before troubleshooting. |

## Stable Source Milestones

These values were knowable before this handover commit and are safe to record:

- PICO navigation prerequisite:
  `4db4795002bef8918537367a3739963935187acc`.
- Original onboarding implementation tip:
  `e2b923d1e986bb9019fa21295c1b16c81226e4dd`.
- Initial handover-design commit:
  `8cd095d5b4fc9f8a47930ea7b494bf5687d8d09c`.
- Approved handover-design tip:
  `7a2d0036e86f7ee380912c322c096b64621d0369`.
- Handover implementation-plan commit:
  `67eb3aa78732c103205376832187d94524a9232b`.
- Repository-source correction commit:
  `f9fac523e38e09f34ca4b21c9e8e610d19051e31`.

This document does not contain its own future source SHA or the future
published SHA. Cherry-picking changes commit IDs, so both final values and the
source-to-published map belong only in the post-push PR body.

No hardware was started while producing or validating these milestones.

## Deviation Ledger

The historical implementation plan promised documentation-only changes. The
approved publication includes the following bounded deviations. There is no
repository `CODEOWNERS` file; `caisarl76` owns PR follow-through until the
role-based maintainers accept their areas.

| Unplanned file | Source commit(s) | Behavioral effect and rationale | Evidence | Role owner |
| --- | --- | --- | --- | --- |
| `decoupled_wbc/control/teleop/streamers/pico_streamer.py` | `4db4795002bef8918537367a3739963935187acc` | Corrects PICO forward, strafe, and yaw input signs. The published manager workflow must not silently reverse real-controller behavior. | `test_pico_streamer_navigation.py` passes in the 14-test suite. | Decoupled-WBC teleoperation maintainers. |
| `gear_sonic/scripts/pico_manager_thread_server.py` | `4db4795002bef8918537367a3739963935187acc` | Corrects accumulated yaw and facing-relative planner movement used by the runbook's manager. | `test_pico_planner_joystick.py` passes in the 14-test suite. | GEAR-SONIC PICO/manager maintainers. |
| `gear_sonic/tests/test_pico_planner_joystick.py` | `4db4795002bef8918537367a3739963935187acc` | Adds yaw-direction and facing-relative movement regression coverage only. | Included in the pinned 14-test suite. | GEAR-SONIC PICO test maintainers. |
| `gear_sonic/tests/test_pico_streamer_navigation.py` | `4db4795002bef8918537367a3739963935187acc` | Adds forward, strafe, and yaw sign regression coverage only. | Included in the pinned 14-test suite. | Decoupled-WBC teleoperation test maintainers. |
| `gear_sonic/camera/composed_camera.py` | `29d680f9d0efa369cc7b3164a8118ab01884b781`, `a3547641cf688ff08afc3f003bdee57169b40beb` | Makes foreground shutdown close workers and ZMQ resources on runtime interruption, partial construction, bind failure, or other startup failure; cleanup is idempotent and unexpected errors still propagate. This makes the documented `Ctrl+C` path terminate. | Six focused shutdown tests and the viewer regression pass. | GEAR-SONIC camera/runtime maintainers. |
| `gear_sonic/tests/test_composed_camera_server_shutdown.py` | `29d680f9d0efa369cc7b3164a8118ab01884b781`, `a3547641cf688ff08afc3f003bdee57169b40beb` | Adds coverage for steady-state and partial-construction cleanup, exact stop-log order, idempotence, and unexpected-error propagation. | Six focused tests pass; seven pass with the viewer regression. | GEAR-SONIC camera test maintainers. |
| `gear_sonic_deploy/scripts/setup_env.sh` | `b1a4f62deff3aaf8e7bc36c4c4cf995828c0c6fc` | Replaces `find ... | head` with `find ... -print -quit`, avoiding a `pipefail`/SIGPIPE false abort during runtime-only CUDA discovery. | Shell syntax, exact safe lookup, and retired-pipeline absence checks pass. | GEAR-SONIC deployment-environment maintainers. |

The final runbook safety sequencing changed in `868bbdf`, `b1a4f62`, and
`e2b923d`. The later repository-source correction changes clone/fetch selection
only; it does not change robot runtime behavior.

## Lab-Owned Inputs

The lab supplies all 15 intentional tokens outside committed documentation:

```text
<PC2_IP>
<PC2_USER>
<PC2_REPO_DIR>
<WORKSTATION_IP>
<WORKSTATION_REPO_DIR>
<ROBOT_NETWORK_INTERFACE>
<REPOSITORY_URL>
<REPO_REVISION>
<TENSORRT_ROOT>
<EGO_CAMERA_TYPE>
<EGO_CAMERA_DEVICE_ID>
<CAMERA_PORT>
<TASK_PROMPT>
<DATASET_NAME>
<LAB_APPROVED_HARDWARE_ESTOP_PROCEDURE>
```

For this draft, `<REPOSITORY_URL>` resolves to
`https://github.com/caisarl76/GR00T-WholeBodyControl.git`. Resolve
`<REPO_REVISION>` from the verified remote head using the command in
**Publication Contract** below; use the identical 40-character result on both
machines.

The robot owner must also supply and rehearse the independent hardware-stop
procedure, confirm the physical robot has no Dex3/Inspire hands before using
the no-hands launch, authorize network access, and identify the validated camera
backend and device. None of these values belongs in committed documentation.

## Verification Evidence

All verification below was non-actuating. No robot, PICO service, deployment
binary, camera server, manager, exporter, or viewer was started.

### Pinned Environment

The worktree first restored the Git LFS-managed XRoboToolkit x86 shared object;
its content SHA-256 was
`2348e1b0bf7f05cd13d95d628f04237b6b9fc50c6008b82f9bf4b046fc9373e6`.
This is why the runbook's clone sequence requires `git lfs pull` before the
installer.

Run from repository root:

```bash
set -euo pipefail
bash install_scripts/install_pico.sh
uv pip install --python "$PWD/.venv_teleop/bin/python" \
  'pytest==9.0.3' \
  'ruff==0.15.20'
test "$(.venv_teleop/bin/python -m pytest --version)" = 'pytest 9.0.3'
test "$(.venv_teleop/bin/ruff --version)" = 'ruff 0.15.20'
```

Recorded result: exit 0 with Python 3.10.20, pytest 9.0.3, and Ruff 0.15.20.

### PICO, Camera Shutdown, and Viewer Tests

```bash
set -euo pipefail
PYTHONPATH="$PWD" \
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.venv_teleop/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_composed_camera_server_shutdown.py \
  gear_sonic/tests/test_run_camera_viewer.py \
  gear_sonic/tests/test_pico_planner_joystick.py \
  gear_sonic/tests/test_pico_streamer_navigation.py
```

Recorded result: exit 0, `14 passed in 3.42s`, with 124 deprecation warnings
and zero test failures.

### Byte-Compilation and Ruff

```bash
set -euo pipefail
PYTHONDONTWRITEBYTECODE=1 .venv_teleop/bin/python -m py_compile \
  decoupled_wbc/control/teleop/streamers/pico_streamer.py \
  gear_sonic/camera/composed_camera.py \
  gear_sonic/scripts/pico_manager_thread_server.py \
  gear_sonic/tests/test_composed_camera_server_shutdown.py \
  gear_sonic/tests/test_pico_planner_joystick.py \
  gear_sonic/tests/test_pico_streamer_navigation.py
.venv_teleop/bin/ruff check --no-cache \
  decoupled_wbc/control/teleop/streamers/pico_streamer.py \
  gear_sonic/camera/composed_camera.py \
  gear_sonic/tests/test_composed_camera_server_shutdown.py \
  gear_sonic/tests/test_pico_planner_joystick.py \
  gear_sonic/tests/test_pico_streamer_navigation.py
.venv_teleop/bin/ruff check --no-cache --ignore I001 \
  gear_sonic/scripts/pico_manager_thread_server.py
```

Recorded result: byte-compilation exited 0 and Ruff printed
`All checks passed!` twice. The manager invocation excludes only the
pre-existing whole-file import-order finding; `4db4795` does not touch that
import block.

### Deployment Environment

```bash
set -euo pipefail
bash -n gear_sonic_deploy/scripts/setup_env.sh
test "$(rg -nF 'cuda_so_path=$(find /usr -name libcuda.so.1 -print -quit 2>/dev/null)' \
  gear_sonic_deploy/scripts/setup_env.sh | wc -l)" -eq 1
if rg -nF 'find /usr -name libcuda.so.1 2>/dev/null | head -n1' \
  gear_sonic_deploy/scripts/setup_env.sh; then
  exit 1
else
  test "$?" -eq 1
fi
```

Recorded result: exit 0, one safe lookup, and no retired pipeline.

### Runbook Fences and Manifest

The runbook contains 32 Bash fences and 5 embedded Python heredocs. Every Bash
fence passes `bash -n`, and every Python body passes `ast.parse`.

The tracked checksum manifest contains exactly four lowercase SHA-256/path
pairs in the required order. After the pinned model files were available, the
content check was:

```bash
set -euo pipefail
LC_ALL=C sha256sum --check \
  docs/source/getting_started/gear_sonic_deployment_9c0ff22.sha256
```

Recorded result: exit 0 with `OK` for `model_encoder.onnx`,
`model_decoder.onnx`, `observation_config.yaml`, and `planner_sonic.onnx`.
Manifest-structure validation remains always runnable when those downloads are
absent; content verification intentionally fails if any artifact is missing.

### Placeholder, Blank Section, and Privacy Gate

```bash
set -euo pipefail
python3 - <<'PY'
from pathlib import Path
import re

paths = [
    Path("docs/source/getting_started/newcomer_onboarding.md"),
    Path("docs/superpowers/specs/2026-08-21-newcomer-onboarding-handover-design.md"),
    Path("docs/superpowers/progress/2026-08-21-newcomer-onboarding-handover.md"),
]
documents = {path: path.read_text() for path in paths}
source = documents[paths[0]]
expected = {
    "<PC2_IP>",
    "<PC2_USER>",
    "<PC2_REPO_DIR>",
    "<WORKSTATION_IP>",
    "<WORKSTATION_REPO_DIR>",
    "<ROBOT_NETWORK_INTERFACE>",
    "<REPOSITORY_URL>",
    "<REPO_REVISION>",
    "<TENSORRT_ROOT>",
    "<EGO_CAMERA_TYPE>",
    "<EGO_CAMERA_DEVICE_ID>",
    "<CAMERA_PORT>",
    "<TASK_PROMPT>",
    "<DATASET_NAME>",
    "<LAB_APPROVED_HARDWARE_ESTOP_PROCEDURE>",
}
for path, text in documents.items():
    found = set(re.findall(r"<[A-Z][A-Z0-9_]*>", text))
    assert found == expected, {
        "path": str(path),
        "missing": sorted(expected - found),
        "extra": sorted(found - expected),
    }
headings = re.findall(r"^## ([0-5])\. .+$", source, re.M)
assert headings == list("012345"), headings
section_3 = re.search(
    r"^## 3\. Set Up PICO Teleoperation\n(.*?)^## 4\. ", source, re.M | re.S
).group(1)
assert section_3 == "", repr(section_3)
private_patterns = [
    r"/" + "home" + r"/",
    r"(?<![0-9.])192\.168(?:\.[0-9]{1,3}){2}(?![0-9.])",
    r"(?<![0-9.])10(?:\.[0-9]{1,3}){3}(?![0-9.])",
    r"(?<![0-9.])172\.(?:1[6-9]|2[0-9]|3[01])(?:\.[0-9]{1,3}){2}(?![0-9.])",
]
credential_patterns = [
    r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",
    r"\bgithub_pat_[A-Za-z0-9_]{50,}\b",
    r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"(?im)^\s*(?:password|passwd|api[_-]?key|access[_-]?token|secret)"
    r"\s*[:=]\s*['\"]?[^\s<][^#\n]*$",
]
for path, text in documents.items():
    for pattern in private_patterns + credential_patterns:
        assert not re.search(pattern, text), (path, pattern)
    for marker in ("TO" + "DO", "T" + "BD", "FIX" + "ME"):
        assert marker not in text, (path, marker)
print("PASS: placeholders, blank Section 3, and privacy")
PY
```

Recorded pre-commit result: exit 0 and the exact `PASS` line. Each document
independently contains exactly the 15 allowlisted tokens; Section 3 remains
byte-empty; no declared privacy, credential, or unfinished-marker pattern
matches.

### Anchored Toctree and Forced Sphinx Build

The structural parser selects only the `toctree` whose caption is
`Getting Started` and requires exactly one
`getting_started/newcomer_onboarding` entry.

Recorded result: `PASS: newcomer onboarding is in the Getting Started toctree
exactly once`.

The genuinely forced build command is:

```bash
set -euo pipefail
BUILD_LOG="$(mktemp)"
trap 'rm -f "$BUILD_LOG"' EXIT
make -C docs \
  SPHINXBUILD=../.venv_docs/bin/sphinx-build \
  SPHINXOPTS='-E -a' \
  html 2>&1 | tee "$BUILD_LOG"
test -s docs/build/html/getting_started/newcomer_onboarding.html
if rg -n 'newcomer_onboarding.*WARNING|WARNING.*newcomer_onboarding' "$BUILD_LOG"; then
  exit 1
else
  test "$?" -eq 1
fi
```

Recorded result: exit 0, nonempty onboarding HTML, 10 unrelated/offline
inventory warnings, and no warning naming the onboarding page.

## Teammate Onboarding Checklist

- [ ] Read the final runbook and the real-robot safety guide; treat the old
  implementation plan as historical only.
- [ ] Populate all 15 lab values outside committed documentation.
- [ ] Resolve the remote head and verify both machines use the same exact
  40-character revision and repository origin.
- [ ] Confirm all four pinned artifacts and the platform-specific TensorRT and
  JetPack gates.
- [ ] Complete the existing VR setup guide, XRoboToolkit PC service, headset
  networking, and tracker calibration even though runbook Section 3 remains
  blank by request.
- [ ] Validate the exact deployment sequence in MuJoCo and rehearse uppercase
  `O`, PICO stop, and the independent robot-owner hardware-stop path.
- [ ] Confirm the physical no-Dex3/Inspire-hands configuration before using the
  direct no-hands launch.
- [ ] Conduct the first real run only with the robot owner, harness, cleared
  zone, spotter, and independent stop ready before pressing Enter for the
  actuated initialization ramp.
- [ ] During normal collection, finalize or discard the active episode while
  streams remain healthy, require exporter idle, then stop deployment first.
- [ ] After any fault-triggered hardware stop, keep it secured and complete the
  runbook's deploy-process and port-5557 cleanup proof before remediation.

## Publication Contract

Publish exactly one draft PR with:

- repository: `caisarl76/GR00T-WholeBodyControl`;
- base branch: `vr3pt-cleanup-minimal-official` at
  `af76fae68930b4a9276af768015fc81fbcedc344`;
- remote head: `refs/heads/docs/newcomer-onboarding`;
- local publication branch: `publish/newcomer-onboarding`; and
- no force push.

Resolve the lab revision only after the remote head exists:

```bash
set -euo pipefail
REPOSITORY_URL='https://github.com/caisarl76/GR00T-WholeBodyControl.git'
REMOTE_HEAD='refs/heads/docs/newcomer-onboarding'
REPO_REVISION="$(git ls-remote --exit-code --heads \
  "$REPOSITORY_URL" "$REMOTE_HEAD" | awk 'NR == 1 {print $1}')"
test "${#REPO_REVISION}" -eq 40
case "$REPO_REVISION" in
  *[!0-9a-f]*) exit 1 ;;
esac
printf 'PASS: set <REPO_REVISION> to %s from %s\n' \
  "$REPO_REVISION" "$REMOTE_HEAD"
```

Before publication the absent head makes this command fail closed. After push,
the lab uses the printed value as `<REPO_REVISION>` on both machines. If the
fork base or remote head differs from the values above, stop rather than
overwriting history.

The post-push PR body—not this handover—must record the final source-handover
SHA, published SHA, source-to-published commit map, and post-commit verification
results.
