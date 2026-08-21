# Newcomer Onboarding Teammate Handover Design

Date: 2026-08-21

## Purpose

Create a concise repository-local handover for the teammate who will review,
adopt, and maintain the PC2/workstation newcomer-onboarding workflow. The
handover explains what actually shipped, calls out every departure from the
documentation-only implementation plan, and gives reproducible verification
and publication instructions. It links to the user-facing runbook instead of
copying that runbook's commands.

## Destination and Audience

Write the handover at:

```text
docs/superpowers/progress/2026-08-21-newcomer-onboarding-handover.md
```

The audience is a repository teammate who must review the implementation,
supply lab-owned configuration, help a newcomer execute the final runbook, and
own follow-up in the relevant documentation, camera-runtime, or deployment
environment area.

## Authority and Supersession

The documents have this authority order:

1. `docs/source/getting_started/newcomer_onboarding.md` is the sole operational
   source of truth for real execution.
2. `docs/source/getting_started/gear_sonic_deployment_9c0ff22.sha256` is the
   source of truth for the four pinned deployment-artifact digests.
3. `docs/superpowers/specs/2026-08-19-newcomer-onboarding-design.md` records the
   approved design rationale but does not override the final runbook.
4. `docs/superpowers/plans/2026-08-20-newcomer-onboarding.md` is historical
   implementation intent. It is not an operational source of truth and is
   superseded wherever it conflicts with the final runbook or the deviation
   ledger below.

The handover must mark these old-plan instructions as superseded explicitly:

| Topic | Superseded plan instruction | Governing final behavior |
| --- | --- | --- |
| `ACTUATE` readiness | The numbered implementation sequence places deployment before camera and manager setup. | Complete Sections 0 and 1, complete Section 4 through the live-frame content `PASS`, start and probe the Section 5 manager from PC2, and only then return to Section 2 and type `ACTUATE`. |
| Episode close | The old normal shutdown stops deployment before saving or discarding an active episode. | While all streams are healthy, finish or discard the active episode and require exporter idle; the robot deployment is then the first **process** stopped. |
| Deployment stop | The old plan treats an independent hardware stop as sufficient robot shutdown. | After any hardware stop, keep the stop secured, terminate deploy, and prove both the documented process and TCP port 5557 are absent before remediation, restart, or robot-power restoration. |
| Remediation order | The old plan uses generic stop-and-resolve language after a failed gate. | A post-`ACTUATE`, pre-engagement failure requires confirmed uppercase-`O` shutdown or the full independent-stop cleanup. Planner-start or measured-state failure uses the independent hardware stop and the same process-cleanup proof. Troubleshooting begins only afterward. |

## Implementation Identity

The handover distinguishes the supporting PICO correction and four milestones
rather than calling the current branch tip the implementation tip:

- `4db4795002bef8918537367a3739963935187acc` is the validated **PICO
  navigation prerequisite**. It corrects real joystick yaw, forward, and
  strafe directions used by the manager that the runbook launches. Although it
  predates the onboarding design, the publication transplant must include it
  because the published fork base does not.

- `e2b923d1e986bb9019fa21295c1b16c81226e4dd` is the source branch's final
  original onboarding **implementation tip**. It includes the final safety
  sequencing, runtime cleanup, tests, and environment hardening, and its tree
  also contains `4db4795` through ancestry.
- `8cd095d5b4fc9f8a47930ea7b494bf5687d8d09c` is the initial **handover-design
  commit**. It is documentation about the later handover and is not part of the
  implementation-tip claim. Subsequent review-fix commits to this design must
  be identified as design-only history.
- A future **publication-contract correction commit**, created only after this
  revised design is approved, adds `<REPOSITORY_URL>` to the runbook's
  configuration and makes both clone/fetch flows use it. That functional docs
  change is separate from the original implementation-tip claim.
- The **future handover document commit** is created after the
  publication-contract correction and the approved implementation plan. The
  completed handover and PR body record it separately from `4db4795`,
  `e2b923d`, `8cd095d`, and the publication-contract correction.

No hardware was started while producing or validating any of these commits.

## Deviation Ledger

The old plan says the implementation changes documentation only. The final
implementation deliberately exceeds that scope. The handover includes the
following ledger without hiding the runtime files in a generic summary.

There is no repository `CODEOWNERS` file. Until reviewers assign named owners,
`caisarl76` owns branch follow-through and review resolution; domain acceptance
belongs to the role-based maintainers below.

| Unplanned file | Source commits | Behavioral effect | Rationale | Verification evidence | Maintainer ownership |
| --- | --- | --- | --- | --- | --- |
| `decoupled_wbc/control/teleop/streamers/pico_streamer.py` | `4db4795002bef8918537367a3739963935187acc` | Maps PICO left-stick forward/strafe and right-stick yaw to the expected navigation signs. | The final runbook launches PICO teleoperation; publishing it against a base without the validated real-controller direction fix would silently change operator behavior. | `gear_sonic/tests/test_pico_streamer_navigation.py` passes as part of the pinned 14-test suite below. | Decoupled-WBC teleoperation maintainers; `caisarl76` owns the PR until acceptance. |
| `gear_sonic/scripts/pico_manager_thread_server.py` | `4db4795002bef8918537367a3739963935187acc` | Corrects accumulated yaw sign and rotates left-stick planner movement into the commanded facing frame. | The manager is the runbook's live planner-command source, so its tested joystick contract is a publication prerequisite. | `gear_sonic/tests/test_pico_planner_joystick.py` passes as part of the pinned 14-test suite below. | GEAR-SONIC PICO/manager maintainers; `caisarl76` owns the PR until acceptance. |
| `gear_sonic/tests/test_pico_planner_joystick.py` | `4db4795002bef8918537367a3739963935187acc` | Adds regression coverage only; it verifies yaw direction and facing-relative forward/strafe planner movement. | The manager correction needs executable sign and coordinate-frame evidence. | Included in the pinned 14-test suite below. | GEAR-SONIC PICO test maintainers; manager reviewers approve alongside production code. |
| `gear_sonic/tests/test_pico_streamer_navigation.py` | `4db4795002bef8918537367a3739963935187acc` | Adds regression coverage only; it verifies PICO forward, strafe, and yaw input signs. | The streamer correction needs executable controller-mapping evidence. | Included in the pinned 14-test suite below. | Decoupled-WBC teleoperation test maintainers; streamer reviewers approve alongside production code. |
| `gear_sonic/camera/composed_camera.py` | `29d680f9d0efa369cc7b3164a8118ab01884b781`, `a3547641cf688ff08afc3f003bdee57169b40beb` | Foreground server shutdown now closes camera workers and ZMQ resources on steady-loop interruption, construction interruption, bind failure, or other startup failure; cleanup is idempotent and unexpected errors still propagate. | The documented foreground `Ctrl+C` shutdown was not runnable because non-daemon camera workers could survive, including failures during construction. | Six focused shutdown tests pass; the shutdown tests plus the existing viewer regression report `7 passed`. | GEAR-SONIC camera/runtime maintainers; `caisarl76` owns the PR until acceptance. |
| `gear_sonic/tests/test_composed_camera_server_shutdown.py` | `29d680f9d0efa369cc7b3164a8118ab01884b781`, `a3547641cf688ff08afc3f003bdee57169b40beb` | Adds regression coverage only; it does not change production behavior. It covers runtime and construction interrupts, bind failure, partial thread/context cleanup, idempotence, exact log order, and unexpected-error propagation. | The runtime change needs executable evidence for both the steady-state and partial-construction paths. | `6 passed` in the focused file; `7 passed` with `gear_sonic/tests/test_run_camera_viewer.py`. | GEAR-SONIC camera test maintainers; camera/runtime reviewers approve alongside production code. |
| `gear_sonic_deploy/scripts/setup_env.sh` | `b1a4f62deff3aaf8e7bc36c4c4cf995828c0c6fc` | Replaces `find /usr ... \| head -n1` with `find /usr ... -print -quit`, preserving first-match behavior without a `pipefail`/SIGPIPE false abort. | The final runbook sources this script under `errexit` and `pipefail`; a successful runtime-only CUDA discovery must not be reported as setup failure. | `bash -n` exits 0, the retired pipeline is absent, and the exact `-print -quit` command is present. | GEAR-SONIC deployment-environment maintainers; `caisarl76` owns the PR until acceptance. |

The final runbook is a planned file, but its safety sequencing differs from the
old plan. Commits `868bbdf`, `b1a4f62`, and `e2b923d` implement the supersession
table above. The future publication-contract correction alters only repository
selection for clone/fetch; it does not alter runtime behavior. The current
handover design and future handover document are post-implementation
documentation requested by the user.

## Required Handover Content

The handover contains these sections:

1. **Outcome and machine topology** — PC2 owns GEAR-SONIC deploy and camera;
   the workstation owns the PICO manager, exporter, and viewer.
2. **Authority and superseded instructions** — the final runbook governs the
   four safety-critical lifecycle topics listed above.
3. **Deviation ledger** — each unplanned file, commit, behavior, rationale,
   test result, and maintainer role.
4. **Lab-owned inputs** — the intentional placeholders, exact publication
   repository URL, robot-owner hardware stop procedure, no-hands confirmation,
   network access, and camera identity.
5. **Reproducible verification** — the exact commands and recorded results in
   the next section.
6. **Teammate onboarding checklist** — document review, lab-value population,
   simulation and stop-path rehearsal, same-revision checks on both machines,
   and a supervised first real run.
7. **Publication context** — source milestones, live fork state, transplant
   strategy, reachable clone contract, target repository/base/head, and PR
   limitations.

The handover must not duplicate the full operational command sequence or add a
PICO procedure. Section 3 of the user-facing runbook remains empty.

## Lab Placeholder Allowlist

The following 15 placeholders are intentional and are the complete allowlist:

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

The handover says these must be supplied by the lab and must never be replaced
with real values in the committed documentation. For this publication,
`<REPOSITORY_URL>` must resolve to
`https://github.com/caisarl76/GR00T-WholeBodyControl.git`, the repository that
will contain `<REPO_REVISION>`; the public URL is recorded in the handover as
the required value, not substituted into the runbook. Any other angle-bracket
token is a verification failure.

## Reproducible Verification Contract

Unless a command says otherwise, its working directory is the root of a
checkout containing the handover commit. All commands are non-actuating.

### Pinned Verification Environment

The PICO installer creates `.venv_teleop`, but it does not install pytest or
Ruff. Create the environment and install the two verification tools at the
exact validated versions before running any Python verification:

```bash
set -euo pipefail
bash install_scripts/install_pico.sh
uv pip install --python "$PWD/.venv_teleop/bin/python" \
  'pytest==9.0.3' \
  'ruff==0.15.20'
test "$(.venv_teleop/bin/python -m pytest --version)" = 'pytest 9.0.3'
test "$(.venv_teleop/bin/ruff --version)" = 'ruff 0.15.20'
```

Expected: exit 0 and both exact version comparisons pass. Record the installer
and pinned-tool result at the future handover document commit. Every later
Python command uses an executable under `.venv_teleop/bin`; bare `python`,
`pytest`, and `ruff` are not accepted.

### PICO, Camera Shutdown, and Viewer Regressions

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

Expected: exit 0 and `14 passed`. A pre-design-review run against the source
tree containing `4db4795` and `e2b923d` reported `14 passed in 1.84s`; rerun
and record the result at the future handover document commit and again on the
publication transplant.

### Python Static Checks

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

Expected: exit 0 and two `All checks passed!` lines from Ruff. The manager-only
invocation excludes its pre-existing whole-file import-order finding; the
`4db4795` patch does not touch that import block. It does not exclude any other
Ruff rule. Rerun and record all three results at the future handover document
commit and on the publication transplant; the historical `e2b923d` result
covered only the two camera files.

### Deployment Environment Syntax and Behavior

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

Expected: exit 0, exactly one safe lookup, and no retired pipeline. Recorded at
`e2b923d`: all three gates passed.

### Runbook Command-Fence Validation

```bash
set -euo pipefail
python3 - <<'PY'
from pathlib import Path
import ast
import re
import subprocess

source = Path("docs/source/getting_started/newcomer_onboarding.md").read_text()
bash_blocks = re.findall(r"```bash\n(.*?)\n```", source, re.S)
python_bodies = re.findall(r"<<'PY'\n(.*?)\nPY", source, re.S)
failures = []
for number, block in enumerate(bash_blocks, start=1):
    result = subprocess.run(["bash", "-n"], input=block, text=True, capture_output=True)
    if result.returncode:
        failures.append((number, result.stderr))
for body in python_bodies:
    ast.parse(body)
assert not failures, failures
assert len(bash_blocks) == 32, len(bash_blocks)
assert len(python_bodies) == 5, len(python_bodies)
print("PASS: 32 Bash fences and 5 Python heredocs")
PY
```

Expected and recorded at `e2b923d`: exit 0 and the exact `PASS` line.

### Manifest Structure: Always Runnable

This validates the tracked manifest without requiring downloaded ONNX files.

```bash
set -euo pipefail
python3 - <<'PY'
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
assert len(lines) == 4, len(lines)
actual_paths = []
for line in lines:
    match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
    assert match, line
    actual_paths.append(match.group(2))
assert actual_paths == expected_paths, actual_paths
print("PASS: checksum manifest structure")
PY
```

Expected and recorded at `e2b923d`: exit 0 and
`PASS: checksum manifest structure`.

### Artifact Checksum Verification: Downloads Required

This separate gate is runnable only after the runbook's pinned Hugging Face
downloads and Git LFS population have created all four files.

```bash
set -euo pipefail
LC_ALL=C sha256sum --check \
  docs/source/getting_started/gear_sonic_deployment_9c0ff22.sha256
```

Expected: exit 0 and four `OK` results. Recorded against the downloaded local
artifacts while reviewing `e2b923d`: all four files matched. Missing artifacts
make this command fail and do not invalidate the always-runnable manifest
structure check.

### Placeholder, Blank-Section, and Privacy Validation

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
    for pattern in private_patterns:
        assert not re.search(pattern, text), (path, pattern)
    for pattern in credential_patterns:
        assert not re.search(pattern, text), (path, pattern)
    for marker in ("TO" + "DO", "T" + "BD", "FIX" + "ME"):
        assert marker not in text, (path, marker)
print("PASS: placeholders, blank Section 3, and privacy")
PY
```

Expected after the publication-contract correction and future handover
document exist: exit 0, each of the three documents independently contains
exactly the 15 allowlisted placeholders, Section 3 has zero body bytes, no
declared private-data or high-confidence credential pattern matches, and the
exact `PASS` line prints. The full-address patterns intentionally avoid false
positives from TensorRT versions such as 10.13 and 10.7. Record this result only
at the future handover document commit and on the transplanted publication
tip. Historical `e2b923d` evidence covered the then-current runbook alone with
its 14-token allowlist; it cannot attest to documents or the repository token
created afterward.

### Getting Started Toctree Membership

This check parses only the `toctree` whose caption is `Getting Started`; a
comment or occurrence under another section cannot satisfy it.

```bash
set -euo pipefail
python3 - <<'PY'
from pathlib import Path

lines = Path("docs/source/index.rst").read_text().splitlines()
blocks = []
index = 0
while index < len(lines):
    if lines[index] != ".. toctree::":
        index += 1
        continue
    end = index + 1
    block = []
    while end < len(lines) and (not lines[end] or lines[end].startswith("   ")):
        block.append(lines[end])
        end += 1
    blocks.append(block)
    index = end

getting_started = [
    block for block in blocks if "   :caption: Getting Started" in block
]
assert len(getting_started) == 1, len(getting_started)
entries = [
    line.strip()
    for line in getting_started[0]
    if line.strip() and not line.lstrip().startswith(":")
]
assert entries.count("getting_started/newcomer_onboarding") == 1, entries
print("PASS: newcomer onboarding is in the Getting Started toctree exactly once")
PY
```

Expected: exit 0 and the exact `PASS` line. Record the result at the future
handover document commit and on the transplanted publication tip.

### Genuinely Forced Sphinx Rebuild

Prerequisite: `.venv_docs` exists and was populated with
`docs/requirements.txt`.

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

`-E -a` discards the cached Sphinx environment and rebuilds every source.
Expected: exit 0, nonempty rendered onboarding HTML, and no warning naming the
onboarding page. Recorded at `e2b923d`: build exit 0 with 10 unrelated existing
or offline-inventory warnings and no onboarding-page warning.

### Git Whitespace and Scope

```bash
set -euo pipefail
git diff --check 1e851175f883dc41299e60dc89f4949eac3ff04d..HEAD
git status --short
```

Expected on the source branch at the final handover commit: both commands exit
0 and `git status` prints nothing. On the transplanted publication branch,
replace the diff base with
`af76fae68930b4a9276af768015fc81fbcedc344`. Record both results in the PR.

## Publication Strategy

### Verified Live State

Read-only verification on 2026-08-21 established:

- repository: `caisarl76/GR00T-WholeBodyControl`;
- published base: `vr3pt-cleanup-minimal-official` at
  `af76fae68930b4a9276af768015fc81fbcedc344`;
- PICO navigation prerequisite:
  `4db4795002bef8918537367a3739963935187acc`, present in the source branch but
  absent from the published base;
- source implementation baseline:
  `1e851175f883dc41299e60dc89f4949eac3ff04d`, 40 commits ahead of the
  published base;
- source implementation tip: `e2b923d`, 51 commits ahead of the published
  base;
- initial handover-design tip: `8cd095d`, 52 commits ahead of the published
  base; and
- no published `docs/newcomer-onboarding` branch.

The upstream `NVlabs/GR00T-WholeBodyControl` repository has no
`vr3pt-cleanup-minimal-official` branch, so this workflow does not create an
upstream PR with a nonexistent base.

### Reachable Clone and Revision Contract

The current runbook hard-codes the upstream NVlabs repository even though the
chosen PR repository is the fork. Before the handover document is written, the
publication-contract correction must update the runbook as follows:

1. Add `<REPOSITORY_URL>` to the Configuration table and export
   `REPOSITORY_URL='<REPOSITORY_URL>'` in every new workstation and PC2 shell.
2. Replace both hard-coded clone URLs with
   `git clone "$REPOSITORY_URL" ...`.
3. After each clone, require
   `test "$(git remote get-url origin)" = "$REPOSITORY_URL"` before fetching
   `<REPO_REVISION>`.
4. State that, for this draft publication, the teammate must set
   `<REPOSITORY_URL>` to
   `https://github.com/caisarl76/GR00T-WholeBodyControl.git` and
   `<REPO_REVISION>` to the final 40-character published head. Neither machine
   may continue with the upstream NVlabs origin unless that exact revision is
   first mirrored there and the handover is revised and revalidated.

After push and before opening the draft PR, prove the advertised remote head is
reachable from the configured repository:

```bash
set -euo pipefail
REPOSITORY_URL='https://github.com/caisarl76/GR00T-WholeBodyControl.git'
PUBLISHED_REVISION="$(git rev-parse publish/newcomer-onboarding)"
REMOTE_REVISION="$(git ls-remote --heads "$REPOSITORY_URL" \
  refs/heads/docs/newcomer-onboarding | awk 'NR == 1 {print $1}')"
test -n "$REMOTE_REVISION"
test "$REMOTE_REVISION" = "$PUBLISHED_REVISION"
test "${#PUBLISHED_REVISION}" -eq 40
printf 'PASS: published revision %s is reachable from %s\n' \
  "$PUBLISHED_REVISION" "$REPOSITORY_URL"
```

Expected: exit 0 and one `PASS` line containing the same 40-character commit
recorded in the handover and PR body. A missing branch, empty result, query
failure, or mismatch blocks the PR.

### Chosen Strategy: Transplant Only Onboarding History

Do **not** push the source branch directly and do **not** update the fork base
with the 40 unrelated commits. Create an isolated publication branch from
`af76fae` and cherry-pick only onboarding-specific history:

1. The PICO navigation prerequisite commit `4db4795`, including its full
   four-file scope and tests.
2. The six onboarding design/plan commits already present before the source
   implementation baseline:
   `ab1aae5`, `c57782f`, `680e9f0`, `026d477`, `7023033`, and `1e85117`.
3. The eleven implementation commits in their existing order: `b2a2860`,
   `bf91b1e`, `3620afc`, `99a3cee`, `59e3c21`, `da5d21c`, `29d680f`,
   `868bbdf`, `a354764`, `b1a4f62`, and `e2b923d`.
4. The handover-design history beginning with `8cd095d` and all reviewed
   revision commits.
5. The future implementation plan at
   `docs/superpowers/plans/2026-08-21-newcomer-onboarding-handover.md`, and the
   then-approved publication-contract correction and handover document commits
   in their source order.

The source branch inherits an unrelated pre-baseline `docs/source/index.rst`
toctree edit. If the onboarding toctree cherry-pick reports an index-context
conflict, resolve only by adding one
`getting_started/newcomer_onboarding` entry to the published base's existing
Getting Started toctree. Do not transplant the unrelated source-branch entry.

Because cherry-picking changes commit identifiers, the handover records the
source milestones above, while the PR body records a source-to-published commit
map. The published tree must match the source branch for every onboarding-owned
file before push.

Publish exactly:

- repository: `caisarl76/GR00T-WholeBodyControl`;
- base: `vr3pt-cleanup-minimal-official` at `af76fae`;
- remote head: `docs/newcomer-onboarding`;
- PR state: draft;
- PR count: one.

Use local publication branch `publish/newcomer-onboarding` and push it as
`publish/newcomer-onboarding:docs/newcomer-onboarding`. Since the remote head
is absent, no force push is permitted or needed. If live base/head state
changes, stop and revise this strategy instead of overwriting remote history.

Before push, compare the onboarding-owned trees from the repository root:

```bash
set -euo pipefail
git diff --exit-code \
  docs/newcomer-onboarding \
  publish/newcomer-onboarding \
  -- \
  docs/source/getting_started/newcomer_onboarding.md \
  docs/source/getting_started/gear_sonic_deployment_9c0ff22.sha256 \
  docs/superpowers/specs/2026-08-19-newcomer-onboarding-design.md \
  docs/superpowers/plans/2026-08-20-newcomer-onboarding.md \
  docs/superpowers/specs/2026-08-21-newcomer-onboarding-handover-design.md \
  docs/superpowers/plans/2026-08-21-newcomer-onboarding-handover.md \
  docs/superpowers/progress/2026-08-21-newcomer-onboarding-handover.md \
  decoupled_wbc/control/teleop/streamers/pico_streamer.py \
  gear_sonic/camera/composed_camera.py \
  gear_sonic/scripts/pico_manager_thread_server.py \
  gear_sonic/tests/test_composed_camera_server_shutdown.py \
  gear_sonic/tests/test_pico_planner_joystick.py \
  gear_sonic/tests/test_pico_streamer_navigation.py \
  gear_sonic_deploy/scripts/setup_env.sh
```

Expected: the command exits 0. Then, from the publication worktree, rerun the
anchored **Getting Started Toctree Membership** check above. `docs/source/index.rst`
is checked semantically rather than by whole-file equality because the source
branch inherits unrelated pre-baseline toctree history that must not be
transplanted.

## Acceptance

The written handover is accepted when:

- it contains the authority hierarchy, supersession table, complete deviation
  ledger including all four PICO files, role-based ownership, lab placeholder
  allowlist, source milestone identities, reachable clone contract, and exact
  publication target;
- every repository path and recorded commit exists;
- every command in the reproducible verification contract is rerun at the
  future handover commit and its actual result is recorded;
- no unfinished-marker token, concrete home path, private IPv4 address,
  declared credential pattern, or secret assignment appears;
- no real robot, PICO, deploy binary, camera server, manager, exporter, or
  viewer is started during handover verification; and
- the publication transplant is tree-equivalent for onboarding-owned files,
  passes verification, pushes without force, proves its 40-character head is
  reachable from the runbook's configured repository, and produces one draft
  PR against the exact fork base above.
