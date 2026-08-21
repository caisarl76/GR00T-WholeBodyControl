# Newcomer Onboarding Handover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the published onboarding revision reachable, write a reproducible teammate handover, transplant only the approved onboarding/PICO history onto the published fork base, and open one draft PR.

**Architecture:** The source branch first gains a repository-URL contract and then a progress handover containing stable milestones and pre-commit evidence. A separate publication worktree cherry-picks the approved commit set onto the exact fork base; final source and published SHAs exist only in the post-push PR body.

**Tech Stack:** MyST Markdown, reStructuredText/Sphinx, Bash, Python 3.10, pytest 9.0.3, Ruff 0.15.20, Git worktrees/cherry-pick, GitHub CLI

---

## File Map

- Modify `docs/source/getting_started/newcomer_onboarding.md`: add the configurable repository source used by both machines and fail closed if `origin` differs.
- Create `docs/superpowers/progress/2026-08-21-newcomer-onboarding-handover.md`: record operational authority, deviations, lab inputs, exact pre-commit evidence, teammate checklist, and publication resolution commands without self-referential SHAs.
- Create `docs/superpowers/plans/2026-08-21-newcomer-onboarding-handover.md`: this execution plan.
- Create no new runtime code. The publication transplant includes the already-reviewed PICO, camera-cleanup, test, and deployment-environment files identified by the approved design.

The angle-bracket strings below are the approved lab configuration tokens, not missing plan content. Do not start a robot, PICO service, deployment binary, camera server, manager, exporter, or viewer while executing this plan.

### Task 1: Make the Published Revision Reachable from the Runbook

**Files:**
- Modify: `docs/source/getting_started/newcomer_onboarding.md:155-311`
- Reference: `docs/superpowers/specs/2026-08-21-newcomer-onboarding-handover-design.md`

- [ ] **Step 1: Prove the current clone contract is missing**

Run from the source-worktree root:

```bash
python3 - <<'PY'
from pathlib import Path

text = Path("docs/source/getting_started/newcomer_onboarding.md").read_text()
assert "<REPOSITORY_URL>" not in text
assert text.count("git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git") == 2
print("EXPECTED FAILING STATE: runbook still hard-codes NVlabs")
PY
```

Expected: exit 0 and the exact diagnostic line. This confirms the reviewed publication gap still exists before editing.

- [ ] **Step 2: Add the repository configuration and origin proof**

Make these exact content changes:

1. Add the Configuration-table row:

```text
| `<REPOSITORY_URL>` | Git repository that contains `<REPO_REVISION>`; for this draft use the fork URL supplied by the handover. |
```

2. Add `export REPOSITORY_URL='<REPOSITORY_URL>'` immediately before `export REPO_REVISION=...` in both the workstation and PC2 export blocks.
3. Replace both clone commands with `git clone "$REPOSITORY_URL" ...`.
4. Add `test "$(git remote get-url origin)" = "$REPOSITORY_URL"` immediately after each `cd` into the new clone.
5. Extend the clone expectation paragraph to say that a repository-origin mismatch is a hard stop and that `<REPO_REVISION>` is the lab-supplied result of the handover's verified remote-head resolver.

- [ ] **Step 3: Validate the new clone contract**

```bash
python3 - <<'PY'
from pathlib import Path

text = Path("docs/source/getting_started/newcomer_onboarding.md").read_text()
assert text.count("<REPOSITORY_URL>") >= 3
assert text.count("export REPOSITORY_URL='<REPOSITORY_URL>'") == 2
assert text.count('git clone "$REPOSITORY_URL"') == 2
assert text.count('test "$(git remote get-url origin)" = "$REPOSITORY_URL"') == 2
assert "git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git" not in text
print("PASS: both machines clone and verify the configured repository")
PY
```

Expected: exit 0 and the exact `PASS` line.

- [ ] **Step 4: Parse all runbook command fences**

Run the exact command under **Runbook Command-Fence Validation** in `docs/superpowers/specs/2026-08-21-newcomer-onboarding-handover-design.md`.

Expected: exit 0 and `PASS: 32 Bash fences and 5 Python heredocs`.

- [ ] **Step 5: Commit the publication-contract correction**

```bash
git add docs/source/getting_started/newcomer_onboarding.md
git diff --cached --check
git commit -m "docs: make onboarding repository source configurable"
```

Expected: one commit changing only the runbook.

### Task 2: Collect Reproducible Source Evidence

**Files:**
- Verify: `decoupled_wbc/control/teleop/streamers/pico_streamer.py`
- Verify: `gear_sonic/scripts/pico_manager_thread_server.py`
- Verify: `gear_sonic/camera/composed_camera.py`
- Verify: `gear_sonic/tests/test_pico_planner_joystick.py`
- Verify: `gear_sonic/tests/test_pico_streamer_navigation.py`
- Verify: `gear_sonic/tests/test_composed_camera_server_shutdown.py`
- Verify: `gear_sonic/tests/test_run_camera_viewer.py`
- Verify: `gear_sonic_deploy/scripts/setup_env.sh`

- [ ] **Step 1: Create the pinned verification environment**

```bash
set -euo pipefail
bash install_scripts/install_pico.sh
uv pip install --python "$PWD/.venv_teleop/bin/python" \
  'pytest==9.0.3' \
  'ruff==0.15.20'
test "$(.venv_teleop/bin/python -m pytest --version)" = 'pytest 9.0.3'
test "$(.venv_teleop/bin/ruff --version)" = 'ruff 0.15.20'
```

Expected: exit 0 with Python 3.10, pytest 9.0.3, and Ruff 0.15.20 available inside `.venv_teleop`.

- [ ] **Step 2: Run the complete 14-test suite**

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

Expected: exit 0 and `14 passed`; deprecation warnings are recorded separately from failures.

- [ ] **Step 3: Run byte-compilation and both Ruff checks**

Run the exact fenced command under **Python Static Checks** in the approved handover design.

Expected: byte-compilation exits 0 and Ruff prints `All checks passed!` twice. The manager-only command excludes only its pre-existing `I001` import-order finding.

- [ ] **Step 4: Verify the deployment environment script**

Run the exact fenced command under **Deployment Environment Syntax and Behavior** in the approved handover design.

Expected: exit 0, exactly one `find ... -print -quit` lookup, and no retired `find ... | head` pipeline.

- [ ] **Step 5: Verify command fences, manifest structure, and the anchored toctree**

Run the exact fenced commands under these approved-design headings:

- **Runbook Command-Fence Validation** — expect `PASS: 32 Bash fences and 5 Python heredocs`.
- **Manifest Structure: Always Runnable** — expect `PASS: checksum manifest structure`.
- **Getting Started Toctree Membership** — expect `PASS: newcomer onboarding is in the Getting Started toctree exactly once`.

- [ ] **Step 6: Verify the downloaded artifact contents without copying model files**

The source worktree intentionally lacks three ignored ONNX files, while the primary checkout contains the already-downloaded pinned artifacts. Run the manifest from that checkout so paths resolve without duplicating 827 MiB:

```bash
set -euo pipefail
cd /home/jihun/work/GR00T-WholeBodyControl
LC_ALL=C sha256sum --check \
  worktrees/newcomer-onboarding/docs/source/getting_started/gear_sonic_deployment_9c0ff22.sha256
```

Expected: exit 0 and four `OK` lines for encoder, decoder, observation config, and planner.

- [ ] **Step 7: Force a complete Sphinx rebuild**

Run the exact fenced command under **Genuinely Forced Sphinx Rebuild** in the approved handover design.

Expected: build exit 0, nonempty `docs/build/html/getting_started/newcomer_onboarding.html`, and no warning naming the onboarding page. Record unrelated warning count separately.

### Task 3: Author and Commit the Teammate Handover

**Files:**
- Create: `docs/superpowers/progress/2026-08-21-newcomer-onboarding-handover.md`
- Reference: `docs/source/getting_started/newcomer_onboarding.md`
- Reference: `docs/superpowers/specs/2026-08-19-newcomer-onboarding-design.md`
- Reference: `docs/superpowers/plans/2026-08-20-newcomer-onboarding.md`
- Reference: `docs/superpowers/specs/2026-08-21-newcomer-onboarding-handover-design.md`

- [ ] **Step 1: Create the complete handover document**

Create the document with these exact sections and contracts:

1. `Outcome and Machine Topology`: link the final runbook and state PC2 owns deploy/camera while the workstation owns manager/exporter/viewer.
2. `Operational Authority and Superseded Instructions`: reproduce the approved four-row table for `ACTUATE` readiness, episode close, deployment stop, and remediation order; state the final runbook alone governs execution.
3. `Stable Source Milestones`: record `4db4795002bef8918537367a3739963935187acc`, `e2b923d1e986bb9019fa21295c1b16c81226e4dd`, `8cd095d5b4fc9f8a47930ea7b494bf5687d8d09c`, the approved design tip `7a2d0036e86f7ee380912c322c096b64621d0369`, this plan's resolved commit, and Task 1's resolved publication-contract commit. Explicitly state that the handover does not contain its own future SHA or the future published SHA.
4. `Deviation Ledger`: include one row for each of the seven unplanned files in the approved design, with the exact source commit, behavioral effect, rationale, test evidence, and role owner.
5. `Lab-Owned Inputs`: list all 15 allowlisted tokens, require the robot-owner stop procedure and no-hands confirmation, and state that `<REPOSITORY_URL>` resolves to `https://github.com/caisarl76/GR00T-WholeBodyControl.git` for this draft.
6. `Verification Evidence`: include every Task 2 command category, exact tool versions, exit status, test count, artifact results, Sphinx result, and the statement that verification was non-actuating. Include the full privacy validator and remote-head resolver from the approved design so a teammate can rerun them.
7. `Teammate Onboarding Checklist`: review the final runbook, populate all lab values outside committed docs, validate MuJoCo and stop paths, verify both machines use the same resolved revision, and conduct only a robot-owner-supervised first run.
8. `Publication Contract`: record repository `caisarl76/GR00T-WholeBodyControl`, base `vr3pt-cleanup-minimal-official` at `af76fae68930b4a9276af768015fc81fbcedc344`, remote head `refs/heads/docs/newcomer-onboarding`, no-force policy, and the fail-closed `git ls-remote --exit-code` resolver. State that final source-handover and published SHAs are recorded only in the post-push PR body.

- [ ] **Step 2: Run the per-document placeholder and privacy gate**

Run the exact fenced command under **Placeholder, Blank-Section, and Privacy Validation** in the approved design.

Expected: exit 0 and `PASS: placeholders, blank Section 3, and privacy`; each of the runbook, design, and handover independently contains exactly 15 allowlisted tokens.

- [ ] **Step 3: Rerun all non-actuating source verification after the handover content is final**

Repeat Task 2 Steps 2 through 7, then run:

```bash
set -euo pipefail
git diff --check 1e851175f883dc41299e60dc89f4949eac3ff04d..HEAD
git status --short
```

Expected: all verification results match Task 2. Before commit, status lists only `docs/superpowers/progress/2026-08-21-newcomer-onboarding-handover.md`.

- [ ] **Step 4: Commit without inserting the commit's future SHA**

```bash
git add docs/superpowers/progress/2026-08-21-newcomer-onboarding-handover.md
git diff --cached --check
git commit -m "docs: add newcomer onboarding teammate handover"
```

Expected: one documentation commit. Do not amend it to add its own hash.

- [ ] **Step 5: Capture post-commit source evidence for the PR body**

```bash
set -euo pipefail
SOURCE_HANDOVER_REVISION="$(git rev-parse HEAD)"
test "${#SOURCE_HANDOVER_REVISION}" -eq 40
git diff --check 1e851175f883dc41299e60dc89f4949eac3ff04d..HEAD
test -z "$(git status --short)"
printf 'SOURCE_HANDOVER_REVISION=%s\n' "$SOURCE_HANDOVER_REVISION"
```

Expected: exit 0, clean source worktree, and one 40-character source-handover SHA for later PR-body use only.

### Task 4: Transplant onto the Exact Published Base and Reverify

**Files:**
- Create worktree: `/home/jihun/work/GR00T-WholeBodyControl/worktrees/newcomer-onboarding-publish`
- Create local branch: `publish/newcomer-onboarding`
- Compare all onboarding-owned paths listed in the approved design's pre-push tree-equivalence command.

- [ ] **Step 1: Recheck the local and remote publication preconditions**

```bash
set -euo pipefail
test "$(git rev-parse af76fae68930b4a9276af768015fc81fbcedc344)" = \
  'af76fae68930b4a9276af768015fc81fbcedc344'
test -z "$(git ls-remote --heads fork refs/heads/docs/newcomer-onboarding)"
test ! -e /home/jihun/work/GR00T-WholeBodyControl/worktrees/newcomer-onboarding-publish
```

Expected: the exact base exists, the remote head is absent, and the publication path is unused. A changed remote blocks publication.

- [ ] **Step 2: Create the isolated publication worktree**

```bash
git worktree add \
  -b publish/newcomer-onboarding \
  /home/jihun/work/GR00T-WholeBodyControl/worktrees/newcomer-onboarding-publish \
  af76fae68930b4a9276af768015fc81fbcedc344
```

Expected: new worktree and branch at the exact base.

- [ ] **Step 3: Cherry-pick the approved stable history**

From the publication worktree:

```bash
set -euo pipefail
git cherry-pick \
  4db4795002bef8918537367a3739963935187acc \
  ab1aae58da7086af929cbc7e83bfb8be5d4de447 \
  c57782fece281d3dc5607d2b4de23b4d7c6fe4ca \
  680e9f07d2a4758ba7b976bcce1b377a1f7f5305 \
  026d4772f05feb559b782cc6ba6f0dffc586dec2 \
  7023033046060271cbe0a671f916ca56b217a96d \
  1e851175f883dc41299e60dc89f4949eac3ff04d \
  b2a2860d56a7bd15bc66ba75f2175d459774509b \
  bf91b1ebe6fc2a28d902e5f2f10887bf8b0388c1 \
  3620afcaac6975c53cfd4908f184804a92056976 \
  99a3cee626bc79aabb73fb007511e6702c032c3d \
  59e3c21d8a808f693930456e043294c722735926 \
  da5d21c8be25483385b3f7f5d6953c3bd9b0b739 \
  29d680f9d0efa369cc7b3164a8118ab01884b781 \
  868bbdf589dc99d96e9b5a8eb58bddf7a46b8004 \
  a3547641cf688ff08afc3f003bdee57169b40beb \
  b1a4f62deff3aaf8e7bc36c4c4cf995828c0c6fc \
  e2b923d1e986bb9019fa21295c1b16c81226e4dd \
  8cd095d5b4fc9f8a47930ea7b494bf5687d8d09c \
  61d7c34ba2042195b4784392074b41ff2a324244 \
  877bbdd66fb1ddbe359acaec821e40f0b09ecda7 \
  7a2d0036e86f7ee380912c322c096b64621d0369
SOURCE_TIP="$(git rev-parse docs/newcomer-onboarding)"
mapfile -t final_source_commits < <(
  git rev-list --reverse 7a2d0036e86f7ee380912c322c096b64621d0369.."$SOURCE_TIP"
)
test "${#final_source_commits[@]}" -ge 3
git cherry-pick "${final_source_commits[@]}"
```

Expected: every commit applies in source order. If and only if `docs/source/index.rst` conflicts, preserve the published base's existing entries and add exactly one `getting_started/newcomer_onboarding` entry under the Getting Started caption before continuing the cherry-pick. Any other conflict blocks publication.

- [ ] **Step 4: Prove tree equivalence and toctree placement**

Run the approved design's full pre-push `git diff --exit-code` command comparing `docs/newcomer-onboarding` with `publish/newcomer-onboarding`, then run **Getting Started Toctree Membership** from the publication worktree.

Expected: no diff for every onboarding-owned file and exactly one onboarding entry in the Getting Started toctree.

- [ ] **Step 5: Recreate the pinned verification environment and rerun the full contract**

From the publication worktree, repeat Task 2 Step 1, Task 2 Steps 2 through 7, and Task 3 Step 2. For artifact verification, run the publication manifest against the primary checkout's downloaded files as in Task 2 Step 6.

Expected: the same 14 tests and all static, manifest, privacy, toctree, artifact, Sphinx, and whitespace gates pass. Record results with the publication SHA for the PR body.

### Task 5: Push Without Force and Open One Draft PR

**Files:**
- No tracked files.
- Temporary PR body: `/tmp/newcomer-onboarding-pr-body.md`

- [ ] **Step 1: Perform the final no-force pre-push checks**

From the publication worktree:

```bash
set -euo pipefail
test -z "$(git status --short)"
test -z "$(git ls-remote --heads fork refs/heads/docs/newcomer-onboarding)"
test "$(git merge-base publish/newcomer-onboarding \
  af76fae68930b4a9276af768015fc81fbcedc344)" = \
  'af76fae68930b4a9276af768015fc81fbcedc344'
```

Expected: clean worktree, absent remote head, and exact ancestry. Do not continue if any check differs.

- [ ] **Step 2: Push the publication branch exactly once**

```bash
git push fork \
  publish/newcomer-onboarding:refs/heads/docs/newcomer-onboarding
```

Expected: a new remote branch is created. No `--force` option is permitted.

- [ ] **Step 3: Resolve and verify the final hashes**

```bash
set -euo pipefail
REPOSITORY_URL='https://github.com/caisarl76/GR00T-WholeBodyControl.git'
SOURCE_HANDOVER_REVISION="$(git rev-parse docs/newcomer-onboarding)"
PUBLISHED_REVISION="$(git rev-parse publish/newcomer-onboarding)"
REMOTE_REVISION="$(git ls-remote --exit-code --heads "$REPOSITORY_URL" \
  refs/heads/docs/newcomer-onboarding | awk 'NR == 1 {print $1}')"
test "$REMOTE_REVISION" = "$PUBLISHED_REVISION"
test "${#SOURCE_HANDOVER_REVISION}" -eq 40
test "${#PUBLISHED_REVISION}" -eq 40
printf 'SOURCE_HANDOVER_REVISION=%s\nPUBLISHED_REVISION=%s\n' \
  "$SOURCE_HANDOVER_REVISION" "$PUBLISHED_REVISION"
```

Expected: exit 0 and two 40-character SHAs. These values go in the PR body only.

- [ ] **Step 4: Create the draft PR body and open one draft PR**

Create `/tmp/newcomer-onboarding-pr-body.md` containing:

- outcome and PC2/workstation topology;
- exact base, head, source-handover SHA, and published SHA;
- source-to-published commit map from `git range-diff` or paired ordered logs;
- deviation summary for PICO navigation, camera shutdown, camera tests, and `setup_env.sh`;
- supersession statement naming `ACTUATE`, episode close/discard, deploy stop, and remediation order;
- exact verification commands/results, including 14 tests, two Ruff passes, 32 Bash fences, 5 Python heredocs, four artifact checksums, privacy/toctree passes, forced Sphinx result, and no hardware start;
- lab prerequisites and the remote-head resolver; and
- explicit reviewer request for PICO/manager, camera/runtime, deployment-environment, documentation, and robot-owner safety roles.

Then run:

```bash
gh pr create \
  --repo caisarl76/GR00T-WholeBodyControl \
  --base vr3pt-cleanup-minimal-official \
  --head docs/newcomer-onboarding \
  --title "docs: add newcomer PC2/workstation onboarding" \
  --body-file /tmp/newcomer-onboarding-pr-body.md \
  --draft
```

Expected: exactly one draft PR URL. Verify its base, head, draft state, and body before reporting completion.

## Final Self-Review

- [ ] Confirm the final runbook, not either historical plan, is the only operational authority.
- [ ] Confirm the handover contains no self or future published SHA.
- [ ] Confirm `<REPO_REVISION>` remains a lab-supplied value resolved from the verified remote head.
- [ ] Confirm the publication branch contains `4db4795`'s full four-file PICO scope.
- [ ] Confirm every validation is non-actuating and no hardware process was started.
- [ ] Confirm the primary dirty worktree remains untouched.
