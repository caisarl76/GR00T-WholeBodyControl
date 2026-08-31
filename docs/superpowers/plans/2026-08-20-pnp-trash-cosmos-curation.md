# PnP Trash Cosmos-Assisted Curation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe, resumable Cosmos3-Nano-assisted `task_index` curation workflow to `lerobot-dataset-visualizer`, use it to review all 92 episodes in `outputs/pnp_trash`, and atomically publish a separately cleaned LeRobot v2.1 dataset at `outputs/pnp_trash_cleaned` only after structural and real GR00T N1.7 loader validation.

**Architecture:** The visualizer's FastAPI backend gains an isolated `backend/curation/` package for registered local assets, SQLite state, Cosmos transport, review APIs, export, validation, and no-clobber publication. The Next.js frontend continues to load v2.1 metadata/parquet/video through Hugging Face-shaped URLs, proxies mutable curation JSON calls through a same-origin server route that injects the curation bearer token, and mounts a dedicated `task_index` editor inside the existing Annotations tab. `/home/jihun/work/Isaac-GR00T` is not modified; its statistics command and `LeRobotEpisodeLoader` are invoked as external validation gates against staging.

**Tech Stack:** Python 3.10, FastAPI, Pydantic, SQLite/WAL, PyArrow, NumPy, PyAV, Pillow, HTTPX, JSON Schema, pytest; Next.js 15, React 19, TypeScript, Bun, hyparquet, Testing Library; Linux `renameat2(RENAME_NOREPLACE)`; NVIDIA Cosmos3-Nano through an OpenAI-compatible vLLM endpoint; Isaac-GR00T N1.7.

---

## Binding context and repository boundaries

The approved design is [2026-08-18-pnp-trash-cosmos-curation-design.md](/home/jihun/work/GR00T-WholeBodyControl/docs/superpowers/specs/2026-08-18-pnp-trash-cosmos-curation-design.md). Its frozen prompt strings, Cosmos v2 request/response schema, state machines, artifact/provenance schemas, publication order, and acceptance gates are normative. If this plan abbreviates a field list, the approved design wins.

Implementation changes belong in the clean feature worktree
`/home/jihun/work/GR00T-WholeBodyControl/worktrees/lerobot-dataset-visualizer-pnp-trash`
unless a step explicitly names the current GR00T-WholeBodyControl repository.
Every executable command uses that value through `CURATION_REPO_ROOT`. Before
execution, the operator supplies an independently reviewed full commit SHA;
the runbook requires exact HEAD equality, baseline `60ef88c` ancestry, the
canonical Git worktree root, and an empty porcelain status. Do not modify
`/home/jihun/work/Isaac-GR00T`; it is a validation dependency at commit
`626af89` or the exact commit recorded when export runs.

The source dataset is immutable and currently has these verified properties:

- alias: `local/pnp_trash`;
- root: `/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash`;
- LeRobot version: `v2.1`;
- 92 episodes, 198,846 parquet/video frames, 50 Hz, one ego H.264 stream;
- 190 regular files and approximately 1.4 GiB, approved by the user on
  2026-08-28 under canonical manifest SHA-256
  `5962d8630f06e6260adbae15a3d7ee5f0a1a745c3a12466add8722c2e0da9577`;
  this includes the immutable top-level ancillary asset `pnp_trash.xlsx`
  (13,644 bytes, SHA-256
  `989f6968e5cf8ee0972b850199c948dd75ce140480c82cbe368053cde6ab34c9`),
  which is preserved as source evidence and is not annotation authority;
- one source prompt and an existing `task_index: int64` column;
- `modality.json` already maps `annotation.human.task_description` to `task_index`;
- left/right hands occupy `observation.state[22:29]` and `[36:43]`.

The final output is `/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_cleaned`; mutable state is `/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_curation`.

The authenticated feature worktree must be clean before any runtime command.
Never redirect this plan to the dirty primary visualizer checkout, and never
stash, reset, or overwrite user-owned changes outside the feature worktree.

## File map

New backend code:

```text
backend/__init__.py
backend/curation/__init__.py
backend/curation/config.py
backend/curation/security.py
backend/curation/models.py
backend/curation/prompts.py
backend/curation/source.py
backend/curation/assets.py
backend/curation/db.py
backend/curation/review.py
backend/curation/cosmos_contract.py
backend/curation/cosmos_transport.py
backend/curation/worker.py
backend/curation/grip.py
backend/curation/contact_sheets.py
backend/curation/audit.py
backend/curation/exporter.py
backend/curation/validation.py
backend/curation/publication.py
backend/curation/router.py
backend/curation/schemas/cosmos_response_v2.schema.json
backend/curation_worker.py
backend/curation_export.py
backend/tests/conftest.py
backend/tests/fixtures.py
backend/tests/test_legacy_v31_regression.py
backend/tests/test_curation_*.py
```

New frontend code:

```text
src/types/curation.types.ts
src/utils/curationClient.ts
src/utils/curationTimeline.ts
src/context/curation-context.tsx
src/components/task-index-curation-workspace.tsx
src/components/task-index-curation-timeline.tsx
src/components/curation-batch-status.tsx
src/components/curation-audit.tsx
src/app/api/curation/[...path]/route.ts
src/test-setup.ts
src/**/__tests__/curation-*.test.ts(x)
```

Existing files changed:

```text
backend/app.py
backend/requirements.txt
backend/README.md
package.json
bun.lock
bunfig.toml
.env.example
src/utils/auth.ts
src/utils/parquetUtils.ts
src/utils/versionUtils.ts
src/app/[org]/[dataset]/[episode]/episode-viewer.tsx
```

## Task 1: Freeze the baseline and add executable backend test infrastructure

**Files:**

- Modify: `backend/requirements.txt`
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/fixtures.py`
- Create: `backend/tests/test_legacy_v31_regression.py`

- [ ] From the visualizer root, record the starting point without changing it:

```bash
cd "$CURATION_REPO_ROOT"
git status --short --branch
git diff -- src/app/'[org]'/'[dataset]'/'[episode]'/episode-viewer.tsx \
  src/app/'[org]'/'[dataset]'/'[episode]'/fetch-data.ts \
  src/utils/versionUtils.ts
bun run validate
```

Expected: the existing frontend gate passes or any pre-existing failure is recorded verbatim before curation changes. Do not repair unrelated failures in this plan.

- [ ] Extend `backend/requirements.txt` with the runtime and test dependencies used by the approved design:

```text
httpx>=0.27
jsonschema>=4.22
numpy>=1.26
Pillow>=10
av>=12
pytest>=8
```

- [ ] Create `backend/.venv` if absent, install requirements, and prove the mandated command is callable:

```bash
python3.10 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
backend/.venv/bin/python -m pytest -q backend/tests
```

Expected initial result: collection succeeds; the new regression test fails until its fixture is complete.

- [ ] In `backend/tests/fixtures.py`, build a tiny valid v3.1 fixture under `tmp_path` with `meta/info.json`, `meta/episodes/chunk-000/file-000.parquet`, `data/chunk-000/file-000.parquet`, and a minimal MP4-independent schema. Include persistent, event, and tool-call atoms so every legacy column route is exercised.

- [ ] In `test_legacy_v31_regression.py`, load that fixture through every existing endpoint, round-trip all atom styles, export it, and assert the output contains `language_persistent`, `language_events`, and dataset-level `tools` with no `task_index` curation fields added. Snapshot the response key sets so later router work cannot silently alter v3.1 responses.

- [ ] Run the regression test:

```bash
backend/.venv/bin/python -m pytest -q backend/tests/test_legacy_v31_regression.py
```

Expected: PASS.

- [ ] Commit only the test scaffold and dependency change:

```bash
git add backend/requirements.txt backend/tests/conftest.py backend/tests/fixtures.py backend/tests/test_legacy_v31_regression.py
git diff --cached --check
git commit -m "test: freeze visualizer v3.1 annotation behavior"
```

## Task 2: Make Hugging Face authentication destination-aware

**Files:**

- Modify: `src/utils/auth.ts`
- Modify: `src/utils/parquetUtils.ts`
- Modify carefully: `src/utils/versionUtils.ts`
- Create: `src/utils/__tests__/auth.test.ts`
- Modify: `src/utils/__tests__/parquetUtils.test.ts`
- Modify: `src/utils/__tests__/versionUtils.test.ts`

- [ ] Write failing tests that seed `lerobot-viz-oauth` with a sentinel token and cover the exact destination matrix:

```ts
const protectedHf =
  "https://huggingface.co/datasets/acme/private/resolve/main/meta/info.json";
const localAsset =
  "http://127.0.0.1:8000/api/local-datasets/local/pnp_trash/resolve/main/meta/info.json";

expect(authHeaders(protectedHf)).toEqual({
  Authorization: "Bearer sentinel-hf-token",
});
expect(authHeaders(localAsset)).toEqual({});
expect(authHeaders("https://huggingface.co.evil.test/file")).toEqual({});
expect(authHeaders("http://huggingface.co/file")).toEqual({});
expect(authHeaders("/api/curation/summary")).toEqual({});
expect(authHeaders("not a url")).toEqual({});
```

Also mock `fetch` and `asyncBufferFromUrl` so metadata, version, parquet full/range, and local-video URL construction are all proven not to carry the sentinel header.

- [ ] Run the focused tests and confirm they fail because `authHeaders()` has no destination parameter:

```bash
bun test src/utils/__tests__/auth.test.ts src/utils/__tests__/parquetUtils.test.ts src/utils/__tests__/versionUtils.test.ts
```

- [ ] Implement one shared predicate and make every caller pass its concrete destination:

```ts
export function isAuthenticatedHfDestination(
  destination: string | URL,
): boolean {
  try {
    const url = destination instanceof URL ? destination : new URL(destination);
    return url.protocol === "https:" && url.hostname === "huggingface.co";
  } catch {
    return false;
  }
}

export function authHeaders(destination: string | URL): Record<string, string> {
  if (!isAuthenticatedHfDestination(destination)) return {};
  const token = getAuthToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}
```

Use the same predicate in `proxyHfUrl`. In `fetchJson`, `fetchParquetFile`, and `getDatasetInfo`, pass the final URL. Do not add an overload that preserves the unsafe zero-argument behavior.

- [ ] Run the focused and complete frontend gates:

```bash
bun test src/utils/__tests__/auth.test.ts src/utils/__tests__/parquetUtils.test.ts src/utils/__tests__/versionUtils.test.ts
bun run format
bun run validate
```

Expected: all pass, and formatting preserves the user-owned semantic changes in `versionUtils.ts`.

- [ ] Commit explicit files only:

```bash
git add src/utils/auth.ts src/utils/parquetUtils.ts src/utils/versionUtils.ts \
  src/utils/__tests__/auth.test.ts src/utils/__tests__/parquetUtils.test.ts \
  src/utils/__tests__/versionUtils.test.ts
git diff --cached --check
git commit -m "fix: scope HF credentials to exact destinations"
```

## Task 3: Add explicit curation configuration, source registration, and local range assets

**Files:**

- Create: `backend/__init__.py`
- Create: `backend/curation/__init__.py`
- Create: `backend/curation/config.py`
- Create: `backend/curation/security.py`
- Create: `backend/curation/source.py`
- Create: `backend/curation/assets.py`
- Create: `backend/curation/router.py`
- Modify: `backend/app.py`
- Create: `backend/tests/test_curation_config.py`
- Create: `backend/tests/test_local_assets.py`

- [ ] Write failing settings tests for these required environment values and default limits:

```text
CURATION_DATASET_ALIASES_JSON
CURATION_WORKSPACE
CURATION_OUTPUT
CURATION_BROWSER_ORIGIN
CURATION_BEARER_TOKEN
COSMOS_BASE_URL
COSMOS_MODEL
COSMOS_API_KEY_ENV
COSMOS_ENDPOINT_IDENTITY
ISAAC_GROOT_ROOT
```

Defaults are worker concurrency 1, HTTP timeout 120 seconds, transport attempts 2, repair attempts 1, target sampling 2 fps, maximum duration 120 seconds, maximum sampled frames 240, and maximum payload bytes 67,108,864. Validate canonical absolute paths, source/output/workspace separation, loopback backend binding as a startup preflight, and exactly one browser origin.

- [ ] Write failing asset tests with a temporary registered dataset. Cover `200`, `HEAD`, prefix/open-ended/suffix `206`, `304`, mismatched `If-Range` fallback, malformed/multiple-range `416`, exact headers, MIME types, no content encoding, unknown aliases, non-`main` revisions, absolute paths, raw and percent-decoded `..`, escaping symlinks, non-regular files, and any `Authorization` header returning `400`.

- [ ] Add an empty `backend/__init__.py`, implement `CurationSettings.from_env()`, and use package-relative backend imports. Implement immutable alias resolution. Compute `source-files.sha256` in POSIX UTF-8 path-byte order, reject CR/LF paths, and use its byte hash as the source fingerprint. Store the manifest in the workspace, never in the source tree.

- [ ] Implement:

```text
GET|HEAD /api/local-datasets/{org}/{dataset}/resolve/main/{asset_path:path}
```

Resolve `f"{org}/{dataset}"` through the startup registry, resolve the candidate path, enforce containment after symlink resolution, open only regular files, and stream an inclusive byte interval without buffering the full object. ETags come from the immutable source manifest. Metadata uses `Cache-Control: no-store`; parquet/video use private ETag revalidation.

- [ ] Replace wildcard CORS in `backend/app.py` with `allow_origins=[settings.browser_origin]`, expose `Accept-Ranges`, `Content-Range`, `Content-Length`, and `ETag`, keep all legacy routes registered, and include a separate curation router.

- [ ] Run focused tests plus the v3.1 gate:

```bash
backend/.venv/bin/python -m pytest -q \
  backend/tests/test_curation_config.py \
  backend/tests/test_local_assets.py \
  backend/tests/test_legacy_v31_regression.py
```

Expected: PASS.

- [ ] Commit:

```bash
git add backend/__init__.py backend/app.py backend/curation \
  backend/tests/test_curation_config.py backend/tests/test_local_assets.py
git diff --cached --check
git commit -m "feat: serve registered local LeRobot assets safely"
```

## Task 4: Implement SQLite migrations and optimistic, concurrency-safe repositories

**Files:**

- Create: `backend/curation/models.py`
- Create: `backend/curation/db.py`
- Create: `backend/tests/test_curation_db.py`

- [ ] Write failing tests that open multiple independent connections and prove:
  - `journal_mode=WAL`, `foreign_keys=ON`, `synchronous=FULL`, and a 5,000 ms busy timeout;
  - migrations are transactional and idempotent through `PRAGMA user_version`;
  - two updates with the same `expected_revision` yield one commit and one optimistic conflict;
  - only one active Cosmos batch and one active export may exist per dataset;
  - audit events are append-only;
  - artifact paths reject absolute paths and `..`;
  - approval snapshots are stable across connection/process order.

- [ ] Define string enums in `models.py` for review, job, attempt, export, proposal, and completion states. The review enum is exactly:

```python
class ReviewState(str, Enum):
    PENDING = "pending"
    DRAFT = "draft"
    APPROVED_KEEP = "approved_keep"
    APPROVED_REJECT = "approved_reject"
```

- [ ] Implement version-1 SQL migrations for `datasets`, `episodes`, `cosmos_jobs`, `cosmos_attempts`, `cosmos_proposals`, `artifacts`, `audit_events`, `exports`, and `export_episodes`. Store the six step-start fields explicitly as nullable `INTEGER` columns so SQLite constraints and approval queries can validate them. Add foreign keys, state `CHECK` constraints, nonnegative sizes/indices, unique `(dataset_id, source_episode_index)`, unique `(job_id, source_episode_index, attempt_number)`, and the two partial unique active-work indexes required by the design.

- [ ] Make every mutation use a short `BEGIN IMMEDIATE` context manager. Convert SQLite lock exhaustion to a typed retryable service error and optimistic revision mismatch to an HTTP-ready `409` payload containing the current episode record.

- [ ] Implement canonical JSON with `ensure_ascii=False`, sorted keys, compact separators, and newline-free hashing. Use it for approval snapshots and all later hash-bearing records.

- [ ] Run:

```bash
backend/.venv/bin/python -m pytest -q backend/tests/test_curation_db.py
```

Expected: PASS, including a process-level concurrent writer test.

- [ ] Commit:

```bash
git add backend/curation/models.py backend/curation/db.py backend/tests/test_curation_db.py
git diff --cached --check
git commit -m "feat: persist curation state transactionally"
```

## Task 5: Implement the frozen prompt generator and human-review state machine

**Files:**

- Create: `backend/curation/prompts.py`
- Create: `backend/curation/review.py`
- Modify: `backend/curation/router.py`
- Create: `backend/tests/test_prompts.py`
- Create: `backend/tests/test_review_api.py`

- [ ] Write prompt tests using the exact seven outputs and assert the version/hash:

```python
assert expand_prompts(object_name="crumpled can", hand="left", turn="right") == [
    "approach the brown table",
    "pick up the crumpled can from the table with the left hand",
    "turn right to find the black trash bin",
    "approach the black trash bin while holding the crumpled can",
    "lean down to the black trash bin",
    "drop the crumpled can into the black trash bin",
    "go to a standing straight pose",
]
```

Normalization trims outer whitespace and collapses repeated internal whitespace; it does not lowercase or otherwise rewrite object text. `hand` and `turn` are lowercase `left|right` enums. Keep all templates in this backend module only; frontend preview arrives in API responses.

- [ ] Write failing state-machine/API tests for every allowed and forbidden edge, approved-row locking, `Reopen`, optional reject reason, approval revision equality, stale prompt-hash invalidation, reviewer identity, six strictly increasing in-range transitions, seven nonempty spans, and bidirectional optimistic conflicts.

- [ ] Implement these authenticated JSON endpoints with `dataset_alias` in request bodies or query parameters, never an absolute path:

```text
POST  /api/curation/workspaces/open
GET   /api/curation/summary
GET   /api/curation/episodes/{source_episode_index}
PATCH /api/curation/episodes/{source_episode_index}/draft
POST  /api/curation/episodes/{source_episode_index}/apply-proposal
POST  /api/curation/episodes/{source_episode_index}/approve-keep
POST  /api/curation/episodes/{source_episode_index}/approve-reject
POST  /api/curation/episodes/{source_episode_index}/reopen
```

Every mutation body includes `expected_revision` and `actor`; approval bodies additionally include `reviewer`. The episode response includes source length, timestamps, current decision, active proposal, prompt preview, warnings, revision, and approval lock status.

- [ ] Implement prompt-template invalidation as one transaction: a changed prompt hash converts every `approved_keep` to `draft`, clears approval identity/time/revision, increments the episode revision, and appends one audit event per episode. `approved_reject` remains approved because it emits no training prompt. A completed Cosmos retry leaves both keep and reject approvals unchanged until an explicit reopen/apply action.

- [ ] Protect mutable curation routes with a constant-time comparison against `CURATION_BEARER_TOKEN`. Do not protect the read-only local asset route with this bearer token; that route instead rejects all `Authorization` headers.

- [ ] Run:

```bash
backend/.venv/bin/python -m pytest -q backend/tests/test_prompts.py backend/tests/test_review_api.py
```

Expected: PASS.

- [ ] Commit:

```bash
git add backend/curation/prompts.py backend/curation/review.py backend/curation/router.py \
  backend/tests/test_prompts.py backend/tests/test_review_api.py
git diff --cached --check
git commit -m "feat: add revisioned episode review workflow"
```

## Task 6: Freeze and validate the Cosmos v2 response contract

**Files:**

- Create: `backend/curation/schemas/cosmos_response_v2.schema.json`
- Create: `backend/curation/cosmos_contract.py`
- Create: `backend/tests/test_cosmos_contract.py`

- [ ] Copy the approved Draft 2020-12 JSON Schema byte-for-byte from the design into `cosmos_response_v2.schema.json`. Add a test that loads it and checks `schema_version == 2`, seven ordered step objects, observed/completed semantics, conditional nullability, and `additionalProperties: false` wherever frozen.

- [ ] Write table-driven failing tests for the approved complete and incomplete examples plus hostile responses: Markdown fences, trailing prose, multiple JSON objects, unterminated `<think>`, absent steps, invented times for unobserved steps, inconsistent `completion`/`missing_steps`, wrong step order, out-of-range or nonfinite times, and a successful partial proposal with nullable transition frames.

- [ ] Implement a strict response parser that removes at most one leading `<think>...</think>` block, accepts exactly one remaining JSON object, validates the static schema, then enforces dynamic video-duration and bidirectional completion invariants. Never persist a `parsed.json` until both validations pass.

- [ ] Implement nearest-source-frame snapping using float64 absolute distance, tie-breaking to the smaller frame index. Preserve `null` for not-observed transitions. A complete response must yield six non-null, strictly increasing frames; an incomplete response may yield nulls and must never be eligible for human approval without edits.

- [ ] Run:

```bash
backend/.venv/bin/python -m pytest -q backend/tests/test_cosmos_contract.py
```

Expected: PASS.

- [ ] Commit:

```bash
git add backend/curation/schemas/cosmos_response_v2.schema.json \
  backend/curation/cosmos_contract.py backend/tests/test_cosmos_contract.py
git diff --cached --check
git commit -m "feat: freeze Cosmos trash annotation contract"
```

## Task 7: Implement deterministic 2 fps sampling, vLLM transport, and atomic artifacts

**Files:**

- Create: `backend/curation/cosmos_transport.py`
- Modify: `backend/curation/source.py`
- Create: `backend/tests/test_cosmos_sampling.py`
- Create: `backend/tests/test_cosmos_transport.py`
- Create: `backend/tests/test_artifacts.py`

- [ ] Build a synthetic 50 Hz parquet/video fixture and write failing sampling tests for the approved alignment proof:
  - `frame_index[i] == i`;
  - finite timestamps within `1/(2F)` of `i/F`;
  - video stream rate within `1e-6` of `F`;
  - decoded/container frame count equals parquet row count `N`;
  - targets `0.5*k <= p[N-1]` choose the nearest timestamp, tie toward the lower index, and deduplicate.

For the real 50 Hz fixture, expected sampled indices begin `[0, 25, 50, 75]` and the request records their actual parquet timestamps.

- [ ] Write deterministic image tests: decode the exact selected PyAV frames, convert RGB, never upscale, resize the long edge to at most 640, and encode JPEG quality 85 with `optimize=False` and `progressive=False`. Hash repeated encodes and require identical bytes.

- [ ] Write limit tests that return `manual_only` without calling HTTP when duration exceeds 120 seconds, sample count exceeds 240, comma-joined base64 payload after the data-URL prefix exceeds 67,108,864 bytes, or alignment cannot be proven. These outcomes must not reject the episode.

- [ ] Write an HTTPX `MockTransport` test for the exact OpenAI-compatible body:

```json
{
  "model": "cosmos3-nano-test",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "video_url",
          "video_url": { "url": "data:video/jpeg;base64,YWJjZA==,ZWZnaA==" }
        },
        { "type": "text", "text": "the frozen Cosmos v2 prompt" }
      ]
    }
  ],
  "temperature": 0,
  "seed": 0,
  "max_completion_tokens": 4096,
  "extra_body": {
    "media_io_kwargs": {
      "video": {
        "fps": 50.0,
        "frames_indices": [0, 25, 50],
        "total_num_frames": 2060,
        "duration": 41.2,
        "do_sample_frames": false
      }
    }
  }
}
```

The values are fixture-derived; `fps` is source fps, not 2. Exercise stop success, length failure, missing choices, connection retry, retryable 408/429/5xx, non-retryable 4xx, timeout, one schema-repair request, and repair failure.

- [ ] Implement artifact writes as `temporary file -> flush -> file fsync -> atomic rename -> parent fsync -> SHA-256 -> database reference`. `request.json` contains the redacted base64 descriptor and full sampling metadata; response text is exact UTF-8 model content; `parsed.json` is written only after validation; repair text exists only after a repair response.

- [ ] Verify that a crash before the database transaction leaves at most an unreferenced complete artifact, never a database row pointing at partial bytes. Add startup cleanup/reporting for unreferenced temporary files without deleting complete evidence automatically.

- [ ] Run:

```bash
backend/.venv/bin/python -m pytest -q \
  backend/tests/test_cosmos_sampling.py \
  backend/tests/test_cosmos_transport.py \
  backend/tests/test_artifacts.py
```

Expected: PASS.

- [ ] Commit:

```bash
git add backend/curation/cosmos_transport.py backend/curation/source.py \
  backend/tests/test_cosmos_sampling.py backend/tests/test_cosmos_transport.py \
  backend/tests/test_artifacts.py
git diff --cached --check
git commit -m "feat: transport deterministic video samples to Cosmos"
```

## Task 8: Implement the resumable Cosmos batch worker and cancellation API

**Files:**

- Create: `backend/curation/worker.py`
- Create: `backend/curation_worker.py`
- Modify: `backend/curation/router.py`
- Create: `backend/tests/test_curation_batches.py`
- Create: `backend/tests/test_curation_worker.py`

- [ ] Write failing API tests for the frozen routes:

```text
POST /api/curation/batches
GET  /api/curation/batches/{id}
POST /api/curation/batches/{id}/retry
POST /api/curation/batches/{id}/cancel
```

Start validates source fingerprint, endpoint capability, model identity, episode selection, and the one-active-batch constraint; it queues attempts but starts no in-process worker. Status returns immutable configuration, counts, per-episode latest states, lease state, and cancel state. Retry creates a new job with explicit episode indices and leaves prior rows/artifacts immutable.

- [ ] Freeze cancellation behavior in tests exactly: an unknown ID returns `404`; queued work and all attempts become cancelled atomically with `200 changed:true`; running becomes `cancel_requested` with `202 changed:true`; a repeated cancel-requested call returns `202 changed:false` without another audit event; cancelled returns `200 changed:false`; completed, completed-with-failures, and failed return `409 batch_terminal` with unchanged state. After cancellation, the worker claims nothing new, lets the in-flight request reach its 120-second bound and persist, then atomically cancels all remaining queued/retryable attempts and the job.

- [ ] Write worker lease tests with an injected UTC clock and owner UUID. Cover atomic claim, concurrency one, heartbeat, lease expiry, process crash after request artifact, crash after response artifact, crash after proposal commit, stale lease reclamation, and exactly-once proposal activation. A retry or recovered worker must never mutate human review fields.

- [ ] Implement the exact CLI contract:

```bash
backend/.venv/bin/python backend/curation_worker.py \
  --workspace /home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_curation \
  run --job-id 00000000-0000-0000-0000-000000000000

backend/.venv/bin/python backend/curation_worker.py \
  --workspace /home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_curation \
  resume --job-id 00000000-0000-0000-0000-000000000000
```

The all-zero UUID is a parser test value only; production uses the UUID returned by batch creation. `run` accepts only the named queued job and a second `run` never attaches to it. `resume` accepts only the named `running` or `cancel_requested` job after its 180-second job/attempt lease expires; it creates no new attempt rows. The owning worker renews its leases every 15 seconds, prints machine-readable status lines, and handles SIGINT/SIGTERM by stopping claims and persisting or time-bounding the active request. Exit codes are exactly `0` for a terminal nonfailed job, `1` for job failure, `2` for argument/configuration/state error, `3` for a live-lease conflict, and `130` for a handled interrupt.

- [ ] Run:

```bash
backend/.venv/bin/python -m pytest -q backend/tests/test_curation_batches.py backend/tests/test_curation_worker.py
```

Expected: PASS, including a fake Cosmos server proving the API remains queued until the CLI worker starts.

- [ ] Commit:

```bash
git add backend/curation/worker.py backend/curation_worker.py backend/curation/router.py \
  backend/tests/test_curation_batches.py backend/tests/test_curation_worker.py
git diff --cached --check
git commit -m "feat: add resumable Cosmos curation worker"
```

## Task 9: Implement deterministic grip diagnostics, contact sheets, and audit summaries

**Files:**

- Create: `backend/curation/grip.py`
- Create: `backend/curation/contact_sheets.py`
- Create: `backend/curation/audit.py`
- Modify: `backend/curation/review.py`
- Modify: `backend/curation/router.py`
- Create: `backend/tests/test_grip.py`
- Create: `backend/tests/test_contact_sheets.py`
- Create: `backend/tests/test_curation_audit.py`

- [ ] Write NumPy reference tests for the approved hand candidate algorithm. Assert exact source slices, percentile method `linear`, usable-joint threshold, clip/scale, the `[0,min(0.5,duration))` median baseline, float64 centered 11-frame edge-padded boxcar, `d[i]=a[i+5]-a[i-5]`, smallest-index tie breaks, release strictly after grasp, and sign checks. Test unavailable diagnostics for every defined failure.

- [ ] Test that a candidate more than 2.0 seconds from final step-2 or step-6 start creates an advisory warning only. It must never edit a transition, invalidate an approval, or auto-reject.

- [ ] Write six-cell contact-sheet tests. Integer proposal transitions decode their frames and show timestamp/status. Null transitions render deterministic `NOT OBSERVED` placeholders with no decoded image, timestamp, or delta. Approved-keep final sheets always use six non-null human transitions; proposal delta is shown only for corresponding integer proposal frames. Approved-reject emits no final sheet. Filenames include immutable proposal ID or approval revision.

- [ ] Implement authenticated endpoints:

```text
GET /api/curation/episodes/{source_episode_index}/grip
GET /api/curation/audit
```

The audit response reports review-state counts, Cosmos lifecycle/result counts, transition and duration distributions, order/coverage errors, grip disagreements, unreadable files, and source-fingerprint status. Keep raw Cosmos reasoning out of this response.

- [ ] Run:

```bash
backend/.venv/bin/python -m pytest -q \
  backend/tests/test_grip.py \
  backend/tests/test_contact_sheets.py \
  backend/tests/test_curation_audit.py
```

Expected: PASS.

- [ ] Commit:

```bash
git add backend/curation/grip.py backend/curation/contact_sheets.py backend/curation/audit.py \
  backend/curation/review.py backend/curation/router.py backend/tests/test_grip.py \
  backend/tests/test_contact_sheets.py backend/tests/test_curation_audit.py
git diff --cached --check
git commit -m "feat: add physical and visual curation audits"
```

## Task 10: Add the same-origin curation proxy and typed frontend state

**Files:**

- Modify: `package.json`
- Modify: `bun.lock`
- Create: `bunfig.toml`
- Create: `src/test-setup.ts`
- Create: `src/types/curation.types.ts`
- Create: `src/utils/curationClient.ts`
- Create: `src/utils/curationTimeline.ts`
- Create: `src/context/curation-context.tsx`
- Create: `src/app/api/curation/[...path]/route.ts`
- Create: `src/app/api/curation/[...path]/__tests__/route.test.ts`
- Create: `src/context/__tests__/curation-context.test.tsx`
- Create: `src/utils/__tests__/curationTimeline.test.ts`

- [ ] Install DOM interaction-test dependencies with Bun:

```bash
bun add --dev @testing-library/react @testing-library/user-event happy-dom
```

Configure Bun to preload `src/test-setup.ts`; install Happy DOM globals and Testing Library cleanup there.

- [ ] Write failing proxy tests that set `CURATION_BACKEND_URL` and `CURATION_BEARER_TOKEN`, invoke GET/POST/PATCH handlers, and assert:
  - only relative paths under `/api/curation/` are forwarded;
  - query strings, method, content type, body, status, and JSON are preserved;
  - client `Authorization` is discarded;
  - the server-side curation bearer is injected;
  - neither bearer nor backend URL is returned to the browser or logged;
  - timeouts become `504` and upstream connection failures become `502`;
  - no local asset, parquet, or video request is proxied here.

- [ ] Implement `src/app/api/curation/[...path]/route.ts` for JSON-only `GET`, `POST`, and `PATCH`, `runtime="nodejs"`, `dynamic="force-dynamic"`, `cache="no-store"`, and a 120-second upstream timeout. Reject missing server configuration with `503`.

- [ ] Define exact TypeScript discriminated unions matching backend enums and response fields. Do not recreate the seven prompt templates. `EpisodeCuration.promptPreview` is the sole frontend prompt source.

- [ ] Write timeline pure-function tests for nearest-frame snapping, six strictly increasing transitions, one-frame nudge bounds, seven exhaustive spans, duration labels, and incomplete nullable proposals.

- [ ] Write provider/reducer interaction tests for load, save draft, optimistic `409` reload prompt, apply proposal, approval lock, reopen, batch polling, terminal-state stop, stale request cancellation on episode navigation, and approve-and-next selection.

- [ ] Run:

```bash
bun test src/app/api/curation/'[...path]'/__tests__/route.test.ts \
  src/context/__tests__/curation-context.test.tsx \
  src/utils/__tests__/curationTimeline.test.ts
```

Expected: PASS.

- [ ] Commit:

```bash
git add package.json bun.lock bunfig.toml src/test-setup.ts src/types/curation.types.ts \
  src/utils/curationClient.ts src/utils/curationTimeline.ts src/context/curation-context.tsx \
  src/app/api/curation/'[...path]'/route.ts \
  src/app/api/curation/'[...path]'/__tests__/route.test.ts \
  src/context/__tests__/curation-context.test.tsx \
  src/utils/__tests__/curationTimeline.test.ts
git diff --cached --check
git commit -m "feat: add secure frontend curation transport"
```

## Task 11: Build and integrate the `task_index` curation workspace

**Files:**

- Create: `src/components/task-index-curation-workspace.tsx`
- Create: `src/components/task-index-curation-timeline.tsx`
- Create: `src/components/curation-batch-status.tsx`
- Create: `src/components/curation-audit.tsx`
- Create: `src/components/__tests__/task-index-curation-workspace.test.tsx`
- Create: `src/components/__tests__/task-index-curation-timeline.test.tsx`
- Modify carefully: `src/app/[org]/[dataset]/[episode]/episode-viewer.tsx`

- [ ] Write interaction tests before components. Cover object free text plus prior-value suggestions, hand/turn selectors, selected boundary from current video time, six draggable frame-snapped boundaries, `-1/+1 frame`, proposal/human/grip markers, nullable proposal display, keep/reject fields, save draft, approve-and-next, reopen, validation warnings, stale approval, batch create/status/cancel/retry, and audit distributions.

- [ ] Implement the timeline as six controlled integer transition handles over `[0, frameCount-1]`. Use `useTime().currentTime` and `seek(frameTimestamp)` from the existing shared player context. Pointer updates snap through the backend-provided source timestamps; keyboard nudges move exactly one source frame and cannot cross neighboring handles or create an empty span.

- [ ] Implement the workspace with explicit modes and lock behavior:
  - `task_index` is the default annotation mode only when `repoId === "local/pnp_trash"` and `codebase_version === "v2.1"`;
  - the existing v3.1 atom editor remains the default and unchanged for every other dataset;
  - approved episodes render read-only until `Reopen` succeeds;
  - conflicts stop mutation and show the current server revision;
  - applying Cosmos creates/updates a draft only;
  - approval buttons remain disabled until the server reports the record valid;
  - raw `<think>` text is never rendered.

- [ ] In `episode-viewer.tsx`, preserve the current user changes and replace only the Annotations-tab editor block with a small mode wrapper. Reuse the existing sidebar, `SimpleVideosPlayer`, `PlaybackBar`, `TimeProvider`, and route-based episode navigation. Do not alter `fetch-data.ts` or `tab-url.ts`.

- [ ] Run focused UI tests:

```bash
bun test src/components/__tests__/task-index-curation-workspace.test.tsx \
  src/components/__tests__/task-index-curation-timeline.test.tsx
```

Expected: PASS.

- [ ] Run mandatory frontend gates and inspect the dirty-file delta:

```bash
bun run format
bun run validate
git diff -- src/app/'[org]'/'[dataset]'/'[episode]'/episode-viewer.tsx \
  src/app/'[org]'/'[dataset]'/'[episode]'/fetch-data.ts \
  src/app/'[org]'/'[dataset]'/'[episode]'/tab-url.ts
```

Expected: PASS; `fetch-data.ts` and `tab-url.ts` contain no curation edits.

- [ ] Commit only the new components/tests and the narrow integration hunk:

```bash
git add src/components/task-index-curation-workspace.tsx \
  src/components/task-index-curation-timeline.tsx \
  src/components/curation-batch-status.tsx src/components/curation-audit.tsx \
  src/components/__tests__/task-index-curation-workspace.test.tsx \
  src/components/__tests__/task-index-curation-timeline.test.tsx \
  src/app/'[org]'/'[dataset]'/'[episode]'/episode-viewer.tsx
git diff --cached --check
git commit -m "feat: add task-index curation workspace"
```

## Task 12: Build deterministic v2.1 staging exports without touching source bytes

**Files:**

- Create: `backend/curation/exporter.py`
- Create: `backend/curation_export.py`
- Modify: `backend/curation/router.py`
- Create: `backend/tests/test_stable_tasks.py`
- Create: `backend/tests/test_exporter.py`
- Create: `backend/tests/test_export_lifecycle.py`

- [ ] Write stable-task tests that expand all kept prompts, deduplicate exact strings, assign each string its minimum step number, sort by `(minimum_step, raw_utf8_bytes)`, enumerate from zero, and write canonical compact JSON lines with sorted keys and a final newline. Shuffle database insertion and worker completion order and require byte-identical `tasks.jsonl`.

- [ ] Write failing export preflight tests: all 92 snapshot rows must be approved; pending/draft blocks; zero kept blocks; invalid approval revision blocks; source fingerprint mismatch blocks; existing final path blocks; active export blocks. Approval snapshot creation and copied `export_episodes` rows occur in one `BEGIN IMMEDIATE` transaction.

- [ ] Build a multi-episode synthetic v2.1 fixture containing fixed-size lists, floats, integers, nulls, unknown columns, Arrow schema metadata, multiple parquet row groups, videos, and extra regular assets. Write tests that source-index-sort kept episodes, renumber output episodes and global indices contiguously, preserve every source frame, and replace only `episode_index`, `frame_index`, `index`, and `task_index` while retaining those fields' original Arrow types/nullability.

- [ ] For each untouched column, assert schema-field equality, `pyarrow.ChunkedArray.combine_chunks().equals`, canonical Arrow IPC hash equality, row order, null positions, and list shapes. Do not reuse the legacy `_materialize_tree` hardlink path. Copy every carried asset as an independent regular file; require a different `(st_dev, st_ino)` and identical SHA-256, including MP4s. Reject source symlinks rather than reproducing them.

- [ ] Write exact output metadata tests:
  - `info.json` totals, splits, paths, counts, and existing feature definitions;
  - `episodes.jsonl` with contiguous output indices, exact lengths, and seven expanded prompts in step order;
  - canonical `tasks.jsonl` with no unreferenced task;
  - source `modality.json` semantics preserved, including task-index language mapping;
  - one recomputed `episodes_stats.jsonl` row per output episode;
  - no discarded source index is silently carried as an output index.

- [ ] Implement unique staging in the final output's parent filesystem, with state transitions `queued -> building`. The CLI contract is:

```bash
backend/.venv/bin/python backend/curation_export.py \
  --workspace /home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_curation \
  run --export-id 00000000-0000-0000-0000-000000000000

backend/.venv/bin/python backend/curation_export.py \
  --workspace /home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_curation \
  resume --export-id 00000000-0000-0000-0000-000000000000
```

As with the worker CLI, the zero UUID is a parser-test value. Add authenticated `POST /api/curation/exports` to freeze the snapshot and return the real export UUID plus exact run command, and `GET /api/curation/exports/{id}` for persisted status. The API never performs the long export in-process.

- [ ] Run:

```bash
backend/.venv/bin/python -m pytest -q \
  backend/tests/test_stable_tasks.py \
  backend/tests/test_exporter.py \
  backend/tests/test_export_lifecycle.py
```

Expected: PASS and source manifest unchanged before/after.

- [ ] Commit:

```bash
git add backend/curation/exporter.py backend/curation_export.py backend/curation/router.py \
  backend/tests/test_stable_tasks.py backend/tests/test_exporter.py \
  backend/tests/test_export_lifecycle.py
git diff --cached --check
git commit -m "feat: build deterministic cleaned v2 datasets"
```

## Task 13: Add structural, GR00T, provenance, checksum, and atomic publication gates

**Files:**

- Create: `backend/curation/validation.py`
- Create: `backend/curation/publication.py`
- Modify: `backend/curation/exporter.py`
- Modify: `backend/curation_export.py`
- Create: `backend/tests/test_structural_validation.py`
- Create: `backend/tests/test_gr00t_validation.py`
- Create: `backend/tests/test_provenance.py`
- Create: `backend/tests/test_publication.py`

- [ ] Write a structural test that covers every core gate in the approved design: snapshot approval/keep count, seven nonempty runs, full coverage, prompt resolution, parquet/metadata/video counts, contiguous indices, `info.json`, per-episode stats rows, untouched Arrow equality, independent regular-file copies, and unchanged source manifest. Persist a canonical `structural-report.json` in the workspace and copy it into staging only after success.

- [ ] Implement an injectable subprocess runner and test the exact GR00T statistics invocation with a canonical nonempty staging path:

```bash
cd /home/jihun/work/Isaac-GR00T
test -n "$STAGING_PATH"
.venv/bin/python gr00t/data/stats.py \
  --dataset-path "$STAGING_PATH" \
  --embodiment-tag UNITREE_G1_SONIC \
  --modality-config-path gr00t/configs/data/embodiment_configs.py
```

Record executable, cwd, exact argv, selected non-secret environment, repository commit/dirty state, exit code, stdout, stderr, start/end UTC times, and traceback in `gr00t-stats-report.json`. Failure leaves staging and advances no lifecycle state.

- [ ] Add a small generated validation script to the subprocess input that imports `MODALITY_CONFIGS["unitree_g1_sonic"]` and `LeRobotEpisodeLoader`, loads every staged episode, and compares `language.annotation.human.task_description` frame for frame to the parquet `task_index -> tasks.jsonl` mapping. Require the seven prompt runs in step order for every episode. Record per-episode row count, run list, pass/fail, exception, and traceback in `gr00t-loader-report.json`.

- [ ] Write `meta/curation_provenance.json` tests against the closed schema. Capture source manifest/original prompt, approval snapshot/templates, visualizer and Isaac-GR00T repository states, Cosmos contract/model/endpoint label/job/attempt/artifact hashes, source-output map, every kept/rejected decision, stable task map, and copied validation/contact-sheet artifacts. Dirty repository entries include tracked-diff SHA-256 and sorted untracked file records; secrets and raw reasoning are excluded.

- [ ] Enforce the write order:

```text
core structural report exists
-> GR00T stats report exists and passed
-> GR00T loader report exists and passed
-> selected contact sheets copied and hashed
-> provenance written
-> checksum written
-> files chmod 0444 and directories chmod 0555
-> every file/directory fsynced bottom-up and staging parent fsynced
-> read-only final consistency gate
-> export state publishing
-> renameat2(RENAME_NOREPLACE)
-> final parent fsync
-> export state published
```

No provenance field may reference a not-yet-existing report. The workspace-only final-consistency report is stored in SQLite but excluded from staging provenance/checksum to avoid a cycle.

- [ ] Write `meta/curation_checksums.sha256` over every output regular file except itself. Implement final consistency as read-only. Repeat structural/preservation checks, validate provenance artifact size/hash/media type/path, validate the checksum contains every other regular file exactly once in UTF-8 path order, reject unlisted files/symlinks/source hardlinks/post-checksum changes, and recheck the source manifest.

- [ ] Implement a small `ctypes` wrapper around Linux libc `renameat2` using `AT_FDCWD=-100` and `RENAME_NOREPLACE=1`. There is no `os.replace`, plain rename, check-then-rename, or copy fallback. Unsupported syscall/filesystem is fatal. `EEXIST` preserves staging and the competing destination and records `publish_destination_exists`.

- [ ] Write process/barrier publication tests for:
  - successful atomic rename, absent staging afterward, then parent fsync before `published` commit;
  - another process creating an empty final directory immediately before rename;
  - another process creating a final directory plus sentinel bytes immediately before rename;
  - crash immediately after rename but before parent fsync;
  - reconciliation with staging absent/final present: final gate, then parent fsync, then `published`;
  - injected parent-fsync failure remaining retryable `publishing` with `publish_parent_fsync_failed`;
  - both paths present and both paths absent becoming failed/operator-inspection states.

- [ ] Run:

```bash
backend/.venv/bin/python -m pytest -q \
  backend/tests/test_structural_validation.py \
  backend/tests/test_gr00t_validation.py \
  backend/tests/test_provenance.py \
  backend/tests/test_publication.py
```

Expected: PASS, including real multi-process race/crash tests.

- [ ] Commit:

```bash
git add backend/curation/validation.py backend/curation/publication.py \
  backend/curation/exporter.py backend/curation_export.py \
  backend/tests/test_structural_validation.py backend/tests/test_gr00t_validation.py \
  backend/tests/test_provenance.py backend/tests/test_publication.py
git diff --cached --check
git commit -m "feat: validate and atomically publish curated datasets"
```

## Task 14: Document setup and run all regression gates

**Files:**

- Modify: `backend/README.md`
- Create or modify: `.env.example`
- Create: `docs/pnp-trash-curation-runbook.md`
- Modify tests only if a gate reveals a curation regression

- [ ] Document every runtime variable, its secrecy, validation, and owning process. Use this non-secret example mapping verbatim:

```bash
export CURATION_REPO_ROOT=/home/jihun/work/GR00T-WholeBodyControl/worktrees/lerobot-dataset-visualizer-pnp-trash
export CURATION_DATASET_ALIASES_JSON='{"local/pnp_trash":"/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash"}'
export CURATION_WORKSPACE=/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_curation
export CURATION_OUTPUT=/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_cleaned
export CURATION_BROWSER_ORIGIN=http://127.0.0.1:3000
export CURATION_BACKEND_URL=http://127.0.0.1:8000
export NEXT_PUBLIC_DATASET_URL=http://127.0.0.1:8000/api/local-datasets
export ISAAC_GROOT_ROOT=/home/jihun/work/Isaac-GR00T
```

Document `CURATION_BEARER_TOKEN`, `COSMOS_BASE_URL`, `COSMOS_MODEL`, `COSMOS_API_KEY_ENV`, and `COSMOS_ENDPOINT_IDENTITY` by variable name only. Require their actual values to come from the operator's existing server/runtime configuration; do not commit them or expose them through `NEXT_PUBLIC_*`.

Document and test the process-specific loaders: FastAPI uses full
`CurationSettings`; the worker uses `WorkerSettings` without bearer, output,
browser origin, or Isaac root; the exporter uses `ExportSettings` without
bearer, Cosmos base/API-key name/target secret, browser origin, or output.
Aliases and workspace remain canonical and separated in every loader.
Document and test `env -i` launchers: the worker receives only its settings and
the dynamic target credential, while the exporter receives neither the bearer
nor Cosmos base/API-key name/dynamic target secret from the ambient shell.
Reject dynamic key-target collisions with inherited process names,
`CURATION_*`/`NEXT_PUBLIC_*`, or the named Cosmos/Isaac settings before a
backend or worker reads or forwards the target; a distinct conventional target
such as `COSMOS_API_KEY` remains valid.

- [ ] Document exact startup commands:

```bash
cd "$CURATION_REPO_ROOT"
backend/.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

```bash
cd "$CURATION_REPO_ROOT"
bun run dev --hostname 127.0.0.1 --port 3000
```

Use package-relative imports and keep this repository-root `backend.app:app` command identical in README, tests, and the runbook.

- [ ] Add a runbook preflight that checks loopback listeners, exact browser origin, source manifest count/hash, absence of the final output, workspace ownership/free space, `renameat2` support, Cosmos `/v1/models` model identity, one-episode capability smoke, Isaac-GR00T Python/config import, and FFmpeg/PyAV frame-count agreement.

- [ ] Run the complete backend gate:

```bash
cd "$CURATION_REPO_ROOT"
backend/.venv/bin/python -m pytest -q backend/tests
```

Expected: PASS, including the v3.1 regression module.

- [ ] Run the complete frontend gate:

```bash
cd "$CURATION_REPO_ROOT"
bun run format:check && bun run validate
```

Expected: PASS without rewriting the checkout. Re-run exact approved HEAD and
`git status --porcelain --untracked-files=all` authentication after install and
static gates, before starting runtime.

- [ ] Run a local-asset integration smoke with the backend bound to loopback. Verify `info.json`, a parquet footer range, an MP4 range, `HEAD`, traversal rejection, and authorization rejection. Then open:

```text
http://127.0.0.1:3000/local/pnp_trash/episode_0?tab=annotations
```

Verify metadata, parquet charts, video seeking, task-index mode, and no curation/HF secret in browser source or network requests.

- [ ] Inspect repository scope and commit docs/test-only corrections:

```bash
git status --short
git diff --check
git diff --stat
git add backend/README.md .env.example docs/pnp-trash-curation-runbook.md
git diff --cached --check
git commit -m "docs: add trash curation operations runbook"
```

## Task 15: Run Cosmos proposals and complete human review for all 92 episodes

**Files/artifacts:**

- Write only under: `/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_curation`
- Read only: `/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash`

- [ ] Start the configured FastAPI and Next.js services and open the workspace. Configured FastAPI startup creates the canonical source manifest and initializes `curation.sqlite3` before serving. Confirm the displayed source fingerprint matches the user-approved 2026-08-28 190-file manifest, including immutable ancillary `pnp_trash.xlsx`, and the review summary is 92 `pending` on a fresh workspace. Task 14's temporary tests do not complete this real approved-source loopback/browser smoke; it remains pending until secure runtime configuration is loaded here.

- [ ] Run Cosmos capability preflight and pin source episode 4 for the representative smoke. Before creating its batch, require the operator-provided lexical source path to equal the configured `local/pnp_trash` alias, resolve that path to its existing canonical directory (supporting the approved `outputs` ancestor symlink), and pass the canonical path to the registered-source validator; changed aliases, dangling targets, and non-directory targets fail before the POST. Authenticate its approved-source metadata, parquet, and video through the persisted manifest and production alignment proof: exactly 2,060 frames at 50 fps, 41.2 seconds, 83 samples, within 120 seconds/240 samples. Episode 0 is ineligible at 6,435 frames/128.7 seconds. The smoke passes only with job state `completed`, a status `job_id` exactly equal to the nonempty ID returned by that smoke POST, exactly one `succeeded` attempt, zero `manual_only`/`retryable`, and active-proposal coverage one. Capture the exact attempt ID; require regular non-symlink `request.json`, initial `response.txt`, optional `repair-response.txt`, `parsed.json`, proposal contact-sheet PNG, and receipt. Resolve the authoritative raw response through `parsed.raw_response_sha256`; enforce the exact closed request/parsed schemas, rerun `prove_alignment_and_select` over the authenticated full float parquet timeline, compare exact selected indices and actual `p[i]` timestamps, require the registered source-video hash, and bind the receipt to dataset/source/proposal/episode/transition frames/path/PNG hash and size. Inspect the authoritative raw response, parsed v2 result, sampling, contact sheet, and UI rendering. Enter the runbook's exact operator confirmation and freeze its canonical evidence authority bound to the smoke job and attempt IDs.

- [ ] Only after the shell contains the exact evidence/UI confirmation and canonical authority bound to the successful smoke job and attempt IDs, refetch and revalidate that smoke and all frozen artifact hashes, again requiring its status `job_id` to equal the confirmed smoke POST ID. Then create a batch for all 92 episodes through `POST /api/curation/batches`; before launching its worker, require the full status to carry the exact nonempty job ID just returned by the POST and exactly 92 closed queued attempt rows, compare the fresh configuration's source/prompt/model/endpoint/transport/sampling/limits with the smoke authority, and require the exact episode set `0..91`. Run:

```bash
run_curation_worker \
  --workspace /home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_curation \
  run --job-id "$JOB_ID"
```

Before executing, copy the API's returned job ID into `JOB_ID` and require `test -n "$JOB_ID"`. Monitor the persisted API status; use the documented cancel or retry route only for explicit operator decisions. Transport/schema failures become `manual_only`, never automatic rejection.

- [ ] In the visualizer, manually review every source episode. For each episode:
  1. inspect the full video and proposal/contact sheet;
  2. reject corrupt, incomplete, unsuccessful, or out-of-order attempts, optionally recording a reason; or
  3. enter normalized object text, pickup hand, turn direction, and six frame starts;
  4. compare grasp/release diagnostics and inspect every boundary frame;
  5. save a draft, then explicitly approve keep or reject;
  6. use approve-and-next until no pending/draft episode remains.

An episode is kept only when all seven steps complete in order, including the final standing-straight pose. Cosmos never supplies approval.

- [ ] Inspect the dataset audit. Resolve every order/coverage error and unreadable file. Review transition/duration outliers and every grip disagreement above 2.0 seconds. The final summary must satisfy:

```text
pending = 0
draft = 0
approved_keep >= 1
approved_keep + approved_reject = 92
invalid approved_keep = 0
```

- [ ] Record the immutable approval snapshot hash returned by export creation. Do not reopen or edit decisions for the in-progress export; later edits belong to a new export snapshot.

## Task 16: Build, validate, and atomically publish `pnp_trash_cleaned`

**Files/artifacts:**

- Staging: unique sibling of `/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_cleaned`
- Final: `/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_cleaned`
- Reports: `/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_curation/exports/{export-id}/`

- [ ] Confirm the final path does not exist, the source manifest still matches, all 92 episodes are approved, and no export is active. Create the export through `POST /api/curation/exports`; require the returned approval snapshot hash and command.

- [ ] Run the returned export command only after verifying its UUID variable is nonempty:

```bash
test -n "$EXPORT_ID"
run_curation_exporter \
  --workspace /home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_curation \
  run --export-id "$EXPORT_ID"
```

- [ ] Monitor the persisted lifecycle in order:

```text
queued
building
core_structural_validated
gr00t_stats_validated
gr00t_loader_validated
provenance_written
final_consistency_validated
publishing
published
```

Any failure must leave the source and final path unchanged and retain staging/reports for diagnosis. Use `curation_export.py resume` only for the documented crash-recovery states.

- [ ] After publication, independently verify:

```bash
test -d /home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_cleaned
test ! -w /home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_cleaned/meta/info.json
cd "$CURATION_REPO_ROOT"
backend/.venv/bin/python -m pytest -q backend/tests
bun run validate
```

Then validate every line of `meta/curation_checksums.sha256`, confirm the source manifest is unchanged, and inspect structural, GR00T stats, loader, provenance, and final-consistency reports. The loader report must show all retained episodes and exactly seven prompt runs in order, frame-for-frame equal to parquet task indices.

- [ ] Final handoff records:
  - source and final canonical paths;
  - source manifest SHA-256 and approval snapshot SHA-256;
  - kept/rejected counts and retained frame count;
  - stable `tasks.jsonl` count/hash;
  - Cosmos model/endpoint identity and contributing job/attempt IDs;
  - visualizer and Isaac-GR00T commits/dirty provenance;
  - hashes and pass states of all four validation reports;
  - published export UUID and timestamp.

Do not publish to Hugging Face and do not delete staging/workspace evidence as part of this plan.

## Final acceptance checklist

- [ ] Every approved design requirement has a test or an explicit human checkpoint above.
- [ ] All 92 source episodes have final human decisions.
- [ ] At least one episode is kept; every kept episode has seven exhaustive nonempty runs.
- [ ] Frozen training prompts come only from the backend prompt generator.
- [ ] No browser request to local assets contains the HF OAuth token.
- [ ] No browser bundle contains the curation bearer or Cosmos credential.
- [ ] Source bytes and source manifest are unchanged.
- [ ] Untouched Arrow columns and every retained video are byte/value preserving as specified.
- [ ] The cleaned dataset contains no symlink or source hardlink.
- [ ] Structural, GR00T stats, GR00T loader, and final consistency gates pass.
- [ ] Provenance references only existing, hashed artifacts.
- [ ] No-clobber publication and parent-directory fsync complete before SQLite says `published`.
- [ ] Backend tests and `bun run validate` pass, including the v3.1 regression gate.
