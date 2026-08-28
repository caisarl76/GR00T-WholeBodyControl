# H100 GR00T N1.7 PnP Trash Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build a pinned Isaac-GR00T N1.7 Docker image on h100, validate the full-prompt PnP trash dataset and W&B-enabled training path on GPU 7, then launch a resumable 20,000-step run.

**Architecture:** The clean H100 Isaac-GR00T checkout is baked into a fresh image, so every dependency installation occurs inside Docker layers. A persistent GPU-7 container receives four explicit mounts: read-only dataset, shared Hugging Face cache, dedicated run output, and read-only W&B secret. Preflight, the real GR00T loader, and a one-step W&B smoke run gate the detached production launch.

**Tech Stack:** Docker 28.4, NVIDIA Container Runtime, H100 80 GB, CUDA 12.8, Python 3.10, PyTorch 2.7, uv, Isaac-GR00T commit 626af89, LeRobot v2.1, W&B online logging, SSH.

---

## File and state map

- Read: /home/kube/jihun/Isaac-GR00T on h100 — clean pinned image build context.
- Read-only mount: /mnt/data01/jhkim/datasets/pnp_trash — 72-episode full-prompt dataset.
- Read-write mount: /mnt/data01/huggingface — shared model cache.
- Read-only mount: /mnt/data01/jhkim/secrets/wandb_api_key — W&B credential.
- Create: /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_n17_20260828 — build, smoke, checkpoint, log, and status evidence.
- Create after launch: docs/superpowers/progress/2026-08-28-h100-gr00t-n17-pnp-trash-training.md — secret-free handoff.
- Do not modify: dataset contents, existing H100 containers, H100 host packages, or the H100 Isaac-GR00T checkout.

### Task 1: Run fail-closed H100 preflight

**Files:**
- Read: /home/kube/jihun/Isaac-GR00T/.git
- Read: /mnt/data01/jhkim/datasets/pnp_trash/meta
- Read metadata only: /mnt/data01/jhkim/secrets/wandb_api_key

- [ ] **Step 1: Verify pinned source, paths, secret metadata, and non-collision state**

Run from /home/jihun/work/GR00T-WholeBodyControl:

~~~bash
ssh h100 'set -euo pipefail
test "$(git -C /home/kube/jihun/Isaac-GR00T rev-parse HEAD)" = "626af89d3e914ec92eab5323e23b9ed44a7b26c8"
test -z "$(git -C /home/kube/jihun/Isaac-GR00T status --porcelain)"
test -d /mnt/data01/jhkim/datasets/pnp_trash
test -d /mnt/data01/huggingface
test -s /mnt/data01/jhkim/secrets/wandb_api_key
test "$(stat -c %a /mnt/data01/jhkim/secrets/wandb_api_key)" = "600"
test "$(find /mnt/data01/jhkim/datasets/pnp_trash -type f -name "*.part" | wc -l)" = "0"
test "$(docker ps -a --filter name=^/jihun_gr00t_n17_pnp_trash_full_prompt_gpu7_20260828$ --format "{{.Names}}" | wc -l)" = "0"
if docker image inspect jihun/gr00t-n1.7:626af89 >/dev/null 2>&1; then
  echo "image tag already exists: jihun/gr00t-n1.7:626af89" >&2
  exit 1
fi
test ! -e /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_n17_20260828
echo preflight-paths-ok'
~~~

Expected: preflight-paths-ok. A nonzero exit stops execution without changing remote state.

- [ ] **Step 2: Verify exact dataset metadata and file counts**

Run:

~~~bash
ssh h100 python3 - <<'PY'
import json
from pathlib import Path
root = Path("/mnt/data01/jhkim/datasets/pnp_trash")
info = json.loads((root / "meta/info.json").read_text())
provenance = json.loads((root / "meta/annotation_provenance.json").read_text())
assert info["codebase_version"] == "v2.1"
assert info["total_episodes"] == 72
assert info["total_frames"] == 154625
assert info["total_tasks"] == 9
assert info["total_videos"] == 72
assert info["fps"] == 50
assert provenance["variant"] == "full_prompt"
assert len(list((root / "data").rglob("*.parquet"))) == 72
assert len(list((root / "videos").rglob("*.mp4"))) == 72
assert sum(bool(line.strip()) for line in (root / "meta/episodes.jsonl").read_text().splitlines()) == 72
print("dataset-contract-ok")
PY
~~~

Expected: dataset-contract-ok.

- [ ] **Step 3: Require less than 5 GiB allocation on physical GPU 7**

Run:

~~~bash
ssh h100 python3 - <<'PY'
import subprocess
used = subprocess.check_output(
    ["nvidia-smi", "--id=7", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
    text=True,
)
used_mib = int(used.strip())
assert used_mib < 5120, f"GPU 7 already has {used_mib} MiB allocated"
print(f"gpu7-preflight-ok used_mib={used_mib}")
PY
~~~

Expected: gpu7-preflight-ok with less than 5,120 MiB.

### Task 2: Build and verify the pinned image

**Files:**
- Read: /home/kube/jihun/Isaac-GR00T/docker/Dockerfile
- Read: /home/kube/jihun/Isaac-GR00T/uv.lock
- Create: /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_n17_20260828/build.log

- [ ] **Step 1: Create the dedicated output root**

~~~bash
ssh h100 'mkdir -p /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_n17_20260828'
~~~

Expected: the directory exists. This is data placement, not a host package installation.

- [ ] **Step 2: Build the official Dockerfile and retain its log**

~~~bash
ssh h100 'set -euo pipefail
cd /home/kube/jihun/Isaac-GR00T
DOCKER_BUILDKIT=1 docker build --network host \
  -f docker/Dockerfile \
  -t jihun/gr00t-n1.7:626af89 \
  . 2>&1 | tee /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_n17_20260828/build.log'
~~~

Expected: exit zero and a successful export of jihun/gr00t-n1.7:626af89. apt, pip, and uv run only in Docker build stages.

- [ ] **Step 3: Record immutable image metadata**

~~~bash
ssh h100 'docker image inspect jihun/gr00t-n1.7:626af89 --format "image_id={{.Id}} created={{.Created}} size={{.Size}}" | tee /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_n17_20260828/image.txt'
~~~

Expected: a sha256 image ID and nonzero size.

### Task 3: Create and audit the GPU-7 container

**Files:**
- Create: Docker container jihun_gr00t_n17_pnp_trash_full_prompt_gpu7_20260828
- Create: /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_n17_20260828/container-inspect.json

- [ ] **Step 1: Create the persistent container**

~~~bash
ssh h100 'docker run -d \
  --name jihun_gr00t_n17_pnp_trash_full_prompt_gpu7_20260828 \
  --gpus device=7 \
  --init \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --label gr00t.source_commit=626af89d3e914ec92eab5323e23b9ed44a7b26c8 \
  --label gr00t.dataset_variant=full_prompt \
  -v /mnt/data01/jhkim/datasets/pnp_trash:/dataset:ro \
  -v /mnt/data01/huggingface:/root/.cache/huggingface:rw \
  -v /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_n17_20260828:/outputs:rw \
  -v /mnt/data01/jhkim/secrets/wandb_api_key:/run/secrets/wandb_api_key:ro \
  jihun/gr00t-n1.7:626af89 \
  sleep infinity'
~~~

Expected: one container ID. The secret value is absent from both the command and Docker environment metadata.

- [ ] **Step 2: Persist and audit effective settings**

~~~bash
ssh h100 'docker inspect jihun_gr00t_n17_pnp_trash_full_prompt_gpu7_20260828 > /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_n17_20260828/container-inspect.json; docker inspect jihun_gr00t_n17_pnp_trash_full_prompt_gpu7_20260828 --format "running={{.State.Running}} image={{.Image}} ipc={{.HostConfig.IpcMode}} restart={{.HostConfig.RestartPolicy.Name}} gpu={{json .HostConfig.DeviceRequests}} mounts={{range .Mounts}}{{.Destination}}:rw={{.RW}} {{end}}"'
~~~

Expected: running true, IPC host, no restart policy, device ID 7 only, dataset/secret read-only, and cache/output read-write.

- [ ] **Step 3: Verify baked source hashes**

~~~bash
ssh h100 'docker exec jihun_gr00t_n17_pnp_trash_full_prompt_gpu7_20260828 sha256sum /workspace/uv.lock /workspace/pyproject.toml /workspace/gr00t/experiment/launch_finetune.py /workspace/gr00t/configs/data/embodiment_configs.py'
~~~

Expected hashes in order:

~~~text
ca340f1947eff0636173ec26880ab33209678ad378b260d2db46314eb0260813
2ab3c128bbd6c6f4e84446e8538574f101d294bc4c29de31cc1508f435de277d
4b08497efad45461e77968536844ba885d5a83470e089c86b4e625636c45d627
4c7626426cfd184034df772bca9f1ab5c2014c139c388f833012ca98ac6f948a
~~~

### Task 4: Pass runtime, W&B, cache, and full-loader gates

**Files:**
- Read-only: /dataset and /run/secrets/wandb_api_key
- Read-write: /root/.cache/huggingface

- [ ] **Step 1: Verify exactly one H100 and the pinned runtime**

~~~bash
ssh h100 'docker exec -i jihun_gr00t_n17_pnp_trash_full_prompt_gpu7_20260828 python -' <<'PY'
import torch
import gr00t
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
assert "H100" in torch.cuda.get_device_name(0)
assert torch.version.cuda == "12.8"
assert "unitree_g1_sonic" in MODALITY_CONFIGS
assert str(gr00t.__file__).startswith("/workspace/")
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), gr00t.__file__)
PY
~~~

Expected: PyTorch 2.7, CUDA 12.8, one H100, and GR00T under /workspace.

- [ ] **Step 2: Authenticate W&B without printing the credential**

~~~bash
ssh h100 'docker exec -i jihun_gr00t_n17_pnp_trash_full_prompt_gpu7_20260828 bash -s' <<'BASH'
set -euo pipefail
export WANDB_API_KEY="$(python -c 'from pathlib import Path; value=Path("/run/secrets/wandb_api_key").read_text().strip(); print(value.split("=", 1)[1] if value.startswith("WANDB_API_KEY=") else value)')"
test -n "$WANDB_API_KEY"
export WANDB_MODE=online
python -c 'import os, wandb; ok=wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True, verify=True); assert ok is not False; print("wandb-auth-ok")'
BASH
~~~

Expected: W&B status and wandb-auth-ok, with no credential value.

- [ ] **Step 3: Verify the base model is already reachable from the cache**

~~~bash
ssh h100 'docker exec -i jihun_gr00t_n17_pnp_trash_full_prompt_gpu7_20260828 python -' <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download("nvidia/GR00T-N1.7-3B", local_files_only=True)
print("base-model-cache-ok", path)
PY
~~~

Expected: base-model-cache-ok beneath /root/.cache/huggingface and no download.

- [ ] **Step 4: Load all 72 episodes through the real GR00T Parquet/language path**

~~~bash
ssh h100 'docker exec -i jihun_gr00t_n17_pnp_trash_full_prompt_gpu7_20260828 python -' <<'PY'
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
loader = LeRobotEpisodeLoader("/dataset", MODALITY_CONFIGS["unitree_g1_sonic"])
assert len(loader) == 72
assert len(loader.tasks_map) == 9
total_frames = 0
column = "language.annotation.human.task_description"
for episode_index in range(len(loader)):
    frame = loader._load_parquet_data(episode_index)
    values = frame[column].tolist()
    assert values and all(isinstance(value, str) and value for value in values)
    assert 1 + sum(left != right for left, right in zip(values, values[1:])) == 1
    total_frames += len(frame)
assert total_frames == 154625
stats = loader.get_dataset_statistics()
assert "state" in stats and "action" in stats
print("loader-contract-ok", len(loader), total_frames, len(loader.tasks_map))
PY
~~~

Expected: loader-contract-ok 72 154625 9.

### Task 5: Run the W&B-enabled one-step smoke gate

**Files:**
- Create: /outputs/smoke.log and /outputs/smoke.exit
- Create: /outputs/smoke/pnp-trash-full-prompt-gpu7-smoke-20260828

- [ ] **Step 1: Repeat the GPU allocation gate**

Repeat Task 1 Step 3.

Expected: GPU 7 remains below 5,120 MiB.

- [ ] **Step 2: Run one optimizer step with production-equivalent settings**

~~~bash
ssh h100 'docker exec -i jihun_gr00t_n17_pnp_trash_full_prompt_gpu7_20260828 bash -s' <<'BASH'
set -uo pipefail
export WANDB_API_KEY="$(python -c 'from pathlib import Path; value=Path("/run/secrets/wandb_api_key").read_text().strip(); print(value.split("=", 1)[1] if value.startswith("WANDB_API_KEY=") else value)')"
export WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 USE_WANDB=1
export MAX_STEPS=1 SAVE_STEPS=1 GLOBAL_BATCH_SIZE=32 DATALOADER_NUM_WORKERS=4
export EPISODE_SAMPLING_RATE=0.1 SHARD_SIZE=1024 NUM_SHARDS_PER_EPOCH=100000
cd /workspace
set +e
bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /dataset \
  --embodiment-tag UNITREE_G1_SONIC \
  --modality-config-path gr00t/configs/data/embodiment_configs.py \
  --output-dir /outputs/smoke \
  --experiment-name pnp-trash-full-prompt-gpu7-smoke-20260828 \
  --wandb-project gr00t-n1.7-pnp-trash \
  --state-dropout-prob 0.2 \
  2>&1 | tee /outputs/smoke.log
GR00T_SMOKE_STATUS=${PIPESTATUS[0]}
printf "%s\n" "$GR00T_SMOKE_STATUS" | tee /outputs/smoke.exit
exit "$GR00T_SMOKE_STATUS"
BASH
~~~

Expected: exit zero, W&B online initialization, one finite loss, and checkpoint/config output.

- [ ] **Step 3: Validate smoke artifacts and finite loss**

~~~bash
ssh h100 'set -euo pipefail
test "$(cat /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_n17_20260828/smoke.exit)" = "0"
test -d /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_n17_20260828/smoke/pnp-trash-full-prompt-gpu7-smoke-20260828/experiment_cfg
test -d /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_n17_20260828/smoke/pnp-trash-full-prompt-gpu7-smoke-20260828/checkpoint-1
grep -Eq "loss.*[0-9]" /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_n17_20260828/smoke.log
if grep -Eiq "loss.*(nan|inf)" /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_n17_20260828/smoke.log; then exit 1; fi
echo smoke-gate-ok'
~~~

Expected: smoke-gate-ok. Production launch is forbidden otherwise.

### Task 6: Launch and confirm production training

**Files:**
- Create: /outputs/train.log and /outputs/train.shell.pid
- Create on exit: /outputs/train.exit
- Create: /outputs/train/pnp-trash-full-prompt-gpu7-20260828

- [ ] **Step 1: Re-run GPU and smoke gates**

Run Task 1 Step 3 and Task 5 Step 3 again.

Expected: both exit zero.

- [ ] **Step 2: Launch the 20,000-step W&B run detached**

~~~bash
ssh h100 'docker exec -i jihun_gr00t_n17_pnp_trash_full_prompt_gpu7_20260828 tee /outputs/train.sh >/dev/null' <<'BASH'
#!/usr/bin/env bash
set -uo pipefail
printf "%s\n" "$$" > /outputs/train.shell.pid
export WANDB_API_KEY="$(python -c 'from pathlib import Path; value=Path("/run/secrets/wandb_api_key").read_text().strip(); print(value.split("=", 1)[1] if value.startswith("WANDB_API_KEY=") else value)')"
export WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 USE_WANDB=1
export MAX_STEPS=20000 SAVE_STEPS=1000 GLOBAL_BATCH_SIZE=32 DATALOADER_NUM_WORKERS=4
export EPISODE_SAMPLING_RATE=0.1 SHARD_SIZE=1024 NUM_SHARDS_PER_EPOCH=100000
cd /workspace
set +e
bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /dataset \
  --embodiment-tag UNITREE_G1_SONIC \
  --modality-config-path gr00t/configs/data/embodiment_configs.py \
  --output-dir /outputs/train \
  --experiment-name pnp-trash-full-prompt-gpu7-20260828 \
  --wandb-project gr00t-n1.7-pnp-trash \
  --state-dropout-prob 0.2 \
  > /outputs/train.log 2>&1
GR00T_TRAIN_STATUS=$?
printf "%s\n" "$GR00T_TRAIN_STATUS" > /outputs/train.exit
exit "$GR00T_TRAIN_STATUS"
BASH
ssh h100 'docker exec jihun_gr00t_n17_pnp_trash_full_prompt_gpu7_20260828 chmod 700 /outputs/train.sh'
ssh h100 'docker exec -d jihun_gr00t_n17_pnp_trash_full_prompt_gpu7_20260828 bash /outputs/train.sh'
~~~

Expected: SSH returns after detached exec creation and train.exit does not yet exist.

- [ ] **Step 3: Confirm process, GPU allocation, W&B identity, and initialization**

Poll no more than 60 seconds apart:

~~~bash
ssh h100 'docker top jihun_gr00t_n17_pnp_trash_full_prompt_gpu7_20260828 -eo pid,ppid,etime,args; nvidia-smi --id=7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader; tail -100 /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_n17_20260828/train.log; if test -f /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_n17_20260828/train.exit; then printf "train_exit="; cat /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_n17_20260828/train.exit; fi'
~~~

Expected: live Python training process, GPU 7 allocation, W&B run identity/URL, and data/model initialization or a finite first-step metric. A nonzero train.exit is failure.

### Task 7: Record a secret-free handoff

**Files:**
- Create: docs/superpowers/progress/2026-08-28-h100-gr00t-n17-pnp-trash-training.md

- [ ] **Step 1: Collect secret-free final evidence**

~~~bash
ssh h100 'docker inspect jihun_gr00t_n17_pnp_trash_full_prompt_gpu7_20260828 --format "container={{.Name}} image={{.Image}} running={{.State.Running}} gpu={{json .HostConfig.DeviceRequests}}"; docker top jihun_gr00t_n17_pnp_trash_full_prompt_gpu7_20260828 -eo pid,ppid,etime,args; nvidia-smi --id=7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader; grep -E "wandb:.*(run|project|https://wandb.ai)|loss|train_runtime" /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_n17_20260828/train.log | tail -30'
~~~

Expected: container/image, active process, GPU, W&B URL/identity, and training status without credentials.

- [ ] **Step 2: Write the progress record using apply_patch**

Create docs/superpowers/progress/2026-08-28-h100-gr00t-n17-pnp-trash-training.md with the exact observed container/image IDs, physical GPU 7 mapping, dataset counts, mounts, W&B project/run URL, smoke result, production PID/status, output/log paths, monitoring commands, and the explicit statement that no host packages were installed.

- [ ] **Step 3: Scan, verify, and commit only the handoff**

~~~bash
if rg -n "API_KEY|api[_-]?key|Authorization|Bearer" docs/superpowers/progress/2026-08-28-h100-gr00t-n17-pnp-trash-training.md; then
  echo "credential-like text found in progress record" >&2
  exit 1
fi
git diff --check
git add docs/superpowers/progress/2026-08-28-h100-gr00t-n17-pnp-trash-training.md
git commit -m "docs: record H100 GR00T pnp trash training"
~~~

Expected: no credential value, clean diff check, and a commit containing only the progress record.
