# H100 GR00T N1.7 PnP Trash Training Design

## Objective

Create an isolated Docker environment on `h100` and launch a real single-GPU
GR00T N1.7 fine-tuning run on the transferred full-prompt PnP trash dataset.
The host must not receive package, Python-environment, CUDA, or framework
installations.

## Confirmed inputs

- Host: SSH alias `h100` (`mncsvr11`), user `kube`.
- GPU: physical GPU 7, NVIDIA H100 80 GB HBM3. The launch gate requires less
  than 5 GiB allocated on GPU 7 immediately before both smoke and production
  training.
- Dataset: `/mnt/data01/jhkim/datasets/pnp_trash`.
  - LeRobot v2.1
  - annotation variant `full_prompt`
  - 72 episodes
  - 154,625 frames
  - 72 videos
  - 9 tasks
  - 50 FPS
- Hugging Face cache: `/mnt/data01/huggingface`; the N1.7 3B cache is already
  present.
- W&B credential: `/mnt/data01/jhkim/secrets/wandb_api_key`, nonempty, mode
  `600`. Its contents must never appear in commands, logs, Docker metadata, or
  committed files.
- Isaac-GR00T checkout: `/home/kube/jihun/Isaac-GR00T`, clean commit
  `626af89d3e914ec92eab5323e23b9ed44a7b26c8`.

## Chosen approach

Build a fresh image from the clean, pinned Isaac-GR00T checkout and its official
`docker/Dockerfile`. This is preferred over the existing `gr00t_pnp:latest`
image because the existing image's `uv.lock`, `pyproject.toml`, and fine-tuning
launcher hashes differ from the pinned checkout. Installing dependencies when
the container starts is also rejected because it is slower and less
reproducible.

The Docker build may download packages into Docker layers and caches. It must
not run `apt`, `pip`, `uv`, or similar installers directly on the host.

## Image and container contract

- Image tag: `jihun/gr00t-n1.7:626af89`
- Container name:
  `jihun_gr00t_n17_pnp_trash_full_prompt_gpu7_20260828`
- Container lifecycle: detached, persistent `sleep infinity`, no automatic
  restart policy.
- GPU exposure: `--gpus device=7`; inside the container the only visible GPU is
  CUDA device 0.
- Runtime settings:
  - `--init`
  - `--ipc=host`
  - `--ulimit memlock=-1`
  - `--ulimit stack=67108864`
- Mounts:
  - `/mnt/data01/jhkim/datasets/pnp_trash:/dataset:ro`
  - `/mnt/data01/huggingface:/root/.cache/huggingface:rw`
  - `/mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_n17_20260828:/outputs:rw`
  - `/mnt/data01/jhkim/secrets/wandb_api_key:/run/secrets/wandb_api_key:ro`

The dataset is mounted read-only so neither statistics generation nor training
can alter the published dataset. The source checkout is baked into the image;
it is not bind-mounted over `/workspace`, which would hide the image's pinned
virtual environment.

## Secret handling and W&B

The API key stays in the read-only secret mount. A shell running inside the
container reads it immediately before authentication or training:

```bash
export WANDB_API_KEY="$(tr -d '\r\n' </run/secrets/wandb_api_key)"
```

The literal key is never interpolated into `docker run`, stored in the
container's Docker-config environment, printed, or committed. Authentication is
verified through the W&B client with output restricted to a success/failure
status.

- W&B mode: `online`
- Project: `gr00t-n1.7-pnp-trash`
- Smoke run name: `pnp-trash-full-prompt-gpu7-smoke-20260828`
- Production run name: `pnp-trash-full-prompt-gpu7-20260828`
- Entity/account: inferred from the API key.

## Validation gate

No production training process starts until every gate succeeds.

1. Recheck that the remote checkout is clean at commit `626af89`, the dataset
   metadata matches the confirmed counts, all expected Parquet/video files are
   present, the output directory is dedicated to this experiment, and GPU 7
   has less than 5 GiB allocated.
2. Build the image from the official Dockerfile and verify its image ID and
   source/lockfile hashes.
3. Create the persistent container with the exact mounts and runtime settings.
4. Inside the container, verify:
   - exactly one H100 is visible;
   - PyTorch reports CUDA 12.8 and CUDA availability;
   - GR00T imports from `/workspace`;
   - the `UNITREE_G1_SONIC` modality configuration loads;
   - the W&B key authenticates without being printed;
   - the N1.7 base-model snapshot is reachable from the mounted cache.
5. Run a read-only GR00T loader probe across all 72 episodes. It must load
   154,625 frames, resolve every `task_index`, and observe one nonempty language
   run per episode.
6. Run a one-step fine-tuning smoke test using the production model, dataset,
   embodiment, batch size, augmentations, and W&B integration. It writes only
   to `/outputs/smoke`. The smoke gate requires exit code zero, a finite loss,
   and a saved checkpoint/config.

Any failure stops the workflow before production launch. Existing H100
containers are not stopped, removed, or modified.

## Production fine-tuning run

Use the official `examples/finetune.sh` launcher with:

- Base model: `nvidia/GR00T-N1.7-3B`
- Dataset: `/dataset`
- Embodiment: `UNITREE_G1_SONIC`
- Modality config: `gr00t/configs/data/embodiment_configs.py`
- GPUs: 1
- Physical GPU: 7 (visible as CUDA device 0 in the container)
- Trainable components: projector and diffusion action head; LLM and visual
  backbone remain frozen.
- Max steps: 20,000
- Global batch size: 32
- Gradient accumulation: 1
- Learning rate: `1e-4`
- Warmup ratio: `0.05`
- Weight decay: `1e-5`
- State dropout: `0.2`
- Data loader workers: 4
- Episode sampling rate: `0.1`
- Shard size: 1,024
- Shards per epoch: 100,000 (official launcher default)
- Color jitter: brightness `0.3`, contrast `0.4`, saturation `0.5`, hue `0.08`
- Save every 1,000 steps
- Retain five checkpoints
- Save full resumable trainer state; do not use `--save-only-model`
- W&B enabled in online mode
- Output: `/outputs/train`
- Log: `/outputs/train.log`
- Exit-status record: `/outputs/train.exit`

The production process is launched detached inside the persistent container.
The launch is considered confirmed only after the process remains alive and the
log reaches model/data initialization or the first optimizer step without an
exception. Completion of 20,000 steps is not required for setup handoff.

## Failure handling and recovery

- Image-build failure leaves the dataset and existing containers untouched.
- Container/preflight failure prevents the smoke and production commands.
- Smoke failure prevents production training and preserves `/outputs/smoke`
  and its log for diagnosis.
- Production failure preserves all checkpoints, W&B metadata, logs, and exit
  status. Because full trainer state is saved, a later recovery can resume from
  the latest complete checkpoint.
- No output or container is deleted automatically.

## Acceptance criteria

The setup is complete when:

1. the pinned image exists;
2. the named GPU-7 container is running with the exact mounts;
3. the secret-safe W&B authentication, GPU/import, full dataset loader, and
   one-step training gates pass;
4. the 20,000-step W&B-enabled production process has been launched;
5. its PID, W&B run identity/URL, output path, log path, and initial status are
   reported without exposing credentials; and
6. no package or framework was installed on the H100 host.
