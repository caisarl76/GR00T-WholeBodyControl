# 2026-06-12 Point-Block VLA Data Review, Training, And Deployment

## Context

Goal: cleanse and verify the point-block LeRobot datasets, merge them into a GR00T training dataset, train a `UNITREE_G1_SONIC` VLA checkpoint on the H100 server, and verify safe inference ramping on sim and the real Unitree G1.

Internal LAN addresses are intentionally omitted from this note. Use the current H100 host, robot PC2 host, and workstation host values from the active deployment environment.

## Dataset Review And Cleansing

- Added the data cleansing and review UI modules:
  - `gear_sonic/data/cleanse_lerobot_dataset.py`
  - `gear_sonic/data/episode_review.py`
  - `gear_sonic/scripts/cleanse_lerobot_dataset.py`
  - `gear_sonic/scripts/review_lerobot_episodes.py`
- Review UI supports recorded ego-view playback, prompt/status display, joint velocity plotting, joint-group toggles, episode table navigation, trim, and move-to-trash.
- Trim semantics: each episode stores only the final accepted trim. Re-trimming the same episode always trims from the original episode, not from the previous trimmed version.
- Move-to-trash asks for confirmation before moving an episode.
- The point-block source set includes `outputs/point_block`, `outputs/point_green_block`, and `outputs/point_yellow_block`. In this run, the missing `point_block` source was under the primary checkout at `~/work/GR00T-WholeBodyControl/outputs/point_block`.

## Training Dataset

- Merged training dataset name: `point_block_260611`.
- H100 container mount path: `/datasets/point_block_260611`.
- Dataset stats command:

```bash
cd /workspace
export DATASET=/datasets/point_block_260611

python gr00t/data/stats.py \
  --dataset-path "$DATASET" \
  --embodiment-tag UNITREE_G1_SONIC
```

## H100 Training Notes

Smoke-test command that completed:

```bash
cd /workspace

export DATASET=/datasets/point_block_260611
export OUT=/workspace/outputs/point_block_260611_vla_test10
export NO_ALBUMENTATIONS_UPDATE=1
mkdir -p "$OUT"

uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DATASET" \
  --embodiment-tag UNITREE_G1_SONIC \
  --modality-config-path gr00t/configs/data/embodiment_configs.py \
  --num-gpus 1 \
  --output-dir "$OUT" \
  --save-total-limit 2 \
  --save-steps 10 \
  --max-steps 10 \
  --global-batch-size 4 \
  --dataloader-num-workers 2 \
  --shard-size 512 \
  --episode-sampling-rate 0.1 \
  --num-shards-per-epoch 20 \
  --no-use-wandb \
  2>&1 | tee "$OUT/train.log"
```

Observed smoke-test result:

- `100%|...| 10/10`
- `Training completed!`
- `train_loss`: approximately `1.2153`
- Runtime: approximately `47s`

Failure signatures and fixes:

- `AssertionError: global_batch_size must be divisible by num_gpus`: set `--global-batch-size` divisible by `--num-gpus`.
- `IndexError: list index out of range` in `sharded_mixture_dataset.py` with `--num-shards-per-epoch 1`: the dataset generated 20 shards, so use `--num-shards-per-epoch 20` for this dataset/shard setting.
- `tee: .../train.log: No such file or directory`: create `$OUT` before piping through `tee`.

Full training run used:

```bash
cd /workspace

export DATASET=/datasets/point_block_260611
export OUT=/outputs/point_block_260611_vla_260611
export NO_ALBUMENTATIONS_UPDATE=1
mkdir -p "$OUT"

uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DATASET" \
  --embodiment-tag UNITREE_G1_SONIC \
  --modality-config-path gr00t/configs/data/embodiment_configs.py \
  --num-gpus 1 \
  --output-dir "$OUT" \
  --save-total-limit 5 \
  --save-steps 1000 \
  --max-steps 20000 \
  --global-batch-size 4 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader-num-workers 2 \
  --shard-size 512 \
  --episode-sampling-rate 0.1 \
  --num-shards-per-epoch 20 \
  --no-use-wandb \
  2>&1 | tee "$OUT/train.log"
```

## Inference Server Notes

- The base checkpoint `nvidia/GR00T-N1.7-3B` does not support `UNITREE_G1_SONIC` directly. It fails with:

```text
ValueError: Embodiment tag 'UNITREE_G1_SONIC' ... is not supported by this checkpoint.
```

- Use a finetuned `UNITREE_G1_SONIC` checkpoint for inference.
- Known server command shape:

```bash
cd /workspace

export NO_ALBUMENTATIONS_UPDATE=1
export CKPT=/workspace/outputs/point_block_260611_vla_260611/checkpoint-15000

uv run python gr00t/eval/run_gr00t_server.py \
  --model-path "$CKPT" \
  --embodiment-tag UNITREE_G1_SONIC \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5550
```

- H100 inference containers need an explicit host port mapping such as `-p 5550:5550`; otherwise the workstation cannot reach the server through the host.

## Ramping And Real-Robot Deployment

Verified real-robot keyboard workflow:

1. Robot starts standing straight.
2. Press `i`: ramp standing straight to `CALIB_FULL` and hold `CALIB_FULL` in planner mode.
3. Press `p`: start inference. The implementation waits to switch into pose streaming until the first VLA action is ready.
4. Press `p`: pause inference and return to `CALIB_FULL`.
5. Press `k`: ramp from `CALIB_FULL` back to standing straight and stop control.

Relevant implementation files:

- `gear_sonic/scripts/launch_inference.py`
- `gear_sonic/scripts/run_vla_inference.py`
- `gear_sonic/utils/inference/control_transitions.py`
- `gear_sonic/utils/inference/initial_pose_ramp.py`
- `gear_sonic/utils/teleop/xr_upperbody_bridge.py`

Bug fixed during deployment:

- Symptom: pressing `i` slowly moved to `CALIB_FULL`, then jumped back to standing.
- Cause: the `CALIB_FULL` planner target was not held robustly while paused, and pose mode could start before the first VLA action was ready.
- Fix: keep refreshing a `CALIB_FULL` planner hold while paused, defer pose-mode start until the first VLA action, and use explicit standing ramp on `k`.

When GEAR-SONIC deploy is already running on robot PC2, launch the workstation inference tmux without local deploy:

```bash
python gear_sonic/scripts/launch_inference.py \
  --no-deploy \
  --policy-host <h100-host> \
  --policy-port 5550 \
  --camera-host <robot-pc2-host> \
  --state-zmq-host <robot-pc2-host> \
  --prompt "point finger to blue block" \
  --initial-pose calib_full \
  --initial-pose-ramp-s 2.0 \
  --standing-ramp-s 2.0
```

## Verification And Integration

Code branch integration:

- Code integration commit: `992879d`.
- Code commit message: `feat(gear_sonic): add point-block review and safe VLA ramping`.
- Local `main` was fast-forwarded to the code integration commit before this memory note was added. Later documentation-only commits may move `main` past `992879d`.

Verification evidence on local `main`:

```text
72 passed, 1 skipped
```

Additional verification:

- `process_dataset` task-index remapping check passed in `.venv_inference`.
- `py_compile` passed for changed scripts and helper modules.
- Real robot deployment was verified by operator after the ramping fix.

Environment caveat:

- The base Conda Python in this workstation had a pandas/numpy ABI mismatch (`numpy.dtype size changed`). For dataset checks, use a compatible project environment such as `.venv_inference` or install a consistent pandas/numpy stack.
