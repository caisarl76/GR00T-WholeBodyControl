# Isolated SONIC v1.1 mode-handoff evaluation — 2026-09-14

The patched deployment completed ten switches (five in each direction) without
falling. The baseline also completed ten switches without falling. This fixture
did **not reproduce the reported live PICO balance loss**. The directly observed
repair is continuous reference playback: the stress run recorded 21 paused
control ticks on the baseline and zero on the patch.

## Results

Each row is one run, with measurements starting five seconds after control start.

| Build / scenario | Switches | Minimum root height (m) | Maximum root tilt (degrees) | Falls |
| --- | ---: | ---: | ---: | ---: |
| Baseline / nominal | 6 | 0.781809 | 8.363454 | 0 |
| Patched / nominal | 6 | 0.780352 | 7.651195 | 0 |
| Baseline / stress | 4 | 0.765514 | 7.984254 | 0 |
| Patched / stress | 4 | 0.781042 | 8.345684 | 0 |

The stress runs each logged 1,551 control ticks after settling. Baseline playback
was paused on 21 ticks (about 420 ms total at 50 Hz); patched playback was never
paused. Planner command-to-reference latency remained about 0.1 seconds:
94–107 ms baseline, 108–118 ms patched. Keeping the outgoing reference playing
during preparation does not make planner inference instantaneous.

![Stress-run root motion and reference playback](stress_comparison.png)

POSE command-to-reference latency was 112–127 ms baseline and 26–38 ms patched.
The patched command was deliberately sent **after** 100 ms of preparation, so
these numbers do not show a 100-ms reduction in total operator-request latency.
Single trials and the slightly higher patched stress-run tilt do not establish
a general balance improvement.

## Method

- Baseline executable source: `7466582`, before the handoff repair. Patched
  executable source: `a069934`, branch `fix/sonic-mode-handoff`, based on main
  `dce9c38`. Both used the patched Python simulator and identical model files.
  Exact executable, model and fixture hashes are in [manifest.json](manifest.json).
- SONIC v1.1 decoder/encoder, V2 planner, G1 43-DOF Dex3 scene, existing local
  TensorRT 10.13.0.35 / CUDA 13 libraries, RTX 3060 GPU 1. No installation.
- Every run used a private user/network namespace containing only loopback.
  DDS domain 0 used `lo`; ZMQ ports were 15556/15557. No PC2 connection or change.
- Headless simulation targeted 200 Hz, publishing recorded pose input at 50 Hz.
  Measured physics/wall-time ratios were 0.963–0.967. An initial elastic support
  was released three seconds after start; the first five seconds were excluded
  from reported extrema. Runs stopped if root height fell below 0.45 m or tilt
  exceeded 60 degrees. Initialization and total runtime were bounded.
- Input repeated the last frame (recorded frame 820) of `pose_000816.npz`, with
  five-frame packets and increasing frame indices. SMPL pose/joints and root
  quaternion supplied the recorded standing-like body pose. Joint velocities
  were zero; the first 23 `joint_pos` entries are recording placeholders, not a
  measured neutral robot pose. This is a fixed recorded pose, not live hands.
- Nominal: 38 seconds, switching at 8/13/18/23/28/33 seconds.
- Stress: 36 seconds, switching at 12/17/26/31 seconds; planner facing was set
  to 45 degrees during seconds 6–17 and 20–31. There was no translation request.
  Each POSE producer had a 100-ms warmup gap. Baseline sent the mode command
  immediately; patched kept publishing PLANNER during warmup, then sent POSE
  data and the command. This emulates manager ordering, rather than running the
  PICO manager itself.

## Evidence and reproduction

[summary.json](summary.json) contains extrema, transition-window measurements,
reference latencies and pause counts. Each run directory retains compressed
physics measurements and events; stress directories additionally retain motion
name, playback and encoder-mode CSVs. `command_age_ms` is an unimplemented `-1`
placeholder in the physics CSV and must not be interpreted as command latency.

The original complete logs and execution scripts remain locally under
`/tmp/sonic-handoff-eval-20260914`; weights, private pose captures and large debug
streams are not packaged in Git. Reproduction requires access to the fixture
identified in the manifest, or an explicitly documented replacement.

The portable helpers here were extracted from the actual run scripts. Their
syntax, CLI and rejection of a normal host network were checked; the refactored
helpers have not themselves repeated the GPU runs. Fill `config.example.json`
with absolute local paths and save it as `/tmp/handoff-config.json`. Set
`models_dir` to **disposable copies** of `model_decoder.onnx`,
`model_encoder.onnx`, `observation_config.yaml` and `V2/planner_sonic.onnx`, since
TensorRT writes caches beside the models. Use an existing Python environment
with this repository's simulation dependencies. Run from this artifact directory:

```bash
SIM_PYTHON=/absolute/path/to/existing/venv/bin/python
unshare --user --map-root-user --net sh -c \
  'ip link set lo up && exec "$@"' sh \
  "$SIM_PYTHON" "$PWD/run_simulation.py" /tmp/handoff-config.json \
  --variant baseline --scenario nominal
```

Repeat with `--variant patched`, then both variants with `--scenario stress`.
Each output directory must be new. The runner refuses to spawn children when
interfaces other than loopback are visible. Do not run it in a robot-connected
namespace. Regenerate the archived measurements without launching control:

```bash
"$SIM_PYTHON" analyze.py > /tmp/handoff-summary.json
```

## Remaining qualification

Live manager A+X transitions with moving upper-body targets, B+Y freeze,
source-loss/cancellation and stop during preparation still need MuJoCo testing.
The manager's warmup/cancellation behavior has regression-test coverage, but
these runs did not exercise headset input, optical fingers, network loss or
arbitrary operator postures. Physical G1 transition quality remains unverified.
Do not interpret this report as proof that the original fall is resolved or as
hardware deployment qualification.
