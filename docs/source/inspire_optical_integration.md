# Inspire optical bridge integration — 2026-09-15

This integration preserves native six-motor optical commands while carrying
forward the working controller bridge's physical transport corrections. The
original dirty optical and controller worktrees were inspected and snapshotted
before edits. Work is isolated on `work/inspire-optical-integration`, based on
the verified PR11 head `6501289` (merged as `818d9d2`).

## Contracts and changes

- Optical commands stay normalized q6: little, ring, middle, index, thumb bend,
  thumb rotation; 1 is open and 0 closed. No Dex3 q7 projection or residual
  allowance was introduced.
- The physical adapter supports measured initialization, separate DDS driver
  participants, disposal without topic metadata, a 200 ms scheduling guard,
  pacing after completed writes, bounded input draining, retained fault samples,
  and truthful shutdown results. Applied sequence advances only after both writes.
- Physical defaults remain 800–1000 counts with five-count steps. Explicit
  `--initialize-hands`, `--full-position-range`, and `--no-active-slew-limit`
  options require publishing. They are not hardware qualification evidence.
- Status includes `left_feedback_age_ns` and `right_feedback_age_ns`. Consumers
  add local elapsed time to the reported physical age and require less than
  500 ms. Legacy v1 status without these suffix fields remains decodable for
  diagnostics but cannot admit tracking or recording. Update publisher and
  consumers together.
- Inspire PLANNER/frozen modes retain hand targets and remain recordable.
  Returning to POSE reseeds recovery from fresh measured hand positions.
  Dex3's native planner fist ownership remains unchanged.
- Native Inspire episodes remain 41-joint/q6. Raw motor counts, errors, physical
  feedback ages and source receive ages are recorded separately from requested
  and applied actions. Quality/provenance fields are excluded from action inputs.
  Dataset cleaning preserves planner locomotion and intentional holds; dry-run
  is available before deleting invalid frames. Incompatible schemas cannot merge.

## MuJoCo test commands

Use the integrated checkout and the already available SONIC v1.1 models below.
Stop previous local simulator/manager/deployment processes using these same
ports before starting. This test uses local ports 5556, 5557 and 5563; it does
not use PC2 or the physical DDS hand bridge.

Terminal 1 — simulator:

```bash
cd /tmp/gr00t-inspire-optical-integration
export PYTHONPATH="$PWD"
/home/jihun/work/GR00T-WholeBodyControl/.venv_sim/bin/python \
  gear_sonic/scripts/run_sim_loop.py \
  --hand-profile inspire_ftp --hand-command-source optical
```

Terminal 2 — integrated SONIC deployment:

```bash
cd /tmp/gr00t-inspire-optical-integration/gear_sonic_deploy
MODELS=/tmp/sonic-handoff-eval-20260914/models
export LD_LIBRARY_PATH="/home/jihun/TensorRT/TensorRT-10.13.0.35/lib:/usr/local/cuda/lib64:/usr/local/lib:${LD_LIBRARY_PATH:-}"
./target/release/g1_deploy_onnx_ref lo \
  "$MODELS/model_decoder.onnx" \
  /home/jihun/work/GR00T-WholeBodyControl/gear_sonic_deploy/reference/example/ \
  --obs-config "$MODELS/observation_config.yaml" \
  --encoder-file "$MODELS/model_encoder.onnx" \
  --planner-file "$MODELS/V2/planner_sonic.onnx" \
  --input-type zmq_manager --output-type all \
  --zmq-host 127.0.0.1 --zmq-port 5556 --zmq-out-port 5557 \
  --disable-crc-check --disable-dex3-hands --live-pose-playback
```

Terminal 3 — optical manager with a new capture directory on every run:

```bash
cd /tmp/gr00t-inspire-optical-integration
export PYTHONPATH="$PWD"
mkdir -p outputs
CAPTURE=$(mktemp -d "$PWD/outputs/inspire-optical-XXXXXXXX")
set -o pipefail
/home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python -u \
  gear_sonic/scripts/pico_manager_thread_server.py --manager \
  --hand-input optical --hand-profile inspire_ftp --hand-max-rate 1.0 \
  --zmq_feedback_host 127.0.0.1 --inspire-status-host 127.0.0.1 \
  --hand-log-dir "$CAPTURE/hands" --record_dir "$CAPTURE/body" \
  2>&1 | tee "$CAPTURE/manager.log"
```

Enable body/hand tracking and Send in XRoboToolkit. Hold A+B+X+Y to start
SONIC, release all buttons and grips, then hold A+X for at least 0.3 seconds
to enter POSE. Put controllers down and test each finger and both thumb axes.
Repeat POSE→PLANNER→POSE with partially closed hands; verify balance and held
fingers during PLANNER. Occlude one hand, verify its hold while the other moves,
then verify measured-state recovery. Inspire rate is normalized travel/second.

Hand/body capture above is diagnostic recording. An exported dataset additionally
requires the data exporter and its START/SAVE acknowledgement workflow described
in [the optical guide](pico_optical_hand_tracking.md).

## Evidence and remaining qualification

Hardware-free checks exercise native q6 transport, manager mode/recording
transitions, physical feedback age, DDS writer/fault fixtures, and each of all
12 actual MuJoCo hand actuators independently. The latter disables gravity and
contact to isolate motor order/polarity; it is not a whole-body balance test.
The integrated deployment builds and both C++ handoff/hand-decoder tests pass.
Final focused Python runs passed 384 core tests, 50 bridge tests and 71
exporter/cleaner tests (505 total); Ruff and scoped diff checks also pass.

A read-only cleaner dry run on the existing controller capture
`outputs/g1-inspire-20260915-105114` retained 3209/3215 frames, all three episodes
and all 1751 PLANNER frames. Six invalid POSE frames were identified; the source
recording was not modified.

The user subsequently confirmed live bilateral finger tracking and A+X
POSE/PLANNER switching in MuJoCo. Intermittent left articulation loss recovered
without an established upstream cause. Subsequent physical hand operation and dataset capture were confirmed by the
user; the latest inspection is summarized below. The [real test procedure](inspire_optical_real_test.md)
contains the optical Python bridge package and manual test commands. The user
selected the original PC2 SONIC at `087f9ac`; no SONIC rebuild or deployment of
the local adapted handoff is part of the selected real workflow. The original
installation already lacks the custom one-second motor-output ramp. The separate Python bridge was installed on Inspire PC2 `.222` on September 21;
the original SONIC installation was retained. Dex3 `.223` is a separate setup.

## September 21 recording inspection and pending tests

The latest inspected LeRobot v2.1 episode contains 1,596 frames at 50 Hz
(31.92 seconds), finite 64D SONIC v1.1 token fields, 41D state/action vectors,
native 6D requested/applied/measured hand fields, and a matching 1,596-frame
video. Applied hand ranges span 0–1000 counts on the left and 1–1000 on the
right across the episode, with the active slew cap disabled. These aggregate
ranges do not mean every motor reached both endpoints.

The episode includes 1,181 PLANNER and 415 POSE frames. PLANNER turning is
present, but translation requests stay zero. Cleaner dry-run preserved all
3,121 frames across both recorded episodes, including intentional holds and
PLANNER intervals. Optical loss/hold flags remain in the data; this is not a
claim of uninterrupted optical tracking.

**Needed to be tested:** the updated Left Grip+A controller recording sequence.
The manager no longer consumes its chord window while grip alone is held.
Software regression tests cover this sequence and Left Grip+B abort; physical
controller confirmation remains pending. See the integration plan hardware TODO.
