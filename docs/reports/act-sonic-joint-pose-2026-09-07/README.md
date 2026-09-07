# Continue the ACT / SONIC joint and palm comparison

This folder preserves the confirmed 2026-09-07 comparison and its numerical
inputs so it can be reproduced from an Ubuntu checkout without H100 access.
Start with [REPORT.md](REPORT.md) for all 28 joint errors, palm pose errors,
plots, methods, and limitations.

The adapter transports its composed references exactly. Measured palms differ
from raw ACT targets by **1.98 cm / 6.50° RMS left** and
**3.84 cm / 8.47° RMS right**. Exact reproduction has not been established.
This commit adds analysis and evidence; it does not change controller behavior.

## Included evidence

- `source_evidence.zip`: the original ACT actions, composed references,
  composition settings, grounded baseline, joint contract, complete native
  reference trace, complete simulator observations, and render provenance.
- `source_evidence_manifest.json`: SHA-256 of the archive and every member.
- CSV files: every compared joint angle, world/torso XYZ, wxyz quaternion,
  and aggregate metric. PNG files show all joints and both palm paths/errors.
- `diagnosis.json`: reference recomposition, clipping, timing sensitivity,
  and approximate hand feedback-clamp checks from the original audit.
- `verification.json`: original audit test results and analysis source hashes.

The archive is numerical evidence only. Checkpoints, dataset camera images,
videos, and the H100 containers are not required for this offline comparison.
The G1 MJCF and meshes already live in the repository; the meshes use Git LFS.

## Ubuntu setup and reproduction

Use branch `feat/vla-sonic-g1-dex3-integration`. In an existing checkout:

```bash
git fetch origin
git switch feat/vla-sonic-g1-dex3-integration
git pull --ff-only
git lfs pull --include='gear_sonic/data/robot_model/model_data/g1/meshes/*'
```

Run from the repository root. Docker keeps analysis dependencies off the host;
it needs neither NVIDIA drivers nor a display for this offline calculation.
The commands assume Docker and Git LFS are already available.

```bash
docker build -t act-sonic-comparison:2026-09-07 \
  docs/reports/act-sonic-joint-pose-2026-09-07

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" act-sonic-comparison:2026-09-07 \
  python docs/reports/act-sonic-joint-pose-2026-09-07/reproduce.py \
  --compare-baseline

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" act-sonic-comparison:2026-09-07 \
  python -m pytest -q \
  gear_sonic/tests/test_g1_act_joint_comparison.py \
  gear_sonic/tests/test_g1_act_hand_poses.py \
  gear_sonic/tests/test_g1_act_reference.py
```

If an existing Python 3.11 analysis environment already has `requirements.txt`,
the same commands can be run with its Python executable instead of Docker.
The pinned environment and fresh evidence extraction were verified on macOS;
the Ubuntu Docker build has not been executed as part of this report commit.

`reproduce.py` verifies all source hashes, extracts to
`.tmp/act-sonic-comparison/corrected-act-run6`, and writes fresh plots/CSVs/JSON
to `.tmp/act-sonic-comparison/results`. `--compare-baseline` checks numerical
metrics against this folder. Floating-point tolerance is `rtol=atol=1e-8`;
image bytes and absolute paths can differ across platforms. Repeating a run
reuses identical evidence; changed evidence requires a fresh `--output-dir`.

Expected baseline: **977 ACT-phase observations**, **1,399 active native
records**, **1,249 distinct reference frames**, reference transport maximum
error **0 rad**, and **18 passing tests**.

To inspect a new recorded run, use the reusable tools directly:

```bash
python -m gear_sonic.scripts.compare_g1_act_sonic_joints \
  --run-dir /path/to/new-run --output-dir .tmp/new-run-comparison
python -m gear_sonic.scripts.compare_g1_act_hand_poses \
  --run-dir /path/to/new-run --output-dir .tmp/new-run-comparison
```

These tools compare absolute joint angles in radians after named joint-order
conversion. The raw normalized decoder tensor was not recorded in this run;
the issued body targets come from received DDS commands. The palm frame is
the XML palm mesh origin, not a grasp TCP or fingertip. All stages use the same
measured base and waist to isolate arm motion.

## Where to continue

1. Add same-control-tick telemetry for the raw decoder output and scaled body
   targets after `CreatePolicyCommand()`. For the hands, record the DDS writer's
   reference, measured feedback, clamp result, and published target. Current
   timestamp association is approximate (median 9.76 ms, maximum 20.07 ms).
2. Investigate continuity between the six 100-action ACT chunks. The largest
   palm errors occur at 13.34 s, just after action 400 starts a new chunk with
   an arm jump up to 37.57°. Preserve the original baseline when evaluating
   blending or temporal aggregation; removing rate limits is not an established fix.
3. Measure reference-to-robot tracking separately from raw-ACT-to-reference
   changes. Large PD target offsets alone do not prove a mapping bug.
4. Evaluate fingers separately. Good palm tracking does not establish grasp
   fidelity; Dex3 fingers bypass the learned body decoder and receive their
   own limited DDS commands.
5. Define application tolerances for each joint and for palm position and
   orientation before labeling a future run as sufficiently similar.

For new VLA/SONIC/MuJoCo execution, use the existing
[H100 Docker runbook](../../source/tutorials/vla_adapter_h100.md).
Keep VLA, SONIC, simulation, and package installation inside containers on H100.
Preserve the confirmed startup order: keyboard **9 → Backspace**, standing
settling, then ACT execution.
