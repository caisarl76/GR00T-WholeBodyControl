"""Verify/extract the committed evidence and reproduce the joint/palm audit.

Run from the repository root using Python 3.11 with requirements.txt installed.
Output stays under --output-dir; the committed baseline is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

REPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPORT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))


def extract_evidence(output_dir):
    manifest = json.loads((REPORT_DIR / "source_evidence_manifest.json").read_text())
    archive = REPORT_DIR / "source_evidence.zip"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("source evidence archive SHA-256 mismatch")
    run_dir = output_dir / "corrected-act-run6"
    with zipfile.ZipFile(archive) as zipped:
        expected_members = {"corrected-act-run6/" + name for name in manifest["files"]}
        if len(zipped.namelist()) != len(expected_members) or set(zipped.namelist()) != expected_members:
            raise ValueError("unexpected or duplicate evidence archive members")
        for name, digest in manifest["files"].items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("evidence member must be a relative path within the run")
            content = zipped.read("corrected-act-run6/" + name)
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"source evidence SHA-256 mismatch: {name}")
            target = run_dir / name
            if target.exists() and target.read_bytes() != content:
                raise ValueError(f"existing evidence differs; choose a fresh --output-dir: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    return run_dir


def compare_baseline(joints, palms):
    """Check numeric metrics, not machine-specific paths, fonts, or PNG bytes."""
    import numpy as np

    baseline_joints = json.loads((REPORT_DIR / "summary.json").read_text())
    baseline_palms = json.loads((REPORT_DIR / "hand_pose_summary.json").read_text())

    def check_tree(current, expected, name):
        if isinstance(expected, dict):
            if current.keys() != expected.keys():
                raise ValueError(f"metric keys changed: {name}")
            for key, value in expected.items():
                check_tree(current[key], value, f"{name}.{key}")
        else:
            np.testing.assert_allclose(current, expected, rtol=1e-8, atol=1e-8, err_msg=name)

    check_tree(joints["metrics"], baseline_joints["metrics"], "joints")
    check_tree(palms["metrics"], baseline_palms["metrics"], "palms")
    if joints["scope"] != baseline_joints["scope"]:
        raise ValueError("comparison sample scope differs from committed baseline")
    if joints["transport_reference"] != baseline_joints["transport_reference"]:
        raise ValueError("reference transport differs from committed baseline")


def main():
    from gear_sonic.scripts.compare_g1_act_hand_poses import analyze_poses, write_outputs
    from gear_sonic.scripts.compare_g1_act_sonic_joints import compare

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / ".tmp/act-sonic-comparison")
    parser.add_argument("--compare-baseline", action="store_true", help="fail if numeric metrics differ")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir == REPORT_DIR or REPORT_DIR in output_dir.parents:
        parser.error("choose an output directory outside the committed report")
    run_dir = extract_evidence(output_dir)
    result_dir = output_dir / "results"
    comparison = compare(run_dir, result_dir)
    poses, errors, summary = analyze_poses(comparison)
    summary["recorded_render_mujoco_version"] = json.loads(
        (run_dir / "result-video/render-manifest.json").read_text()
    )["mujoco_version"]
    write_outputs(result_dir, comparison, poses, errors, summary)
    if args.compare_baseline:
        compare_baseline(comparison["summary"], summary)
    print(
        json.dumps(
            {
                "output_dir": str(result_dir),
                "samples": summary["samples"],
                "baseline_checked": args.compare_baseline,
            },
            indent=2,
        )
    )
    return 0 if comparison["summary"]["transport_reference"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
