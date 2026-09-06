"""Bounded recorded-observation ACT → native SONIC simulation replay.

Run inside the SONIC container, alongside MuJoCo and the native controller.
The publisher binds loopback only. The HTTP policy server is a separate Docker
container on a private network. This experiment does not assess task success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np
import zmq

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gear_sonic.utils.inference.reference_adapter.g1_act import (
    NOMINAL_BODY,
    compose_reference,
    compose_standing_reference,
    leased_payload,
)
from gear_sonic.utils.inference.reference_adapter.startup import settled_observation
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message, pack_pose_message

MODEL_SHA256 = "3ab2dcd032442c4ee6bf9b217d1ba32ac9059f7b86c184b67a30412165cec42b"


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # A writer may currently be appending the final line.
    return records


def run(args) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    observations_in = args.observation
    if not 1 <= len(observations_in) <= 6:
        raise ValueError("select one to six contiguous recorded ACT observations")
    sources = []
    responses = []
    for path in observations_in:
        provenance = json.loads(path.with_suffix(".json").read_text())
        if sources:
            previous = sources[-1]
            if (
                provenance["episode_index"] != previous["episode_index"]
                or provenance["dataset_revision"] != previous["dataset_revision"]
                or provenance["frame_index"] != previous["frame_index"] + 100
                or not np.isclose(provenance["timestamp"] - previous["timestamp"], 100 / 30, atol=1e-5)
            ):
                raise ValueError("recorded observations must advance 100 frames within the same episode")
        observation = path.read_bytes()
        digest = hashlib.sha256(observation).hexdigest()
        if digest != provenance["npz_sha256"]:
            raise ValueError("recorded observation checksum does not match provenance")
        sources.append({**provenance, "path": str(path)})
        request = urllib.request.Request(
            args.act_url.rstrip("/") + "/infer", data=observation,
            headers={"Content-Type": "application/octet-stream"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            inference = json.load(response)
        if inference.get("model_digest") != MODEL_SHA256:
            raise ValueError("unexpected model checkpoint served by ACT")
        chunk = np.asarray(inference["actions"], dtype=np.float32)
        if chunk.shape != (100, 28) or not np.all(np.isfinite(chunk)):
            raise ValueError("invalid ACT server output")
        responses.append(inference)
    actions = np.concatenate([np.asarray(row["actions"], dtype=np.float32) for row in responses])
    np.save(args.output_dir / "actions.npy", actions)
    (args.output_dir / "inference.json").write_text(json.dumps({
        "model_digest": MODEL_SHA256, "chunks": responses, "source_observations": sources,
    }, allow_nan=False) + "\n")

    ready_deadline = time.monotonic() + 60
    while True:
        records = read_records(args.sim_output / "observations.jsonl")
        observations = [r for r in records if "body_q" in r]
        initialized = args.controller_log.exists() and "Init Done" in args.controller_log.read_text()
        if initialized and observations and observations[-1]["command_received"]["body"]:
            break
        if time.monotonic() > ready_deadline:
            raise TimeoutError("MuJoCo and SONIC did not reach initialized state")
        time.sleep(0.1)
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    publisher.bind("tcp://127.0.0.1:5556")
    try:
        # Establish PUB/SUB before reset so SONIC can balance immediately.
        time.sleep(1.0)
        args.startup_file.touch(exist_ok=False)
        reset_path = args.sim_output / "startup_reset.json"
        startup_deadline = time.monotonic() + 20
        while not reset_path.exists():
            if time.monotonic() > startup_deadline:
                raise TimeoutError("simulation did not acknowledge keyboard startup")
            time.sleep(0.01)
        reset = json.loads(reset_path.read_text())
        if reset["keyboard_sequence"] != ["9", "backspace"] or reset["elastic_band_enabled"]:
            raise ValueError("simulation did not acknowledge unhang then reset")
        standing = compose_standing_reference(reset)
        np.savez(args.output_dir / "standing-reference.npz", **standing)
        session_id = time.monotonic_ns()
        standing_packet = pack_pose_message(leased_payload(
            standing, session_id=session_id, chunk_id=1, deadline_ns=session_id + 25_000_000_000,
        ), version=1)
        publisher.send(build_command_message(start=False, stop=False, planner=False))
        for _ in range(3):
            publisher.send(standing_packet)
            publisher.send(build_command_message(start=True, stop=False, planner=False))
            time.sleep(0.05)
        initial = None
        while initial is None:
            trace = read_records(args.native_trace)
            bootstrap = [r for r in trace if r.get("session_id") == session_id
                         and r.get("chunk_id") == 1 and r.get("mode") == "active"]
            if bootstrap:
                initial = settled_observation(
                    read_records(args.sim_output / "observations.jsonl"),
                    reset_ns=bootstrap[0]["monotonic_ns"],
                )
            if initial is not None and time.monotonic_ns() - initial["monotonic_ns"] > 250_000_000:
                initial = None
            sim_summary = args.sim_output / "summary.json"
            if sim_summary.exists() and json.loads(sim_summary.read_text()).get("error"):
                raise RuntimeError("simulation failed during standing startup")
            if time.monotonic() > startup_deadline:
                raise TimeoutError("SONIC did not settle on both feet with aligned waist after keyboard reset")
            time.sleep(0.05)
        (args.output_dir / "grounded_baseline.json").write_text(json.dumps(initial, indent=2) + "\n")
        contract = json.loads((args.sim_output / "joint_contract.json").read_text())
        # Keep the proven standing reference for legs/waist/root; ramp the arms
        # and hands from their measured grounded state to the ACT targets.
        anchor = {**initial, "body_q": list(initial["body_q"]),
                  "floating_base_pose": list(initial["floating_base_pose"])}
        anchor["body_q"][:15] = NOMINAL_BODY[:15].tolist()
        anchor["floating_base_pose"][3:7] = [1.0, 0.0, 0.0, 0.0]
        reference, composition = compose_reference(actions, anchor, contract)
        composition["lower_body_source"] = "native nominal standing reference retained after measured settling"
        composition["root_orientation_source"] = (
            "upright standing reference, native heading reanchored at ACT replacement"
        )
        np.savez(args.output_dir / "reference.npz", **reference)
        (args.output_dir / "composition.json").write_text(json.dumps(composition, indent=2) + "\n")
        started_ns = time.monotonic_ns()
        lease_seconds = math.ceil((len(reference["frame_index"]) - 46) / 50 + 3)
        if lease_seconds > 30:
            raise ValueError("reference exceeds the native 30-second lease limit")
        deadline_ns = started_ns + lease_seconds * 1_000_000_000
        # Same session, increasing chunk: frame 0 deliberately replaces the
        # standing timeline and reanchors heading through native catch-up.
        payload = leased_payload(reference, session_id=session_id, chunk_id=2, deadline_ns=deadline_ns)
        packet = pack_pose_message(payload, version=1)
        for _ in range(3):
            publisher.send(packet)
            time.sleep(0.1)
        active_deadline = time.monotonic() + 3.0
        while not any(
            r.get("session_id") == session_id and r.get("chunk_id") == 2 and r.get("mode") == "active"
            for r in read_records(args.native_trace)
        ):
            if time.monotonic() >= active_deadline:
                raise TimeoutError("SONIC did not acknowledge active adapter control after grounded startup")
            time.sleep(0.05)
        time.sleep(1.5)
        malformed = leased_payload(reference, session_id=session_id, chunk_id=3, deadline_ns=deadline_ns)
        malformed["joint_vel"] = reference["joint_vel"][:, :1]
        publisher.send(pack_pose_message(malformed, version=1))
        while time.monotonic_ns() < deadline_ns + 1_000_000_000:
            if sim_summary.exists() and json.loads(sim_summary.read_text()).get("error"):
                raise RuntimeError("simulation failed during ACT playback")
            time.sleep(0.05)
        publisher.send(
            pack_pose_message(
                leased_payload(reference, session_id=session_id, chunk_id=4, deadline_ns=deadline_ns), version=1
            )
        )
        time.sleep(0.3)
        trace = [r for r in read_records(args.native_trace) if r.get("monotonic_ns", 0) >= started_ns]
        active = [r for r in trace if r["mode"] == "active" and r["chunk_id"] == 2]
        held = [r for r in trace if r["mode"] == "expired_hold" and r["chunk_id"] == 2]
        errors = []
        if not active or not held:
            errors.append("missing active or expiry-hold native trace")
        for row in active:
            index = row["reference_frame"]
            if row["session_id"] != session_id or row["chunk_id"] != 2 or row["encoder_mode"] != 0:
                errors.append("unexpected session, chunk, or native encoder mode")
                break
            if not 0 <= index < len(reference["frame_index"]):
                errors.append("native frame outside reference")
                break
            for field, source in (
                ("body_reference_isaac", "joint_pos"),
                ("left_hand_reference", "left_hand_joints"),
                ("right_hand_reference", "right_hand_joints"),
            ):
                if not np.allclose(row[field], reference[source][index], atol=1e-6, rtol=0):
                    errors.append(f"reference mismatch: {field}")
            if not np.all(np.isfinite(row["token"])):
                errors.append("nonfinite native token")
        if held and active:
            for row in held:
                for field in ("body_reference_isaac", "left_hand_reference", "right_hand_reference"):
                    if not np.array_equal(row[field], active[-1][field]):
                        errors.append(f"expiry changed the last applied reference: {field}")
                        break
        summary = {
            "scope": "recorded-observation replay through native SONIC; not task success",
            "model_digest": MODEL_SHA256,
            "source_observation_sha256": [row["npz_sha256"] for row in sources],
            "source_action_frames": len(actions),
            "keyboard_sequence": reset["keyboard_sequence"],
            "reset_monotonic_ns": reset["monotonic_ns"],
            "baseline_monotonic_ns": initial["monotonic_ns"],
            "first_active_monotonic_ns": active[0]["monotonic_ns"] if active else None,
            "baseline_waist_rad": initial["body_q"][12:15],
            "active_ticks": len(active),
            "expired_hold_ticks": len(held),
            "frames_executed": len({r["reference_frame"] for r in active}),
            "session_id": session_id,
            "bootstrap_first_active_monotonic_ns": bootstrap[0]["monotonic_ns"],
            "bootstrap_chunk_id": 1,
            "act_chunk_id": 2,
            "deadline_ns": deadline_ns,
            "reference_pairing_passed": bool(active) and not errors,
            "malformed_and_replayed_chunks_executed": any(r["chunk_id"] not in (1, 2) for r in trace),
            "errors": sorted(set(errors)),
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    finally:
        publisher.close()
        context.term()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--act-url", default="http://jihun-g1-act-integration-20260906:8080")
    parser.add_argument("--observation", required=True, type=Path, action="append")
    parser.add_argument("--sim-output", required=True, type=Path)
    parser.add_argument("--controller-log", required=True, type=Path)
    parser.add_argument("--native-trace", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--startup-file", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2))
    return 0 if summary["reference_pairing_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
