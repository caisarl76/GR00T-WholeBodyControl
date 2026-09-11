"""Build/run the production C++ callback and getters, with SDK functions stubbed.

Run: PYTHONPATH=. pytest gear_sonic/tests/test_pico_hand_snapshot.py -q
Requires a C++17 compiler, Python development headers and pybind11 headers (a
PyTorch installation's bundled headers also work). No XR service is started.
Set PYBIND11_INCLUDE_DIR if the headers are outside the usual include paths.
"""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import threading
import time

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
BINDING = ROOT / "external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"


@pytest.fixture(scope="module")
def compiled_binding(tmp_path_factory):
    candidates = [Path(os.environ.get("PYBIND11_INCLUDE_DIR", "/nonexistent"))]
    for entry in sys.path:
        candidates.extend([Path(entry) / "pybind11/include", Path(entry) / "torch/include"])
    candidates.append(Path("/usr/include"))
    include = next((p for p in candidates if (p / "pybind11/pybind11.h").exists()), None)
    assert include is not None, "Install pybind11 headers or set PYBIND11_INCLUDE_DIR"
    output = tmp_path_factory.mktemp("pico_binding") / (
        "_pico_snapshot_test" + sysconfig.get_config_var("EXT_SUFFIX")
    )
    subprocess.run(
        [
            os.environ.get("CXX", "c++"),
            "-std=c++17",
            "-O1",
            "-shared",
            "-fPIC",
            "-pthread",
            f"-I{include}",
            f"-I{sysconfig.get_paths()['include']}",
            f"-I{BINDING / 'include'}",
            str(BINDING / "bindings/snapshot_test_harness.cpp"),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    spec = importlib.util.spec_from_file_location("_pico_snapshot_test", output)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def binding(compiled_binding):
    compiled_binding.reset()
    return compiled_binding


def hand(value=1):
    return {
        "scale": float(value),
        "isActive": value % 2,
        "timeStampNs": value,
        "HandJointLocations": [
            {"p": ",".join([str(value)] * 7), "s": value, "r": float(value)} for _ in range(26)
        ],
    }


def controller(value=1):
    return {
        "pose": ",".join([str(value)] * 7),
        "trigger": float(value),
        "grip": float(value),
        "axisX": float(value),
        "axisY": float(value),
        "menuButton": bool(value % 2),
        "primaryButton": bool(value % 2),
        "secondaryButton": bool(value % 2),
        "axisClick": bool(value % 2),
    }


def inject(binding, **value):
    binding.inject(json.dumps(value))


def assert_snapshot_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


def test_hand_packet_diagnostics_preserve_raw_hand_field(binding):
    assert binding.get_hand_packet_diagnostics() == {"packets": 0, "hand_json": ""}

    inject(binding, Controller={"left": controller(1)})
    assert binding.get_hand_packet_diagnostics() == {"packets": 1, "hand_json": ""}

    inject(binding, Hand={})
    assert binding.get_hand_packet_diagnostics() == {"packets": 2, "hand_json": "{}"}

    malformed = {"leftHand": {"HandJointLocations": []}}
    inject(binding, Hand=malformed)
    assert json.loads(binding.get_hand_packet_diagnostics()["hand_json"]) == malformed

    raw = {"leftHand": hand(4), "metadata": {"source": "test"}}
    inject(binding, Hand=raw)
    raw["metadata"]["source"] = "mutated"
    diagnostics = binding.get_hand_packet_diagnostics()
    assert diagnostics["packets"] == 4
    assert json.loads(diagnostics["hand_json"])["metadata"]["source"] == "test"


def test_snapshot_shapes_types_and_owned_copy(binding):
    initial = binding.get_left_hand_snapshot()
    assert initial["binding_generation"] == 0
    assert binding.get_left_controller_snapshot()["receipt_age_ns"] == 2**63 - 1
    sample = hand(7)
    sample["HandJointLocations"][0]["s"] = 2**64 - 1
    inject(binding, Hand={"leftHand": sample, "rightHand": hand(8)})
    snapshot = binding.get_left_hand_snapshot()
    assert snapshot.keys() == {
        "pose",
        "location_flags",
        "radius",
        "scale",
        "is_active",
        "source_timestamp_ns",
        "timestamp_source",
        "binding_generation",
    }
    for key, shape, dtype in (
        ("pose", (26, 7), np.float64),
        ("location_flags", (26,), np.uint64),
        ("radius", (26,), np.float64),
    ):
        assert snapshot[key].shape == shape
        assert snapshot[key].dtype == dtype
    assert snapshot["location_flags"][0] == 2**64 - 1
    assert snapshot["source_timestamp_ns"] == 7
    assert snapshot["timestamp_source"] == 0
    assert snapshot["binding_generation"] == 1
    assert binding.get_right_hand_snapshot()["source_timestamp_ns"] == 8
    snapshot["pose"][:] = 999
    assert np.all(binding.get_left_hand_snapshot()["pose"] == 7)
    assert initial["binding_generation"] == 0


@pytest.mark.parametrize(
    "corruption",
    [
        "junk",
        "trailing_comma",
        "short_pose",
        "long_pose",
        "nan_pose",
        "inf_pose",
        "negative_flags",
        "float_flags",
        "bool_flags",
        "overflow_flags",
        "missing_flag",
        "missing_radius",
        "bad_radius",
        "short_joints",
        "null_side",
        "float_timestamp",
        "overflow_timestamp",
        "bad_active",
        "bad_scale",
    ],
)
def test_malformed_hand_preserves_entire_side_and_does_not_block_others(binding, corruption):
    inject(binding, Hand={"leftHand": hand(1)})
    before = binding.get_left_hand_snapshot()
    sample = hand(5)
    joint = sample["HandJointLocations"][25]
    poses = {
        "junk": "5junk,5,5,5,5,5,5",
        "trailing_comma": "5,5,5,5,5,5,5,",
        "short_pose": "5,5",
        "long_pose": "5,5,5,5,5,5,5,5",
        "nan_pose": "nan,5,5,5,5,5,5",
        "inf_pose": "inf,5,5,5,5,5,5",
    }
    if corruption in poses:
        joint["p"] = poses[corruption]
    elif corruption.endswith("flags"):
        joint["s"] = {
            "negative_flags": -1,
            "float_flags": 1.5,
            "bool_flags": True,
            "overflow_flags": 2**64,
        }[corruption]
    elif corruption == "missing_flag":
        del joint["s"]
    elif corruption == "missing_radius":
        del joint["r"]
    elif corruption == "bad_radius":
        joint["r"] = "bad"
    elif corruption == "short_joints":
        sample["HandJointLocations"].pop()
    elif corruption == "null_side":
        sample = None
    elif corruption == "float_timestamp":
        sample["timeStampNs"] = 2.5
    elif corruption == "overflow_timestamp":
        sample["timeStampNs"] = 2**63
    elif corruption == "bad_active":
        sample["isActive"] = 0.5
    elif corruption == "bad_scale":
        sample["scale"] = "bad"
    inject(
        binding,
        Hand={"leftHand": sample, "rightHand": hand(9)},
        Controller={"right": controller(9)},
        Body={"timeStampNs": 42, "joints": []},
    )
    assert_snapshot_equal(binding.get_left_hand_snapshot(), before)
    assert np.all(np.asarray(binding.get_left_hand_tracking_state()) == 1)
    assert binding.get_left_hand_is_active() == 1
    assert binding.get_right_hand_snapshot()["source_timestamp_ns"] == 9
    assert binding.get_right_controller_snapshot()["binding_generation"] == 1
    assert binding.get_body_timestamp_ns() == 42


@pytest.mark.parametrize(
    "flags, accepted",
    [(10.0, True), (9007199254740991.0, True), (1.5, False), (-1.0, False),
     (9007199254740992.0, False), (float("inf"), False)],
)
def test_original_apk_float_flags_accept_only_safe_integral_values(binding, flags, accepted):
    inject(binding, Hand={"leftHand": hand(1)})
    before = binding.get_left_hand_snapshot()
    sample = hand(2)
    for joint in sample["HandJointLocations"]:
        joint["s"] = flags
    inject(binding, Hand={"leftHand": sample})
    snapshot = binding.get_left_hand_snapshot()
    if accepted:
        assert snapshot["location_flags"][0] == int(flags)
        assert snapshot["binding_generation"] == before["binding_generation"] + 1
    else:
        assert_snapshot_equal(snapshot, before)


@pytest.mark.parametrize("missing", ["flags", "both", "radius"])
def test_legacy_schema_updates_legacy_pose_without_optical_snapshot(binding, missing):
    sample = hand(3)
    if missing in ("timestamp", "both"):
        del sample["timeStampNs"]
    if missing in ("flags", "both"):
        for joint in sample["HandJointLocations"]:
            del joint["s"]
    if missing == "radius":
        del sample["timeStampNs"]
        for joint in sample["HandJointLocations"]:
            del joint["r"]
    inject(binding, Hand={"leftHand": sample}, Controller={"left": controller(3)})
    assert binding.get_left_hand_snapshot()["binding_generation"] == 0
    assert np.all(np.asarray(binding.get_left_hand_tracking_state()) == 3)
    assert binding.get_left_hand_is_active() == 1
    assert binding.get_left_controller_snapshot()["binding_generation"] == 1


@pytest.mark.parametrize("changed", ["pose", "flags", "radius", "scale", "active"])
def test_original_apk_content_clock_only_advances_when_that_side_changes(binding, changed):
    sample = hand(3)
    del sample["timeStampNs"]
    before_ns = time.monotonic_ns()
    inject(binding, Hand={"leftHand": sample, "rightHand": sample})
    original = binding.get_left_hand_snapshot()
    assert original["timestamp_source"] == 1
    assert before_ns <= original["source_timestamp_ns"] <= time.monotonic_ns()
    for value in range(10, 15):
        inject(binding, Hand={"leftHand": sample, "rightHand": hand(value)}, timeStampNs=value)
    duplicate = binding.get_left_hand_snapshot()
    assert duplicate["source_timestamp_ns"] == original["source_timestamp_ns"]
    assert duplicate["binding_generation"] == original["binding_generation"] + 5
    if changed == "pose":
        sample["HandJointLocations"][0]["p"] = "4,3,3,3,3,3,3"
    elif changed == "flags":
        sample["HandJointLocations"][0]["s"] += 1
    elif changed == "radius":
        sample["HandJointLocations"][0]["r"] += 0.001
    elif changed == "scale":
        sample["scale"] += 0.1
    else:
        sample["isActive"] = 0
    inject(binding, Hand={"leftHand": sample})
    assert binding.get_left_hand_snapshot()["source_timestamp_ns"] > original["source_timestamp_ns"]


@pytest.mark.parametrize("timestamp", [None, True, "12", 1.5, 0, -1, 2**63])
def test_original_apk_never_falls_back_from_invalid_explicit_timestamp(binding, timestamp):
    sample = hand(3)
    del sample["timeStampNs"]
    inject(binding, Hand={"leftHand": sample})
    before = binding.get_left_hand_snapshot()
    sample["scale"] = 4.0
    sample["timeStampNs"] = timestamp
    inject(binding, Hand={"leftHand": sample})
    assert_snapshot_equal(binding.get_left_hand_snapshot(), before)


@pytest.mark.parametrize("metadata", ["missing_flag", "missing_radius", "bad_flags", "bad_radius"])
def test_original_apk_incomplete_metadata_cannot_refresh_content_clock(binding, metadata):
    sample = hand(3)
    del sample["timeStampNs"]
    inject(binding, Hand={"leftHand": sample})
    before = binding.get_left_hand_snapshot()
    sample["scale"] = 4.0
    joint = sample["HandJointLocations"][-1]
    if metadata == "missing_flag":
        del joint["s"]
    elif metadata == "missing_radius":
        del joint["r"]
    elif metadata == "bad_flags":
        joint["s"] = -1
    else:
        joint["r"] = "bad"
    inject(binding, Hand={"leftHand": sample})
    assert_snapshot_equal(binding.get_left_hand_snapshot(), before)


def test_device_timestamp_cannot_downgrade_to_host_content_clock(binding):
    inject(binding, Hand={"leftHand": hand(3)})
    before = binding.get_left_hand_snapshot()
    sample = hand(5)
    del sample["timeStampNs"]
    inject(binding, Hand={"leftHand": sample, "rightHand": sample})
    assert_snapshot_equal(binding.get_left_hand_snapshot(), before)
    assert np.all(np.asarray(binding.get_left_hand_tracking_state()) == 5)
    assert binding.get_right_hand_snapshot()["timestamp_source"] == 1


def test_original_apk_native_snapshots_admit_and_hold_cached_hand_independently(binding):
    from gear_sonic.tests.test_pico_hand_tracking import Retargeter, snapshot
    from gear_sonic.utils.teleop.pico_hand_tracking import HandReason, HandTracker, TrackingState

    source = snapshot()
    sample = {
        "scale": source["scale"],
        "isActive": source["is_active"],
        "HandJointLocations": [
            {"p": ",".join(map(str, pose)), "s": int(flags), "r": float(radius)}
            for pose, flags, radius in zip(source["pose"], source["location_flags"], source["radius"])
        ],
    }
    trackers = [HandTracker(Retargeter()), HandTracker(Retargeter())]

    def step():
        now = time.monotonic_ns()
        return [
            tracker.step(getattr(binding, f"get_{side}_hand_snapshot")(), np.zeros(3), np.zeros(7), now)
            for tracker, side in zip(trackers, ("left", "right"))
        ]

    for index in range(5):
        sample["scale"] = 1 + index * 0.001
        inject(binding, Hand={"leftHand": sample, "rightHand": sample})
        outputs = step()
        expected = TrackingState.WAITING if index < 4 else TrackingState.RECOVERING
        assert all(output.state == expected for output in outputs)
    for _ in range(4):
        outputs = step()
    assert all(output.state == TrackingState.TRACKING for output in outputs)
    for tracker in trackers:
        tracker.retargeter.target[:] = 0.5
    previous_tick = trackers[0].previous_tick_ns
    time.sleep(0.002)
    sample["scale"] = 1.01
    inject(binding, Hand={"leftHand": sample, "rightHand": sample})
    outputs = step()
    for tracker, output in zip(trackers, outputs):
        assert np.all(output.command > 0)
        assert np.all(output.command < 0.5)
        dt = min((tracker.previous_tick_ns - previous_tick) * 1e-9, 0.04)
        assert np.max(output.command) <= tracker.max_rate * dt + 1e-8
    held = outputs[0].command.copy()
    time.sleep(0.105)
    changed_right = {**sample, "scale": 1.1}
    inject(binding, Hand={"leftHand": sample, "rightHand": changed_right}, timeStampNs=999)
    left, right = step()
    assert left.reason == HandReason.SOURCE_STALE
    assert left.state == TrackingState.HOLDING
    np.testing.assert_array_equal(left.command, held)
    assert right.reason == HandReason.OK
    assert right.state == TrackingState.TRACKING


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        "bad",
        {"pose": "1,2"},
        {**controller(5), "pose": "bad"},
        {**controller(5), "grip": "bad"},
        {**controller(5), "axisClick": 1},
    ],
)
def test_bad_controller_cannot_partially_commit_or_block_other_data(binding, bad):
    inject(binding, Controller={"left": controller(1)})
    before = binding.get_left_controller_snapshot()
    inject(
        binding,
        Controller={"left": bad, "right": controller(7)},
        Hand={"leftHand": hand(7)},
        Body={"timeStampNs": 77, "joints": []},
    )
    after = binding.get_left_controller_snapshot()
    assert after.pop("receipt_age_ns") >= before.pop("receipt_age_ns")
    assert_snapshot_equal(after, before)
    assert np.all(np.asarray(binding.get_left_controller_pose()) == 1)
    assert binding.get_right_controller_snapshot()["grip"] == 7
    assert binding.get_left_hand_snapshot()["source_timestamp_ns"] == 7
    assert binding.get_body_timestamp_ns() == 77


def test_absent_and_invalid_containers_do_not_refresh_evidence(binding):
    inject(binding, Hand={"leftHand": hand(1)}, Controller={"left": controller(1)})
    before = binding.get_left_controller_snapshot()
    time.sleep(0.002)
    for malformed in (None, [], "bad", 4, {}):
        inject(
            binding, Controller=malformed, Hand=malformed, timeStampNs=500, Body={"timeStampNs": 500, "joints": []}
        )
    after = binding.get_left_controller_snapshot()
    assert after["binding_generation"] == before["binding_generation"]
    assert after["receipt_age_ns"] > before["receipt_age_ns"] + 1_000_000
    assert binding.get_left_hand_snapshot()["binding_generation"] == 1
    assert binding.get_left_hand_snapshot()["source_timestamp_ns"] == 1
    inject(binding, Controller={"left": controller(2)})
    assert binding.get_left_controller_snapshot()["binding_generation"] == 2


def test_concurrent_callback_snapshots_never_tear_and_generations_do_not_get_lost(binding):
    inject(binding, Hand={"leftHand": hand(1)}, Controller={"left": controller(1)})
    start = threading.Barrier(3)

    def write(offset):
        start.wait()
        for value in range(offset, offset + 100):
            inject(binding, Hand={"leftHand": hand(value)}, Controller={"left": controller(value)})

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(write, offset) for offset in (10, 1000)]
        start.wait()
        reads = 0
        while not all(future.done() for future in futures):
            snapshot = binding.get_left_hand_snapshot()
            value = snapshot["source_timestamp_ns"]
            assert np.all(snapshot["pose"] == value)
            assert np.all(snapshot["location_flags"] == value)
            assert np.all(snapshot["radius"] == value)
            assert snapshot["scale"] == value
            assert snapshot["is_active"] == value % 2
            snapshot = binding.get_left_controller_snapshot()
            value = snapshot["grip"]
            assert snapshot["trigger"] == value
            assert np.all(np.asarray(snapshot["pose"]) == value)
            assert np.all(np.asarray(snapshot["axis"]) == value)
            assert snapshot["primary_button"] == bool(value % 2)
            reads += 1
        for future in futures:
            future.result()
    assert reads > 0
    assert binding.get_left_hand_snapshot()["binding_generation"] == 201
    assert binding.get_left_controller_snapshot()["binding_generation"] == 201


def test_xr_client_wrapper_preserves_snapshot_arrays(binding, monkeypatch):
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", binding)
    spec = importlib.util.spec_from_file_location(
        "snapshot_xr_client", ROOT / "decoupled_wbc/control/teleop/device/pico/xr_client.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client = module.XrClient.__new__(module.XrClient)
    inject(binding, Hand={"leftHand": hand(2), "rightHand": hand(3)})
    for side in ("left", "RIGHT"):
        snapshot = client.get_hand_snapshot(side)
        assert snapshot["pose"].shape == (26, 7)
        assert snapshot["pose"].dtype == np.float64
        assert snapshot["location_flags"].shape == (26,)
        assert snapshot["location_flags"].dtype == np.uint64
        assert snapshot["radius"].shape == (26,)
        assert snapshot["radius"].dtype == np.float64
    with pytest.raises(ValueError):
        client.get_hand_snapshot("invalid")
    inject(binding, Controller={"left": controller(2), "right": controller(3)})
    assert client.get_controller_snapshot("LEFT")["grip"] == 2
    assert client.get_controller_snapshot("right")["grip"] == 3
    with pytest.raises(ValueError):
        client.get_controller_snapshot("invalid")
