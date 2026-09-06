import numpy as np

from gear_sonic.utils.inference.reference_adapter.startup import settled_observation


def records():
    return [
        {"monotonic_ns": 1_000_000_000 + i * 20_000_000,
         "body_q": np.zeros(29).tolist(), "body_dq": np.zeros(29).tolist(),
         "floating_base_pose": [0, 0, 0.77, 1, 0, 0, 0],
         "floating_base_vel": [0] * 6, "foot_contacts": {"left": True, "right": True}}
        for i in range(31)
    ]


def test_post_reset_grounded_settled_baseline():
    rows = records()
    assert settled_observation(rows, reset_ns=999_999_999) is rows[-1]
    assert settled_observation(rows, reset_ns=rows[-1]["monotonic_ns"]) is None


def test_hanging_waist_and_airborne_records_do_not_start_act():
    rows = records()
    rows[-1]["body_q"][13:15] = [0.4643, 0.5128]
    assert settled_observation(rows, reset_ns=0) is None
    rows = records()
    rows[-1]["foot_contacts"]["right"] = False
    assert settled_observation(rows, reset_ns=0) is None


def test_reset_transient_and_record_gap_do_not_count_as_settled():
    rows = records()
    rows[-4]["body_dq"][13] = 1.0
    assert settled_observation(rows, reset_ns=0) is None
    rows = records()
    assert settled_observation([rows[0], rows[-1]], reset_ns=0) is None
    rows[-1]["floating_base_pose"][2] = float("nan")
    assert settled_observation(rows, reset_ns=0) is None
