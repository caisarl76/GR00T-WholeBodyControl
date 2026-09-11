"""Independent optical clocks, validation and bounded recovery without hardware."""

import numpy as np
import pytest

from gear_sonic.utils.teleop.pico_hand_tracking import (
    HandReason,
    HandTracker,
    TrackingState,
    validate_hand,
)


def snapshot(stamp=1):
    pose = np.zeros((26, 7), np.float64)
    pose[0, :3] = 100  # Palm must not enter the canonical geometry.
    for digit, (start, stop) in enumerate(((2, 6), (6, 11), (11, 16), (16, 21), (21, 26))):
        for index in range(start, stop):
            pose[index, :3] = (0.02 * (index - start + 1), 0.02 * (digit - 2), 0)
    flags = np.full(26, 0xA, np.uint64)
    flags[0] = 0
    return {
        "pose": pose,
        "location_flags": flags,
        "radius": np.full(26, 0.005, np.float64),
        "scale": 1.0,
        "is_active": 1,
        "source_timestamp_ns": stamp,
        "binding_generation": stamp,
    }


class Retargeter:
    def __init__(self, size=7, target=0.0):
        self.lower = np.full(size, -1.0 if size == 7 else 0.0)
        self.upper = np.ones(size)
        self.target = np.full(size, target)
        self.seen = []

    def retarget(self, points):
        self.seen.append(points.copy())
        return self.target.copy()


def step(tracker, stamp, now, measured=None, **kwargs):
    return tracker.step(
        snapshot(stamp), np.zeros(3), np.zeros(tracker.lower.size) if measured is None else measured, now, **kwargs
    )


def test_canonical_order_uses_raw_wrist_and_positions_only():
    sample = snapshot()
    sample["pose"][:, 3:] = 0  # Quaternion normalization/orientation bits are not required.
    points, reason = validate_hand(sample, np.zeros(3))
    assert reason == HandReason.OK
    np.testing.assert_array_equal(points, sample["pose"][1:, :3])
    for tip in (4, 9, 14, 19, 24):
        np.testing.assert_array_equal(points[tip], sample["pose"][tip + 1, :3])
    points[:] = 0
    assert sample["pose"][5, 0] > 0


def test_clock_provenance_change_requires_readmission():
    tracker = HandTracker(Retargeter())
    for i in range(10):
        sample = snapshot(1000 + i)
        sample["timestamp_source"] = 1
        out = tracker.step(sample, np.zeros(3), np.zeros(7), i * 20_000_000)
    assert out.state == TrackingState.TRACKING
    sample = snapshot(100_000)  # Clock can jump forward when switching to a patched APK.
    sample["timestamp_source"] = 0
    out = tracker.step(sample, np.zeros(3), np.zeros(7), 200_000_000)
    assert out.state == TrackingState.HOLDING
    assert out.reason == HandReason.SOURCE_RESTART
    assert out.source_epoch == 1


@pytest.mark.parametrize("source", [True, -1, 2, "host", 0.5])
def test_invalid_clock_provenance_rejected(source):
    sample = snapshot()
    sample["timestamp_source"] = source
    out = HandTracker(Retargeter()).step(sample, np.zeros(3), np.zeros(7), 0)
    assert out.reason == HandReason.MALFORMED


def test_inactive_hand_with_zero_radii_reports_inactive():
    sample = snapshot()
    sample["is_active"] = 0
    sample["radius"][:] = 0
    sample["location_flags"][:] = 0
    assert validate_hand(sample, None)[1] == HandReason.INACTIVE


def test_original_apk_zero_radii_admit_and_move_fingers():
    tracker = HandTracker(Retargeter(target=0.3), max_rate=0.5)
    commands = []
    for i in range(50):
        sample = snapshot(i + 1)
        sample["radius"][:] = 0  # Active original-APK captures report all zeros.
        sample["timestamp_source"] = 1
        out = tracker.step(sample, np.zeros(3), np.zeros(7), i * 20_000_000)
        commands.append(out.command)
    assert out.reason == HandReason.OK
    assert out.state == TrackingState.TRACKING
    assert np.all(out.command > 0.25)
    assert np.max(np.abs(np.diff(commands, axis=0))) <= 0.010001


@pytest.mark.parametrize(
    "field,value",
    [
        ("pose", np.zeros((25, 7), np.float64)),
        ("pose", np.zeros((26, 7), np.float32)),
        ("radius", np.full(26, -1.0, np.float64)),
        ("radius", np.full(26, np.inf, np.float64)),
        ("scale", 0.0),
        ("scale", np.nan),
        ("location_flags", np.full(26, 10, np.int64)),
    ],
)
def test_bad_binding_values_reject(field, value):
    sample = snapshot()
    sample[field] = value
    assert validate_hand(sample, np.zeros(3))[1] == HandReason.MALFORMED


def test_anatomical_geometry_flags_and_wrist_rejections():
    sample = snapshot()
    sample["location_flags"][25] = 2
    assert validate_hand(sample, np.zeros(3))[1] == HandReason.UNTRACKED
    sample = snapshot()
    sample["pose"][3, :3] = sample["pose"][2, :3]
    assert validate_hand(sample, np.zeros(3))[1] == HandReason.GEOMETRY
    sample = snapshot()
    sample["pose"][25, 0] = 0.6
    assert validate_hand(sample, np.zeros(3))[1] == HandReason.GEOMETRY
    assert validate_hand(snapshot(), None)[1] == HandReason.BODY_WRIST_UNAVAILABLE
    assert validate_hand(snapshot(), [0.301, 0, 0])[1] == HandReason.WRIST_DISTANCE
    assert validate_hand(snapshot(), [0.30, 0, 0])[1] == HandReason.OK


def test_five_advancing_samples_and_independent_freeze_at_100ms():
    left, right = HandTracker(Retargeter()), HandTracker(Retargeter())
    for frame in range(4):
        assert step(left, frame + 1, frame * 20_000_000).state == TrackingState.WAITING
    for tick in range(4, 8):
        assert step(left, 4, tick * 20_000_000).state == TrackingState.WAITING
    admitted = step(left, 5, 160_000_000)
    assert admitted.state == TrackingState.RECOVERING
    for tick in range(5):
        out = step(left, 5, 160_000_000 + tick * 20_000_000)
        right_out = step(right, tick + 1, 160_000_000 + tick * 20_000_000)
        assert out.reason == HandReason.OK
    assert right_out.state == TrackingState.RECOVERING
    stale = step(left, 5, 260_000_000)
    assert stale.reason == HandReason.SOURCE_STALE
    assert stale.state == TrackingState.HOLDING
    assert stale.age_ns == 100_000_000
    assert step(right, 6, 260_000_000).reason == HandReason.OK


@pytest.mark.parametrize("profile,size,rate", [("dex3", 7, 2.0), ("inspire_ftp", 6, 1.0)])
def test_recovery_filter_resets_and_caps_dt_and_rate(profile, size, rate):
    tracker = HandTracker(Retargeter(size, 1.0), profile)
    for frame in range(5):
        out = step(tracker, frame + 10, frame * 20_000_000)
    assert out.state == TrackingState.RECOVERING
    np.testing.assert_array_equal(out.command, 0.0)
    out = step(tracker, 15, 1_080_000_000)
    np.testing.assert_allclose(out.command, rate * 0.04, atol=1e-8)
    held = out.command.copy()
    # Clock regression immediately freezes; regressing sample does not count toward re-admission.
    out = step(tracker, 1, 1_100_000_000)
    assert out.state == TrackingState.HOLDING
    assert out.reason == HandReason.SOURCE_RESTART
    assert out.source_epoch == 1
    for i in range(1, 5):
        out = step(tracker, 1 + i, 1_100_000_000 + i * 20_000_000)
        assert out.state == TrackingState.HOLDING
        np.testing.assert_array_equal(out.command, held)
    out = step(tracker, 6, 1_200_000_000)
    assert out.state == TrackingState.RECOVERING
    np.testing.assert_array_equal(out.command, held)
    out = step(tracker, 7, 1_220_000_000)
    assert np.all(out.command - held <= rate * 0.02 + 1e-7)
    assert np.all(out.command > held)
    assert not out.command.flags.writeable


def test_lowpass_coefficient_and_five_converged_ticks():
    tracker = HandTracker(Retargeter(target=0.01))
    for frame in range(5):
        out = step(tracker, frame + 1, frame * 20_000_000)
    assert out.state == TrackingState.RECOVERING
    out = step(tracker, 6, 100_000_000)
    np.testing.assert_allclose(out.command, -0.01 * np.expm1(-0.02 / 0.06), rtol=1e-6)
    for frame in range(7, 10):
        out = step(tracker, frame, (frame - 1) * 20_000_000)
    assert out.state == TrackingState.TRACKING


@pytest.mark.parametrize("measured", [None, "bad", [[0], [0, 1]], [np.nan] * 7, [2.0] * 7])
def test_missing_or_malformed_measurement_never_arms_or_crashes(measured):
    tracker = HandTracker(Retargeter())
    for frame in range(8):
        out = tracker.step(snapshot(frame + 1), np.zeros(3), measured, frame * 20_000_000)
        assert out.command is None
        assert out.reason == HandReason.FEEDBACK_UNAVAILABLE
        assert out.state == TrackingState.WAITING


def test_invalid_generation_and_retarget_failure_hold():
    tracker = HandTracker(Retargeter())
    sample = snapshot()
    sample["binding_generation"] = True
    out = tracker.step(sample, np.zeros(3), np.zeros(7), 0)
    assert out.reason == HandReason.MALFORMED
    for frame in range(5):
        step(tracker, frame + 1, frame * 20_000_000)
    tracker.retargeter.target[:] = np.nan
    out = step(tracker, 6, 100_000_000)
    assert out.state == TrackingState.HOLDING
    assert out.reason == HandReason.RETARGET_FAILED
    np.testing.assert_array_equal(out.command, 0.0)


@pytest.mark.parametrize("profile,size,ceiling", [("dex3", 7, 2.0), ("inspire_ftp", 6, 1.0)])
@pytest.mark.parametrize("rate", [0.0, -0.1, np.nan, np.inf, -np.inf, "above"])
def test_custom_rate_rejects_invalid_or_above_nominal(profile, size, ceiling, rate):
    if rate == "above":
        rate = np.nextafter(ceiling, np.inf)
    with pytest.raises(ValueError, match="Hand rate"):
        HandTracker(Retargeter(size), profile, rate)


@pytest.mark.parametrize("profile,size,ceiling", [("dex3", 7, 2.0), ("inspire_ftp", 6, 1.0)])
@pytest.mark.parametrize("fraction", [None, 0.25, 1.0])
def test_custom_rate_accepts_default_lower_and_nominal(profile, size, ceiling, fraction):
    rate = None if fraction is None else ceiling * fraction
    tracker = HandTracker(Retargeter(size, 1.0), profile, rate)
    expected = ceiling if rate is None else rate
    assert tracker.max_rate == expected
    for frame in range(5):
        step(tracker, frame + 1, frame * 20_000_000)
    output = step(tracker, 6, 100_000_000)
    np.testing.assert_allclose(output.command, expected * 0.02, rtol=1e-6)


@pytest.mark.parametrize("profile,size", [("dex3", 7), ("inspire_ftp", 6)])
@pytest.mark.parametrize("enabled", [False, True])
def test_startup_invalid_or_disabled_optical_uses_latest_measured_baseline(profile, size, enabled):
    tracker = HandTracker(Retargeter(size, 0.4), profile)
    # Disabled startup or unavailable optics must not freeze the first measurement.
    for tick, measured in enumerate((0.2, 0.8)):
        sample = snapshot(tick + 1) if not enabled else None
        output = tracker.step(sample, np.zeros(3), np.full(size, measured), tick * 20_000_000, enabled=enabled)
        assert output.state == TrackingState.WAITING
        np.testing.assert_allclose(output.command, measured)
        assert tracker.valid_streak == 0
    # Admission starts from the latest physical measurement, with zero recovery dt.
    for tick in range(2, 7):
        output = step(tracker, tick + 1, tick * 20_000_000, measured=np.full(size, 0.8))
    assert output.state == TrackingState.RECOVERING
    np.testing.assert_allclose(output.command, 0.8)
    held = output.command.copy()
    # After admission, disabling or losing optics holds the emitted target even if
    # measured joints move; physical feedback must not bypass the recovery limiter.
    output = tracker.step(None, np.zeros(3), np.full(size, 0.1), 140_000_000, enabled=False)
    assert output.state == TrackingState.HOLDING
    np.testing.assert_array_equal(output.command, held)
    output = tracker.step(None, np.zeros(3), np.full(size, 0.2), 160_000_000)
    assert output.state == TrackingState.HOLDING
    np.testing.assert_array_equal(output.command, held)



def test_right_index_measurement_allowance_does_not_accept_invalid_target():
    retargeter = Retargeter()
    retargeter.side = "right"
    retargeter.lower[:] = 0
    retargeter.target[5] = -0.0007002827478572726
    tracker = HandTracker(retargeter)
    measured = np.zeros(7)
    measured[5] = retargeter.target[5]
    for frame in range(5):
        output = step(tracker, frame + 1, frame * 20_000_000, measured=measured)
    assert output.reason == HandReason.RETARGET_FAILED
    assert output.state == TrackingState.HOLDING
    assert output.command[5] == 0
    assert output.target is None


def test_inspire_does_not_inherit_dex3_measurement_allowance():
    retargeter = Retargeter(size=6)
    retargeter.side = "right"
    tracker = HandTracker(retargeter, "inspire_ftp")
    measured = np.zeros(6)
    measured[5] = -0.0007
    output = step(tracker, 1, 0, measured=measured)
    assert output.reason == HandReason.FEEDBACK_UNAVAILABLE
    assert output.command is None
