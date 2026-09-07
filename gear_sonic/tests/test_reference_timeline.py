import numpy as np
import pytest

from gear_sonic.utils.inference.reference_adapter.timeline import (
    InputLease,
    resample_trajectory,
    select_execution_index,
    trajectory_velocities,
    validate_timestamps,
)


def test_validate_and_resample_physical_time_ramp() -> None:
    source = np.array([0.0, 0.1, 0.2])
    target = np.arange(0.0, 0.201, 0.02)
    values = np.column_stack((source, 2 * source))
    assert np.allclose(resample_trajectory(source, values, target), np.column_stack((target, 2 * target)))
    assert validate_timestamps(source).dtype == np.float64


def test_resample_tail_policy_and_execution_expiry() -> None:
    with pytest.raises(ValueError):
        resample_trajectory([0, 1], [[1], [2]], [0, 2])
    assert np.allclose(resample_trajectory([0, 1], [[1], [2]], [0, 2], tail_policy="hold"), [[1], [2]])
    assert select_execution_index([1, 2, 3], 2.5, valid_until=3.5) == 1
    assert select_execution_index([1, 2, 3], 4, valid_until=4) is None


def test_velocities_single_and_ramp() -> None:
    times = np.array([0.0, 0.1, 0.2])
    expected = [[1, 0.1], [1, 0.2], [1, 0.3]]
    assert np.allclose(trajectory_velocities(times, np.column_stack((times, times**2))), expected)
    assert np.array_equal(trajectory_velocities([1], [[4, 5]]), np.zeros((1, 2)))


def test_input_lease_rejects_duplicate_expiry_and_backward_clock() -> None:
    lease = InputLease(max_action_age_seconds=1, receipt_timeout_seconds=2)
    assert lease.accept(1, received_at=10, observation_time=9.5, execution_deadline=12)
    assert not lease.accept(1, received_at=11, observation_time=10.5, execution_deadline=13)
    assert lease.is_fresh(10.5)
    assert not lease.is_fresh(12)
    assert not lease.accept(2, received_at=9, observation_time=9, execution_deadline=12)
    assert not lease.accept(2, received_at=13, observation_time=10, execution_deadline=14)
    assert lease.accept(3, received_at=14, observation_time=13.5, execution_deadline=15)


@pytest.mark.parametrize("bad", [[], [1, 1], [0, np.nan], [True, False]])
def test_bad_timestamps_rejected(bad) -> None:
    with pytest.raises(ValueError):
        validate_timestamps(bad)


@pytest.mark.parametrize("bad", [["1", "2"], [1 + 0j, 2 + 0j], [True, 1.0]])
def test_timestamp_values_must_be_real_numeric(bad) -> None:
    with pytest.raises(ValueError):
        validate_timestamps(bad)


def test_matrix_contract_rejects_nonreal_empty_and_bad_target_shapes() -> None:
    with pytest.raises(ValueError):
        resample_trajectory([0, 1], [[1 + 0j], [2 + 0j]], [0.5])
    with pytest.raises(ValueError):
        trajectory_velocities([0, 1], [[True], [False]])
    with pytest.raises(ValueError):
        resample_trajectory([0, 1], [[1], [2]], np.empty((0, 2)))


def test_invalid_execution_horizon_raises() -> None:
    with pytest.raises(ValueError):
        select_execution_index([0, 1], 0.5, valid_until=1)


def test_declared_held_tail_has_zero_velocity_even_between_source_ticks():
    source = [0.0, 1.0]
    target = [0.0, 2.0, 3.0]
    positions = resample_trajectory(source, [[0.0], [1.0]], target, tail_policy="hold")
    velocities = trajectory_velocities(target, positions, held_from=source[-1])
    assert np.array_equal(velocities[1:], np.zeros((2, 1)))
    with pytest.raises(ValueError, match="held"):
        trajectory_velocities(target, [[0.0], [1.0], [2.0]], held_from=1.0)
