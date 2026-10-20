import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa


def _xyzw(quaternions_wxyz: np.ndarray) -> np.ndarray:
    return quaternions_wxyz[:, [1, 2, 3, 0]]


def _assert_unit_finite(quaternions: np.ndarray) -> None:
    assert np.all(np.isfinite(quaternions))
    np.testing.assert_allclose(
        np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-12
    )


def test_decompose_rotation_aa_maps_zero_to_identity_twist_and_swing():
    twist, swing = decompose_rotation_aa(
        np.zeros((1, 3), dtype=np.float64), np.array([0.0, 1.0, 0.0])
    )

    np.testing.assert_array_equal(twist, [[1.0, 0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(swing, [[1.0, 0.0, 0.0, 0.0]])


def test_decompose_rotation_aa_keeps_mixed_batch_finite_and_reconstructable():
    rotations = np.array(
        [[0.0, 0.0, 0.0], [0.2, -0.3, 0.1]], dtype=np.float64
    )
    twist, swing = decompose_rotation_aa(
        rotations, np.array([0.0, 1.0, 0.0])
    )

    _assert_unit_finite(twist)
    _assert_unit_finite(swing)
    reconstructed = Rotation.from_quat(_xyzw(twist)) * Rotation.from_quat(
        _xyzw(swing)
    )
    np.testing.assert_allclose(
        reconstructed.as_matrix(),
        Rotation.from_rotvec(rotations).as_matrix(),
        atol=1e-12,
    )


def test_decompose_rotation_aa_uses_identity_twist_for_degenerate_projection():
    rotations = np.array([[np.pi, 0.0, 0.0]], dtype=np.float64)
    twist, swing = decompose_rotation_aa(
        rotations, np.array([0.0, 1.0, 0.0])
    )

    _assert_unit_finite(twist)
    _assert_unit_finite(swing)
    np.testing.assert_allclose(
        twist, [[1.0, 0.0, 0.0, 0.0]], atol=1e-12
    )
    reconstructed = Rotation.from_quat(_xyzw(twist)) * Rotation.from_quat(
        _xyzw(swing)
    )
    np.testing.assert_allclose(
        reconstructed.as_matrix(),
        Rotation.from_rotvec(rotations).as_matrix(),
        atol=1e-12,
    )
