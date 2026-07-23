import numpy as np
import pytest

from gear_sonic.scripts.verify_xr_dex3_controller_pinch import (
    DEPLOY_HAND_HARD_MAX,
    DEPLOY_HAND_HARD_MIN,
    SIDE_ORDER,
    validate_pinch_targets,
)

CALIBRATED = {
    "left": np.array([-0.379616, 0.516712, 0.121406, 0.0, 0.0, -1.273903, -0.419393]),
    "right": np.array([-0.379617, -0.516714, -0.121407, 0.0, 0.0, 1.273907, 0.419395]),
}


def test_verifier_uses_symmetric_physical_dex3_dds_order() -> None:
    expected = (
        "thumb_0",
        "thumb_1",
        "thumb_2",
        "middle_0",
        "middle_1",
        "index_0",
        "index_1",
    )

    assert SIDE_ORDER["left"] == expected
    assert SIDE_ORDER["right"] == expected


def test_verifier_pins_current_deploy_hard_limits() -> None:
    np.testing.assert_allclose(
        DEPLOY_HAND_HARD_MIN["left"],
        [-1.05, -0.724, 0.0, -1.57, -1.75, -1.57, -1.75],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        DEPLOY_HAND_HARD_MAX["left"],
        [1.05, 1.05, 1.75, 0.0, 0.0, 0.0, 0.0],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        DEPLOY_HAND_HARD_MIN["right"],
        [-1.05, -1.05, -1.75, 0.0, 0.0, 0.0, 0.0],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        DEPLOY_HAND_HARD_MAX["right"],
        [1.05, 0.742, 0.0, 1.57, 1.75, 1.57, 1.75],
        rtol=0,
        atol=0,
    )


def test_calibrated_targets_pass_middle_and_deploy_limit_contract() -> None:
    validate_pinch_targets(CALIBRATED)


def test_verifier_rejects_nonzero_middle_slot() -> None:
    bad = {side: values.copy() for side, values in CALIBRATED.items()}
    bad["right"][3] = 0.01
    with pytest.raises(ValueError, match="middle"):
        validate_pinch_targets(bad)


def test_verifier_rejects_target_outside_deploy_hard_limits() -> None:
    bad = {side: values.copy() for side, values in CALIBRATED.items()}
    bad["left"][0] = -1.051
    with pytest.raises(ValueError, match="hard limits"):
        validate_pinch_targets(bad)
