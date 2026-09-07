import numpy as np
import pytest

pytest.importorskip("mujoco")

from gear_sonic.scripts.compare_g1_act_hand_poses import PalmKinematics, pose_error


def test_pose_error_handles_quaternion_sign_and_known_rotation():
    root_half = np.sqrt(0.5)
    distance, angle = pose_error(
        [[0, 0, 0], [0, 0, 0]],
        [[1, 0, 0, 0], [1, 0, 0, 0]],
        [[0.03, 0.04, 0], [0, 0, 0]],
        [[-1, 0, 0, 0], [root_half, 0, 0, root_half]],
    )
    np.testing.assert_allclose(distance, [0.05, 0], atol=1e-12)
    np.testing.assert_allclose(angle, [0, np.pi / 2], atol=1e-12)


def test_palm_fk_uses_common_base_and_is_independent_of_finger_angles():
    fk = PalmKinematics()
    observation = {"body_q": np.zeros(29), "floating_base_pose": [0, 0, 0.8, 1, 0, 0, 0]}
    action = np.zeros(28)
    start = fk.compute(observation, action)
    action[14:] = 0.2
    finger_change = fk.compute(observation, action)
    translated = fk.compute({**observation, "floating_base_pose": [1, 2, 3.8, 1, 0, 0, 0]}, action)
    for side in ("left", "right"):
        for key in start[side]:
            np.testing.assert_allclose(start[side][key], finger_change[side][key], atol=1e-12)
        np.testing.assert_allclose(
            translated[side]["world_position_m"] - start[side]["world_position_m"], [1, 2, 3], atol=1e-12
        )
        np.testing.assert_allclose(
            translated[side]["torso_position_m"], start[side]["torso_position_m"], atol=1e-12
        )
    action[0] = 0.3
    moved = fk.compute(observation, action)
    assert np.linalg.norm(moved["left"]["world_position_m"] - start["left"]["world_position_m"]) > 0.02
    np.testing.assert_allclose(moved["right"]["world_position_m"], start["right"]["world_position_m"], atol=1e-12)
