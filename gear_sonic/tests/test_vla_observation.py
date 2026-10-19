"""The inference observation must retain the measured training state."""

from types import SimpleNamespace

import numpy as np

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.scripts.run_vla_inference import prepare_observation_from_sensors


def test_observation_retains_independent_measured_left_middle_fingers():
    robot = instantiate_g1_robot_model(waist_location="lower_and_upper_body")
    state = {"body_q": np.zeros(29),
             "left_hand_q": np.array([0.1, 0.2, 0.3, -0.4, -0.5, -0.6, -0.7]),
             "right_hand_q": np.zeros(7), "base_quat": np.array([1., 0., 0., 0.])}
    image = np.full((2, 3, 3), [10, 50, 120], dtype=np.uint8)
    camera = SimpleNamespace(read=lambda: {"images": {"ego_view": image}, "timestamps": {"ego_view": 1.0}})
    subscriber = SimpleNamespace(get_msg=lambda: state)
    before = state["left_hand_q"].copy()

    obs = prepare_observation_from_sensors(camera, subscriber, robot, "registered prompt")

    # Dataset observation.state stores fingers then thumb in this joint group.
    np.testing.assert_allclose(obs["state"]["left_hand"][0, 0], [-0.4, -0.5, -0.6, -0.7, 0.1, 0.2, 0.3])
    np.testing.assert_array_equal(state["left_hand_q"], before)
    np.testing.assert_array_equal(obs["video"]["ego_view"][0, 0], image)
    np.testing.assert_array_equal(obs["state"]["projected_gravity"][0, 0], [0., 0., -1.])
