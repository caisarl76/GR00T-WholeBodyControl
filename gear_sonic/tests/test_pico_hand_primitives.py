from itertools import permutations
import unittest

import numpy as np

from gear_sonic.scripts import pico_manager_thread_server as pico_server
from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
    G1GripperInverseKinematicsSolver,
)

THUMB_TIP = 4
INDEX_TIP = 9
MIDDLE_TIP = 14
RIGHT_PINCH_OPEN = np.array(
    [
        0.40253138542175293,
        -0.1496349275112152,
        -0.016162125393748283,
        0.6290951371192932,
        -0.03359885886311531,
        -0.02111954055726528,
        -0.030526945367455482,
    ]
)
RIGHT_PINCH_CLOSED = np.array(
    [
        0.4024966359138489,
        -0.7084636688232422,
        -0.01614512875676155,
        1.4996159076690674,
        -0.033972788602113724,
        -0.021152639761567116,
        -0.030526945367455482,
    ]
)
DEX3_MIRROR = np.array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
LEFT_PINCH_OPEN = RIGHT_PINCH_OPEN * DEX3_MIRROR
LEFT_PINCH_CLOSED = RIGHT_PINCH_CLOSED * DEX3_MIRROR
RIGHT_FIRST_SEGMENT_MIDPOINT = 0.5 * RIGHT_PINCH_OPEN
RIGHT_SECOND_SEGMENT_MIDPOINT = 0.5 * (RIGHT_PINCH_OPEN + RIGHT_PINCH_CLOSED)
LEFT_FIRST_SEGMENT_MIDPOINT = RIGHT_FIRST_SEGMENT_MIDPOINT * DEX3_MIRROR
LEFT_SECOND_SEGMENT_MIDPOINT = RIGHT_SECOND_SEGMENT_MIDPOINT * DEX3_MIRROR
CALIBRATED_ZERO_OFFSET_TOLERANCE = np.array([0.0, 0.0, 0.0, 0.0, 0.04, 0.04, 0.04])


class GraspFingerEncodingTest(unittest.TestCase):
    def test_open_trigger_encodes_no_finger_contact(self):
        fingertips = pico_server.generate_grasp_finger_data("left", trigger=0.0)

        self.assertEqual(fingertips.shape, (25, 4, 4))
        self.assertEqual(fingertips[THUMB_TIP, 0, 3], 1.0)
        self.assertEqual(fingertips[INDEX_TIP, 0, 3], 0.0)
        self.assertEqual(fingertips[MIDDLE_TIP, 0, 3], 0.0)

    def test_index_finger_trigger_encodes_existing_grasp(self):
        fingertips = pico_server.generate_grasp_finger_data("left", trigger=1.0)

        self.assertEqual(fingertips[INDEX_TIP, 0, 3], 0.0)
        self.assertEqual(fingertips[MIDDLE_TIP, 0, 3], 1.0)

    def test_primitive_threshold_is_strict(self):
        fingertips = pico_server.generate_grasp_finger_data("left", trigger=0.5)

        self.assertEqual(fingertips[INDEX_TIP, 0, 3], 0.0)
        self.assertEqual(fingertips[MIDDLE_TIP, 0, 3], 0.0)


class PinchPrimitiveInterpolationTest(unittest.TestCase):
    def test_endpoint_results_cannot_mutate_calibration_state(self):
        right_open, right_closed = pico_server._pinch_endpoints("right")
        original_open = right_open.copy()
        original_closed = right_closed.copy()

        try:
            right_open.fill(99.0)
            right_closed.fill(-99.0)
            fresh_open, fresh_closed = pico_server._pinch_endpoints("right")

            np.testing.assert_allclose(fresh_open, RIGHT_PINCH_OPEN)
            np.testing.assert_allclose(fresh_closed, RIGHT_PINCH_CLOSED)
        finally:
            np.copyto(right_open, original_open)
            np.copyto(right_closed, original_closed)

    def test_pinch_endpoints_follow_deployment_motor_contract(self):
        expected_motor_order = (
            "thumb0",
            "thumb1",
            "thumb2",
            "middle0",
            "middle1",
            "index0",
            "index1",
        )
        self.assertEqual(pico_server.DEX3_MOTOR_ORDER, expected_motor_order)

        deployment_contracts = {
            "left": {
                "lower": np.array([-1.05, -0.724, 0.0, -1.57, -1.75, -1.57, -1.75]),
                "upper": np.array([1.05, 1.05, 1.75, 0.0, 0.0, 0.0, 0.0]),
                "endpoints": (
                    (0.2, LEFT_PINCH_OPEN),
                    (1.0, LEFT_PINCH_CLOSED),
                ),
            },
            "right": {
                "lower": np.array([-1.05, -1.05, -1.75, 0.0, 0.0, 0.0, 0.0]),
                "upper": np.array([1.05, 0.742, 0.0, 1.57, 1.75, 1.57, 1.75]),
                "endpoints": (
                    (0.2, RIGHT_PINCH_OPEN),
                    (1.0, RIGHT_PINCH_CLOSED),
                ),
            },
        }
        for hand, contract in deployment_contracts.items():
            for grip, expected_endpoint in contract["endpoints"]:
                with self.subTest(hand=hand, grip=grip):
                    endpoint = pico_server.compute_pinch_joints(hand, grip)
                    self.assertEqual(endpoint.shape, (len(expected_motor_order),))
                    np.testing.assert_allclose(endpoint, expected_endpoint)
                    lower_violation = np.maximum(contract["lower"] - endpoint, 0.0)
                    upper_violation = np.maximum(endpoint - contract["upper"], 0.0)
                    self.assertTrue(np.all(lower_violation <= CALIBRATED_ZERO_OFFSET_TOLERANCE))
                    self.assertTrue(np.all(upper_violation <= CALIBRATED_ZERO_OFFSET_TOLERANCE))
                    self.assertTrue(np.all(np.isfinite(endpoint)))

    def test_calibrated_targets_include_open_and_closed_positions(self):
        np.testing.assert_allclose(pico_server.compute_pinch_joints("right", 0.0), np.zeros(7))
        np.testing.assert_allclose(pico_server.compute_pinch_joints("left", 0.0), np.zeros(7))
        np.testing.assert_allclose(pico_server.compute_pinch_joints("right", 0.2), RIGHT_PINCH_OPEN)
        np.testing.assert_allclose(pico_server.compute_pinch_joints("left", 0.2), LEFT_PINCH_OPEN)
        np.testing.assert_allclose(pico_server.compute_pinch_joints("right", 1.0), RIGHT_PINCH_CLOSED)
        np.testing.assert_allclose(pico_server.compute_pinch_joints("left", 1.0), LEFT_PINCH_CLOSED)

    def test_first_segment_midpoint_interpolates_from_zero_to_open(self):
        np.testing.assert_allclose(
            pico_server.compute_pinch_joints("right", 0.1),
            RIGHT_FIRST_SEGMENT_MIDPOINT,
        )
        np.testing.assert_allclose(
            pico_server.compute_pinch_joints("left", 0.1),
            LEFT_FIRST_SEGMENT_MIDPOINT,
        )

    def test_second_segment_midpoint_interpolates_from_open_to_closed(self):
        np.testing.assert_allclose(
            pico_server.compute_pinch_joints("right", 0.6),
            RIGHT_SECOND_SEGMENT_MIDPOINT,
        )
        np.testing.assert_allclose(
            pico_server.compute_pinch_joints("left", 0.6),
            LEFT_SECOND_SEGMENT_MIDPOINT,
        )

    def test_closed_transition_moves_middle_while_index_stays_neutral(self):
        open_pose, closed_pose = pico_server._pinch_endpoints("right")
        delta = closed_pose - open_pose
        thumb1 = pico_server.DEX3_MOTOR_ORDER.index("thumb1")
        middle0 = pico_server.DEX3_MOTOR_ORDER.index("middle0")
        index_joints = [
            pico_server.DEX3_MOTOR_ORDER.index("index0"),
            pico_server.DEX3_MOTOR_ORDER.index("index1"),
        ]

        self.assertGreater(abs(delta[thumb1]), 0.5)
        self.assertGreater(abs(delta[middle0]), 0.8)
        self.assertTrue(np.all(np.abs(open_pose[index_joints]) < 0.04))
        self.assertTrue(np.all(np.abs(delta[index_joints]) < 5e-5))

    def test_out_of_range_grip_is_clamped(self):
        np.testing.assert_allclose(pico_server.compute_pinch_joints("left", -1.0), np.zeros(7))
        np.testing.assert_allclose(pico_server.compute_pinch_joints("right", 2.0), RIGHT_PINCH_CLOSED)

    def test_nonfinite_grip_fails_open(self):
        for grip in (np.nan, np.inf, -np.inf):
            with self.subTest(grip=grip):
                np.testing.assert_allclose(pico_server.compute_pinch_joints("right", grip), np.zeros(7))

    def test_invalid_hand_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "hand must be 'left' or 'right'"):
            pico_server.compute_pinch_joints("center", 0.2)


class DataCollectionChordTrackerTest(unittest.TestCase):
    def tracker_type(self):
        tracker_type = getattr(pico_server, "DataCollectionChordTracker", None)
        self.assertIsNotNone(tracker_type, "stateful data-collection chord tracker is missing")
        return tracker_type

    @staticmethod
    def update(tracker, state, grip=1.0):
        return tracker.update(state["a"], state["b"], state["x"], state["y"], grip)

    def test_clean_recording_chords_emit_only_after_face_button_release(self):
        tracker_type = self.tracker_type()
        for button, expected in (("a", (True, False)), ("b", (False, True))):
            tracker = tracker_type()
            state = {"a": False, "b": False, "x": False, "y": False}
            with self.subTest(button=button):
                state[button] = True
                self.assertEqual(self.update(tracker, state), (False, False))
                state[button] = False
                self.assertEqual(self.update(tracker, state), expected)

    def test_recording_gesture_requires_grip_through_face_button_release(self):
        tracker_type = self.tracker_type()
        tracker = tracker_type()
        state = {"a": True, "b": False, "x": False, "y": False}

        self.assertEqual(self.update(tracker, state), (False, False))
        self.assertEqual(self.update(tracker, state, grip=0.0), (False, False))
        state["a"] = False
        self.assertEqual(self.update(tracker, state, grip=0.0), (False, False))

    def test_recording_chord_threshold_is_strict(self):
        tracker_type = self.tracker_type()
        tracker = tracker_type()
        state = {"a": True, "b": False, "x": False, "y": False}

        self.assertEqual(self.update(tracker, state, grip=0.5), (False, False))
        state["a"] = False
        self.assertEqual(self.update(tracker, state, grip=0.5), (False, False))

    def test_mode_chords_are_suppressed_for_every_press_and_release_order(self):
        tracker_type = self.tracker_type()
        conflicting_chords = (
            ("a", "x"),
            ("b", "y"),
            ("a", "b"),
            ("a", "b", "x", "y"),
        )

        for chord in conflicting_chords:
            for press_order in permutations(chord):
                for release_order in permutations(chord):
                    tracker = tracker_type()
                    state = {"a": False, "b": False, "x": False, "y": False}
                    with self.subTest(
                        chord=chord,
                        press_order=press_order,
                        release_order=release_order,
                    ):
                        for button in press_order:
                            state[button] = True
                            self.assertEqual(self.update(tracker, state), (False, False))
                        for button in release_order:
                            state[button] = False
                            self.assertEqual(self.update(tracker, state), (False, False))


class HandJointPrimitiveTest(unittest.TestCase):
    def setUp(self):
        self.left_solver = G1GripperInverseKinematicsSolver(side="left")
        self.right_solver = G1GripperInverseKinematicsSolver(side="right")

    def test_inspire_profile_preserves_independent_thumb_rotation(self):
        left, right = pico_server.compute_hand_joints_from_inputs(
            None, None,
            left_trigger=0.0, left_grip=1.0,
            right_trigger=1.0, right_grip=0.0,
            hand_profile="inspire_ftp",
        )

        np.testing.assert_array_equal(left, [[1, 1, 1, 1, 1, 0]])
        np.testing.assert_array_equal(right, [[0, 0, 0, 0, 0, 1]])
        self.assertEqual(left.dtype, np.float32)
        self.assertEqual(right.dtype, np.float32)

    def test_full_grip_selects_calibrated_closed_pinch_pose(self):
        left, right = pico_server.compute_hand_joints_from_inputs(
            self.left_solver,
            self.right_solver,
            left_trigger=0.0,
            left_grip=1.0,
            right_trigger=0.0,
            right_grip=1.0,
        )

        np.testing.assert_allclose(left, LEFT_PINCH_CLOSED)
        np.testing.assert_allclose(right, RIGHT_PINCH_CLOSED)

    def test_grips_route_independently_to_their_pinch_trajectory_positions(self):
        left, right = pico_server.compute_hand_joints_from_inputs(
            self.left_solver,
            self.right_solver,
            left_trigger=0.0,
            left_grip=0.2,
            right_trigger=0.0,
            right_grip=0.6,
        )

        np.testing.assert_allclose(left, LEFT_PINCH_OPEN)
        np.testing.assert_allclose(right, RIGHT_SECOND_SEGMENT_MIDPOINT)

    def test_trigger_preserves_existing_middle_close_grasp_pose(self):
        left, right = pico_server.compute_hand_joints_from_inputs(
            self.left_solver,
            self.right_solver,
            left_trigger=1.0,
            left_grip=0.0,
            right_trigger=1.0,
            right_grip=0.0,
        )

        np.testing.assert_allclose(left, [0.0, 0.7, 0.7, -1.0, -1.5, -1.0, -1.5])
        np.testing.assert_allclose(right, [0.0, -0.7, -0.7, 1.0, 1.5, 1.0, 1.5])

    def test_trigger_grasp_has_priority_over_simultaneous_grip(self):
        left, right = pico_server.compute_hand_joints_from_inputs(
            self.left_solver,
            self.right_solver,
            left_trigger=1.0,
            left_grip=1.0,
            right_trigger=1.0,
            right_grip=1.0,
        )

        np.testing.assert_allclose(left, [0.0, 0.7, 0.7, -1.0, -1.5, -1.0, -1.5])
        np.testing.assert_allclose(right, [0.0, -0.7, -0.7, 1.0, 1.5, 1.0, 1.5])

    def test_left_and_right_controls_are_routed_independently(self):
        left, right = pico_server.compute_hand_joints_from_inputs(
            self.left_solver,
            self.right_solver,
            left_trigger=0.0,
            left_grip=0.2,
            right_trigger=1.0,
            right_grip=0.0,
        )

        np.testing.assert_allclose(left, LEFT_PINCH_OPEN)
        np.testing.assert_allclose(right, [0.0, -0.7, -0.7, 1.0, 1.5, 1.0, 1.5])

    def test_missing_solvers_keep_zero_fallback(self):
        left, right = pico_server.compute_hand_joints_from_inputs(
            None,
            None,
            left_trigger=1.0,
            left_grip=1.0,
            right_trigger=1.0,
            right_grip=1.0,
        )

        np.testing.assert_array_equal(left, np.zeros((1, 7), dtype=np.float32))
        np.testing.assert_array_equal(right, np.zeros((1, 7), dtype=np.float32))

    def test_one_missing_solver_keeps_both_hands_at_zero(self):
        for left_solver, right_solver in (
            (None, self.right_solver),
            (self.left_solver, None),
        ):
            with self.subTest(
                left_solver=left_solver is not None,
                right_solver=right_solver is not None,
            ):
                left, right = pico_server.compute_hand_joints_from_inputs(
                    left_solver,
                    right_solver,
                    left_trigger=1.0,
                    left_grip=1.0,
                    right_trigger=1.0,
                    right_grip=1.0,
                )

                np.testing.assert_array_equal(left, np.zeros((1, 7), dtype=np.float32))
                np.testing.assert_array_equal(right, np.zeros((1, 7), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
