import unittest
from unittest import mock

import numpy as np

from gear_sonic.scripts import pico_manager_thread_server as manager


class _StopInitialization(RuntimeError):
    pass


class _Reader:
    def get_timestamp_ns(self):
        return 123


class PicoManagerInputSourceTest(unittest.TestCase):
    def test_isaac_teleop_reader_is_used_without_xrt_initialization(self):
        reader = object()
        with (
            mock.patch.object(manager, "xrt", None),
            mock.patch.object(manager, "_init_input_source", return_value=reader) as init_source,
            mock.patch.object(manager.zmq, "Context") as context,
            mock.patch.object(manager.subprocess, "Popen") as popen,
            mock.patch.object(manager, "ControllerPoseReader") as controller_reader,
            mock.patch.object(manager, "ThreePointPose"),
            mock.patch.object(manager.time, "sleep"),
            mock.patch.object(
                manager, "PoseStreamer", side_effect=_StopInitialization
            ) as pose_streamer,
        ):
            with self.assertRaises(_StopInitialization):
                manager.run_pico_manager(input_source="isaac-teleop")

        init_source.assert_called_once_with("isaac-teleop", 15)
        context.assert_called_once_with()
        popen.assert_not_called()
        controller_reader.assert_not_called()
        # This path must pass the source reader through to the pose streamer.
        self.assertIs(pose_streamer.call_args.kwargs["reader"], reader)

    def test_controller_3pt_rejects_isaac_teleop_before_resources(self):
        with (
            mock.patch.object(manager, "_init_input_source") as init_source,
            mock.patch.object(manager.zmq, "Context") as context,
            mock.patch.object(manager.subprocess, "Popen") as popen,
        ):
            with self.assertRaises(ValueError):
                manager.run_pico_manager(controller_3pt=True, input_source="isaac-teleop")

        init_source.assert_not_called()
        context.assert_not_called()
        popen.assert_not_called()

    def test_controller_3pt_constructs_reader_with_offsets_and_conventions(self):
        reader = mock.Mock()
        xrt_mock = mock.Mock()
        with (
            mock.patch.object(
                manager, "ControllerPoseReader", return_value=reader
            ) as controller_reader,
            mock.patch.object(manager, "xrt", xrt_mock),
            mock.patch.object(manager.subprocess, "Popen") as popen,
            mock.patch.object(manager, "_init_input_source") as init_source,
            mock.patch.object(manager, "ThreePointPose", side_effect=_StopInitialization),
            mock.patch.object(manager.zmq, "Context"),
            mock.patch.object(manager.time, "sleep"),
        ):
            with self.assertRaises(_StopInitialization):
                manager.run_pico_manager(
                    controller_3pt=True,
                    left_controller_offset_rpy=(1.0, 2.0, 3.0),
                    right_controller_offset_rpy=(4.0, 5.0, 6.0),
                    controller_pose_convention="xrobotoolkit_unity",
                    headset_pose_convention="openxr_unitree",
                    headset_orientation_convention="openxr_unitree",
                )

        controller_reader.assert_called_once_with(
            max_queue_size=15,
            left_controller_offset_rpy=(1.0, 2.0, 3.0),
            right_controller_offset_rpy=(4.0, 5.0, 6.0),
            controller_pose_convention="xrobotoolkit_unity",
            headset_pose_convention="openxr_unitree",
            headset_orientation_convention="openxr_unitree",
        )
        reader.start.assert_called_once_with()
        popen.assert_called_once()
        xrt_mock.init.assert_called_once_with()
        xrt_mock.is_body_data_available.assert_not_called()
        init_source.assert_not_called()

    def test_planner_isaac_timestamp_and_all_buttons_do_not_change_mode(self):
        planner = manager.PlannerStreamer.__new__(manager.PlannerStreamer)
        planner.controller_3pt = False
        planner.reader = mock.Mock(spec=_Reader)
        planner.reader.get_timestamp_ns.return_value = 123
        planner.last_xrt_timestamp = None
        planner.freeze_vr3pt_target_once = False
        planner._last_vr3pt_pose = None
        planner.prev_ab = False
        planner.prev_xy = False
        planner.mode = manager.LocomotionMode.IDLE
        planner.dt = 0.0
        planner.last_send = manager.time.time()
        planner.yaw_accumulator = mock.Mock()
        planner.yaw_accumulator.update.return_value = np.array([1.0, 0.0])
        planner.socket = mock.Mock()

        with (
            mock.patch.object(
                manager, "get_abxy_buttons", return_value=(True, True, True, True)
            ) as buttons,
            mock.patch.object(
                manager, "get_controller_axes", return_value=(0.0, 0.0, 0.0, 0.0)
            ) as axes,
            mock.patch.object(manager, "build_planner_message", return_value=b"planner"),
        ):
            planner.run_once(manager.StreamMode.PLANNER)

        self.assertEqual(planner.last_xrt_timestamp, 123)
        self.assertEqual(planner.mode, manager.LocomotionMode.IDLE)
        self.assertFalse(planner.prev_ab)
        self.assertFalse(planner.prev_xy)
        planner.reader.get_timestamp_ns.assert_called_once_with()
        buttons.assert_called_once_with(planner.reader)
        axes.assert_called_once_with(planner.reader)
        planner.socket.send.assert_called_once_with(b"planner")


if __name__ == "__main__":
    unittest.main()
