"""Test-only native IPC runtime with simulated measurements and a memory sink.

There is deliberately no PUB socket, DDS channel, or robot connection here.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import uuid

import numpy as np

from gear_sonic.utils.inference.harness_control import HarnessControl, RuntimeFacts
from gear_sonic.utils.inference.harness_rpc import HarnessRPCServer
from gear_sonic.utils.inference.observation_snapshot import ObservationSnapshotCache
from gear_sonic.utils.inference.standing_reset import StandingReset
from gear_sonic.utils.teleop.xr_upperbody_bridge import G1_UPPER_BODY_JOINT_INDICES
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message


class MemoryPublisher:
    def __init__(self, evidence):
        self.path = evidence

    def record(self, event, **fields):
        with self.path.open("a") as stream:
            stream.write(json.dumps(dict(at=time.monotonic(), event=event, **fields)) + "\n")

    def send(self, payload, kind):
        assert isinstance(payload, bytes)
        self.record("publish", publisher="memory-only", kind=kind, bytes=len(payload))


class Hooks:
    def __init__(self, sink, scenario="success"):
        self.sink = sink
        self.scenario, self.inference_busy = scenario, False
        self.enabled, self.epoch, self.command, self.reset = False, 0, None, None

    def runtime_facts(self):
        return RuntimeFacts(True, self.enabled, True, self.inference_busy,
                            "POSE" if self.enabled else "PLANNER")

    def invalidate_policy_actions(self):
        self.epoch += 1
        self.sink.record("invalidate", epoch=self.epoch)
        return self.epoch

    def set_policy_prompt(self, prompt):
        self.sink.record("prompt", prompt=prompt)

    def set_policy_enabled(self, enabled):
        self.enabled = enabled
        if enabled:
            self.command, self.reset = None, None

    def request_planner_hold(self, feedback, open_hands):
        self.reset = None
        self.command = StandingReset(feedback, feedback["left_hand_q"], feedback["right_hand_q"]).command
        self.sink.record(
            "hold",
            open_hands=open_hands,
            left=self.command.left_hand_position,
            right=self.command.right_hand_position,
        )

    def begin_standing_reset(self, feedback, open_hands):
        self.reset = StandingReset(
            feedback,
            np.zeros(7) if open_hands else feedback["left_hand_q"],
            np.zeros(7) if open_hands else feedback["right_hand_q"],
        )
        self.command = self.reset.command
        self.sink.record("reset", open_hands=open_hands)
        return self.reset

    def set_planner_command(self, command):
        self.reset, self.command = None, command

    def stop_planner_motion(self):
        if self.command is not None:
            self.command = replace(self.command, mode=0, movement=(0.0, 0.0, 0.0), speed=0.0)
        self.reset = None


class MemoryHarnessControl(HarnessControl):
    """Make the first handoff encounter a busy worker without wall-clock races."""

    busy_handoff_seen = False

    def _dispatch(self, req, now):
        first_handoff = (
            self.hooks.scenario == "subskill_handoff"
            and req["method"] == "start_manipulation"
            and self.phase == "PAUSED"
            and not self.busy_handoff_seen
        )
        if first_handoff:
            self.busy_handoff_seen = True
            self.hooks.inference_busy = True
            self.hooks.sink.record("handoff_worker_busy")
        try:
            return super()._dispatch(req, now)
        finally:
            if first_handoff:
                self.hooks.inference_busy = False


def main():
    parser = argparse.ArgumentParser()
    for name in ["endpoint", "profile", "evidence"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--scenario", default="success")
    args = parser.parse_args()
    sink = MemoryPublisher(Path(args.evidence))
    hooks = Hooks(sink, args.scenario)
    control = MemoryHarnessControl(Path(args.profile), hooks, uuid.uuid4().hex)
    control.locomotion_enabled = args.scenario == "locomotion"
    cache = ObservationSnapshotCache(lambda: None, "ego_view")
    server = HarnessRPCServer(args.endpoint, cache)
    server.start()
    action = dict(
        motion_token=np.zeros((1, 40, 64)),
        left_hand_joints=np.full((1, 40, 7), 0.3),
        right_hand_joints=np.full((1, 40, 7), 0.4),
    )
    feedback = dict(
        index=0,
        body_q=np.zeros(29),
        body_q_measured_motor=np.zeros(29),
        harness_planner_hold_enabled=[1],
        left_hand_q=np.full(7, 0.3),
        right_hand_q=np.full(7, 0.4),
        left_hand_q_measured=np.full(7, 0.3),
        right_hand_q_measured=np.full(7, 0.4),
        base_quat=[1, 0, 0, 0],
        reference_heading_quat=[1, 0, 0, 0],
        planner_reference_active=[1],
    )
    camera_stamp = 0.0
    previous_phase, old_epoch, rejected = None, None, False
    takeover_at = None
    try:
        while True:
            now = time.monotonic()
            feedback = {
                **feedback,
                "index": feedback["index"] + 1,
                "planner_reference_active": [0 if hooks.enabled or args.scenario == "failed_ack" else 1],
            }
            if args.scenario != "frozen_camera" or control.phase != "MANIPULATING":
                camera_stamp = now
            cache.ingest(
                {
                    "timestamps": {"ego_view": camera_stamp},
                    "images": {"ego_view": np.zeros((12, 16, 3), np.uint8)},
                },
                now,
            )
            control.tick(now, feedback, now)
            server.drain(control, now)
            if args.scenario == "operator_pause" and control.phase == "MANIPULATING":
                takeover_at = now + 0.2 if takeover_at is None else takeover_at
                if now >= takeover_at:
                    old_epoch = control.epoch
                    control.operator_override("p")
                    sink.record("operator_override")
                    assert not control.accept_policy_result(old_epoch, now, action, now)
                    sink.record("late_result_rejected")
            if control.phase == "MANIPULATING":
                old_epoch = control.epoch
                if control.accept_policy_result(control.epoch, now, action, now):
                    sink.send(
                        pack_pose_message(
                            {
                                "token_state": action["motion_token"][:, 0],
                                "frame_index": np.array([feedback["index"]], dtype=np.int64),
                                "left_hand_joints": action["left_hand_joints"][:, 0],
                                "right_hand_joints": action["right_hand_joints"][:, 0],
                            },
                            topic="pose",
                            version=4,
                        ),
                        "pose",
                    )
            if args.scenario == "late_result" and control.phase == "RESETTING" and not rejected:
                assert not control.accept_policy_result(old_epoch, now, action, now)
                sink.record("late_result_rejected")
                rejected = True
            if args.scenario == "subskill_handoff" and control.phase == "PAUSED" and not rejected:
                assert not control.accept_policy_result(old_epoch, now, action, now)
                sink.record("late_result_rejected", epoch=old_epoch)
                rejected = True
            if hooks.reset is not None and control.hold_confirmed:
                hooks.command = hooks.reset.advance(feedback, 0.02)
            if hooks.command is not None:
                sink.send(hooks.command.encode(), "planner")
                body = feedback["body_q"].copy()
                body[G1_UPPER_BODY_JOINT_INDICES] = hooks.command.upper_body_position
                yaw = np.arctan2(hooks.command.facing[1], hooks.command.facing[0])
                feedback = {
                    **feedback,
                    "body_q": body,
                    "body_q_measured_motor": body.copy(),
                    "left_hand_q": list(hooks.command.left_hand_position),
                    "right_hand_q": list(hooks.command.right_hand_position),
                    "left_hand_q_measured": list(hooks.command.left_hand_position),
                    "right_hand_q_measured": list(hooks.command.right_hand_position),
                    "base_quat": [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)],
                }
            if control.phase != previous_phase:
                sink.record("phase", phase=control.phase)
                if control.phase == "COMPLETED" and control.reset is not None:
                    assert control.reset.is_settled(
                        feedback,
                        control.profile.limits.reset_joint_tolerance_rad,
                        control.profile.limits.reset_yaw_tolerance_rad,
                    )
                    sink.record("measured_settled")
                previous_phase = control.phase
            time.sleep(0.02)
    finally:
        server.close()


if __name__ == "__main__":
    main()
