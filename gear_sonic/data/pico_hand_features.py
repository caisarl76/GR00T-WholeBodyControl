"""Profile-specific episode hand fields and exact manager-generation joins."""

from pathlib import Path

import numpy as np

from gear_sonic.utils.teleop.pico_recording import PROFILES, parse_pv

INSPIRE_JOINTS = ("little_1", "ring_1", "middle_1", "index_1", "thumb_2", "thumb_1")
INSPIRE_RANGES = np.array([1.4381, 1.4381, 1.4381, 1.4381, 0.5864, 1.1641])
DEX3_API_TO_MODEL = [0, 1, 2, 5, 6, 3, 4]
HAND_MAX_AGE_NS = 100_000_000


def exact(fields, name, dtype, shape):
    value = fields.get(name)
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or value.shape != shape:
        raise ValueError(f"invalid {name}")
    if not np.isfinite(value).all():
        raise ValueError(f"nonfinite {name}")
    return value


def hand_episode_features(profile):
    features = {"teleop.hand_sample_generation": {"dtype": "int64", "shape": (1,), "names": ["generation"]}}
    features["teleop.hand_input"] = {"dtype": "int32", "shape": (1,), "names": ["hand_input"]}
    for side in ("left", "right"):
        for key, dtype in (
            ("tracking_state", "int32"),
            ("source_epoch", "int64"),
            ("source_timestamp_ns", "int64"),
            ("timestamp_source", "int32"),
            ("valid", "bool"),
            ("held", "bool"),
        ):
            features[f"teleop.{side}_hand_{key}"] = {"dtype": dtype, "shape": (1,), "names": [key]}
        if profile == "inspire_ftp":
            for prefix, key in (("teleop", "command"), ("observation", "state"), ("action", "applied")):
                features[f"{prefix}.{side}_inspire_hand_{key}"] = {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": list(INSPIRE_JOINTS),
                }
    if profile == "inspire_ftp":
        features["teleop.inspire_applied_message_seq"] = {
            "dtype": "int64",
            "shape": (1,),
            "names": ["message_seq"],
        }
    if profile == "inspire_ftp":
        features["teleop.inspire_intended_message_seq"] = {
            "dtype": "int64",
            "shape": (1,),
            "names": ["message_seq"],
        }
    return features


def join_hand_frame(
    body, diagnostics, profile, now_ns, manager_session, manager_mode, inspire_status=None, *, manager_epoch=None
):
    """Reject incomplete/stale generations; return only real hand episode fields."""
    if body is None or diagnostics is None:
        raise ValueError("missing body or hand diagnostics")
    for packet in (body, diagnostics):
        if not 0 <= now_ns - packet["received_ns"] < HAND_MAX_AGE_NS:
            raise ValueError("stale hand/body generation")
    session, epoch, mode = parse_pv(diagnostics.get("pv"))
    if manager_epoch is not None and epoch != manager_epoch:
        raise ValueError("wrong manager mode epoch")
    if session != manager_session or mode != manager_mode or mode not in (1, 5):
        raise ValueError("wrong manager provenance")
    generation = exact(diagnostics, "sample_generation", np.int64, (1,))
    if generation.item() < 0 or not np.array_equal(generation, exact(body, "sample_generation", np.int64, (1,))):
        raise ValueError("different body/hand generation")
    if not np.array_equal(body.get("pv"), diagnostics["pv"]):
        raise ValueError("different body/hand provenance")
    if exact(diagnostics, "hand_profile", np.int32, (1,)).item() != PROFILES[profile]:
        raise ValueError("wrong hand profile")
    if not exact(diagnostics, "body_sent", np.bool_, (1,)).item():
        raise ValueError("no body packet for hand generation")
    hand_input = exact(diagnostics, "hand_input", np.int32, (1,))
    if hand_input.item() not in (0, 1):
        raise ValueError("unsupported recording hand input")
    result = {"teleop.hand_sample_generation": generation.copy(), "teleop.hand_input": hand_input.copy()}
    for side in ("left", "right"):
        state = exact(diagnostics, f"{side}_state", np.int32, (1,))
        if state.item() not in (1, 2, 3):
            raise ValueError("uninitialized hand")
        epoch = exact(diagnostics, f"{side}_source_epoch", np.int64, (1,))
        stamp = exact(diagnostics, f"{side}_source_timestamp_ns", np.int64, (1,))
        # Older diagnostics and controller inputs do not identify an optical clock.
        # Preserve that distinction: -1=unavailable, 0=device, 1=host fallback.
        source_key = f"{side}_timestamp_source"
        timestamp_source = (
            exact(diagnostics, source_key, np.int32, (1,))
            if source_key in diagnostics
            else np.array([-1], np.int32)
        )
        if timestamp_source.item() not in (-1, 0, 1):
            raise ValueError("invalid hand timestamp source")
        valid = exact(diagnostics, f"{side}_valid", np.bool_, (1,))
        if epoch.item() < 0 or stamp.item() <= 0:
            raise ValueError("invalid optical source metadata")
        result.update(
            {
                f"teleop.{side}_hand_tracking_state": state.copy(),
                f"teleop.{side}_hand_source_epoch": epoch.copy(),
                f"teleop.{side}_hand_source_timestamp_ns": stamp.copy(),
                f"teleop.{side}_hand_timestamp_source": timestamp_source.copy(),
                f"teleop.{side}_hand_valid": valid.copy(),
                f"teleop.{side}_hand_held": np.array([state.item() == 1], np.bool_),
            }
        )
        command = exact(diagnostics, f"{side}_command", np.float32, (7 if profile == "dex3" else 6,))
        if profile == "dex3":
            body_command = exact(body, f"{side}_hand_joints", np.float32, (7,))
            if not np.array_equal(command, body_command):
                raise ValueError("body and diagnostics commands differ")
            result[f"teleop.{side}_hand_joints"] = command.copy()
        else:
            if ((command < 0) | (command > 1)).any():
                raise ValueError("Inspire command out of range")
            result[f"teleop.{side}_inspire_hand_command"] = command.copy()
    if profile == "inspire_ftp":
        if inspire_status is None or not 0 <= now_ns - inspire_status["received_ns"] < 500_000_000:
            raise ValueError("missing/stale Inspire feedback")
        if (
            not np.array_equal(inspire_status["accepted_pv"], diagnostics["pv"])
            or not inspire_status["feedback_healthy"].item()
        ):
            raise ValueError("Inspire feedback provenance/health")
        if not inspire_status.get("ready", False):
            raise ValueError("Inspire feedback rearming")
        if exact(inspire_status, "bridge_state", np.int32, (1,)).item() != 2:
            raise ValueError("Inspire bridge is not actively applying commands")
        applied_seq = exact(inspire_status, "last_applied_message_seq", np.int64, (1,))
        intended_seq = exact(diagnostics, "inspire_message_seq", np.int64, (1,))
        if not 0 <= applied_seq.item() <= intended_seq.item():
            raise ValueError("no applied Inspire command or application ahead of intent")
        result["teleop.inspire_applied_message_seq"] = applied_seq.copy()
        result["teleop.inspire_intended_message_seq"] = intended_seq.copy()
        for side in ("left", "right"):
            result[f"observation.{side}_inspire_hand_state"] = (inspire_status[f"{side}_angle_act"] / 1000).astype(
                np.float32
            )
            result[f"action.{side}_inspire_hand_applied"] = inspire_status[f"{side}_applied"].copy()
    return result


def get_inspire_episode_model(waist_location="lower_and_upper_body", high_elbow_pose=False):
    """41 actuators in episode order; the geometry-free model is used for wrist FK.

    Passive finger joints are fixed in the derived URDF because episode FK
    queries wrists only. They are never added as fake measured coordinates.
    """
    from gear_sonic.data.robot_model.robot_model import RobotModel
    from gear_sonic.data.robot_model.supplemental_info.g1.g1_supplemental_info import (
        ElbowPose,
        G1SupplementalInfo,
        WaistLocation,
    )

    info = G1SupplementalInfo(
        waist_location=WaistLocation(waist_location),
        elbow_pose=ElbowPose.HIGH if high_elbow_pose else ElbowPose.LOW,
    )
    info.name = "G1_InspireFTP"
    for side in ("left", "right"):
        names = [f"{side}_{joint}_joint" for joint in INSPIRE_JOINTS]
        setattr(info, f"{side}_hand_actuated_joints", names)
        info.joint_groups[f"{side}_hand"]["joints"] = names
    episode_names = info.body_actuated_joints + info.left_hand_actuated_joints + info.right_hand_actuated_joints

    class InspireEpisodeModel(RobotModel):
        @property
        def joint_names(self):
            return episode_names

        def get_joint_group_indices(self, group_names):
            native = super().get_joint_group_indices(group_names)
            if not hasattr(self, "_episode_to_native"):
                return native
            reverse = {native_index: index for index, native_index in enumerate(self._episode_to_native)}
            return sorted(reverse[index] for index in native)

        def get_configuration_from_actuated_joints(
            self,
            body_actuated_joint_values=None,
            left_hand_actuated_joint_values=None,
            right_hand_actuated_joint_values=None,
        ):
            values = np.concatenate(
                [body_actuated_joint_values, left_hand_actuated_joint_values, right_hand_actuated_joint_values]
            ).astype(np.float64)
            if values.shape != (41,) or not np.isfinite(values).all():
                raise ValueError("invalid Inspire episode configuration")
            return values

        def cache_forward_kinematics(self, q, auto_clip=True):
            native = np.empty(41, np.float64)
            native[self._episode_to_native] = q
            super().cache_forward_kinematics(native, auto_clip=auto_clip)

    root = Path(__file__).parent / "robot_model/model_data/g1"
    model = InspireEpisodeModel(str(root / "g1_41dof_inspire_episode.urdf"), str(root), supplemental_info=info)
    model._episode_to_native = [model.dof_index(name) for name in episode_names]
    if model.num_joints != 41:
        raise ValueError("Inspire episode model must have 41 actuators")
    return model
