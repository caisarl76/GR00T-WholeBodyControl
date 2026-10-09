"""Validated workstation profile and agent skill arguments."""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import yaml

from .harness_contract import SkillCall, finite_number, validate_params

LIMIT_DEFAULTS = dict(
    feedback_max_age_s=0.5,
    camera_max_age_s=0.5,
    lease_s=2.0,
    heartbeat_s=0.25,
    rpc_timeout_s=0.5,
    prewarm_deadline_s=10.0,
    policy_deadline_s=2.0,
    planner_deadline_s=2.0,
    reset_joint_tolerance_rad=0.05,
    reset_yaw_tolerance_rad=math.pi / 36,
    settle_dwell_s=0.5,
    reset_deadline_s=15.0,
    manipulation_deadline_s=120.0,
    monitor_interval_s=1.0,
    monitor_deadline_s=10.0,
    decision_expiry_s=10.0,
    walk_max_speed_mps=0.2,
    walk_max_duration_s=5.0,
    turn_max_angle_rad=math.pi / 4,
    turn_max_rate_rps=math.pi / 18,
    turn_lead_rad=math.pi / 36,
    turn_tolerance_rad=math.pi / 60,
    turn_deadline_s=10.0,
)


@dataclass(frozen=True)
class HarnessLimits:
    feedback_max_age_s: float = 0.5
    camera_max_age_s: float = 0.5
    lease_s: float = 2.0
    heartbeat_s: float = 0.25
    rpc_timeout_s: float = 0.5
    prewarm_deadline_s: float = 10.0
    policy_deadline_s: float = 2.0
    planner_deadline_s: float = 2.0
    reset_joint_tolerance_rad: float = 0.05
    reset_yaw_tolerance_rad: float = math.pi / 36
    settle_dwell_s: float = 0.5
    reset_deadline_s: float = 15.0
    manipulation_deadline_s: float = 120.0
    monitor_interval_s: float = 1.0
    monitor_deadline_s: float = 10.0
    decision_expiry_s: float = 10.0
    walk_max_speed_mps: float = 0.2
    walk_max_duration_s: float = 5.0
    turn_max_angle_rad: float = math.pi / 4
    turn_max_rate_rps: float = math.pi / 18
    turn_lead_rad: float = math.pi / 36
    turn_tolerance_rad: float = math.pi / 60
    turn_deadline_s: float = 10.0


@dataclass(frozen=True)
class SkillDefinition:
    skill_id: str
    prompt: str
    completion_criteria: str
    completion_action: str = "reset_standing"


@dataclass(frozen=True)
class HandoverSettings:
    skill_id: str
    checkpoint_verified: bool
    ready_pose_reviewed: bool
    ready_right_arm_joints: dict[str, float]

    @property
    def right_arm_target(self) -> tuple[float, ...]:
        return tuple(self.ready_right_arm_joints[name] for name in RIGHT_ARM_BOUNDS)


# Motor order and limits from g1_29dof_with_hand.xml (right arm motors 22–28).
RIGHT_ARM_BOUNDS = {
    "right_shoulder_pitch_joint": (-3.0892, 2.6704),
    "right_shoulder_roll_joint": (-2.2515, 1.5882),
    "right_shoulder_yaw_joint": (-2.618, 2.618),
    "right_elbow_joint": (-1.0472, 2.0944),
    "right_wrist_roll_joint": (-1.97222, 1.97222),
    "right_wrist_pitch_joint": (-1.61443, 1.61443),
    "right_wrist_yaw_joint": (-1.61443, 1.61443),
}


@dataclass(frozen=True)
class G1Profile:
    registry_sha256: str
    policy_host: str
    policy_port: int
    checkpoint: str
    embodiment: str
    camera_key: str
    action_horizon: int
    publish_rate: int
    limits: HarnessLimits
    skills: dict[str, SkillDefinition]
    handover: HandoverSettings | None = None

    def require_skill(self, skill_id: str) -> SkillDefinition:
        if skill_id not in self.skills:
            raise ValueError(f"Unsupported skill: {skill_id}")
        return self.skills[skill_id]


def load_profile(path: Path) -> G1Profile:
    raw = Path(path).read_bytes()
    data = yaml.safe_load(raw)
    expected = {
        "schema_version",
        "policy_host",
        "policy_port",
        "checkpoint",
        "embodiment",
        "camera_key",
        "action_horizon",
        "publish_rate",
        "limits",
        "skills",
    }
    if (
        not isinstance(data, dict)
        or data.keys() not in (expected, expected | {"handover"})
        or type(data["schema_version"]) is not int
        or data["schema_version"] != 1
    ):
        raise ValueError("Invalid profile fields/version")
    for key in ("policy_host", "checkpoint", "camera_key", "embodiment"):
        if type(data[key]) is not str or not data[key]:
            raise ValueError(f"Invalid {key}")
    if (
        data["embodiment"] != "unitree_g1_sonic"
        or data["camera_key"] != "ego_view"
        or data["action_horizon"] != 40
        or data["publish_rate"] != 50
    ):
        raise ValueError("Unsupported native representation")
    if type(data["policy_port"]) is not int or not 0 < data["policy_port"] < 65536:
        raise ValueError("Invalid policy port")
    limits = data["limits"]
    if not isinstance(limits, dict) or set(limits) - LIMIT_DEFAULTS.keys():
        raise ValueError("Invalid limits")
    limits = {**LIMIT_DEFAULTS, **limits}
    for key, value in limits.items():
        if not finite_number(value) or not 0 < value <= LIMIT_DEFAULTS[key]:
            raise ValueError(f"Invalid limit: {key}")
    if limits["settle_dwell_s"] != 0.5 or limits["heartbeat_s"] * 4 > limits["lease_s"]:
        raise ValueError("Invalid dwell/heartbeat relationship")
    skills = {}
    if not isinstance(data["skills"], dict) or not data["skills"]:
        raise ValueError("Empty skills")
    for key, spec in data["skills"].items():
        if (
            type(key) is not str
            or not key
            or key in {"reset_standing", "reset_ready", "walk_for", "turn_by"}
            or not isinstance(spec, dict)
            or not {"prompt", "completion_criteria"} <= spec.keys()
            or spec.keys() - {"prompt", "completion_criteria", "completion_action"}
        ):
            raise ValueError("Invalid skill entry")
        if any(type(v) is not str or not v.strip() for v in spec.values()):
            raise ValueError("Empty skill text")
        if spec.get("completion_action", "reset_standing") not in {
            "reset_standing",
            "hold",
        }:
            raise ValueError("Invalid completion action")
        skills[key] = SkillDefinition(key, **spec)
    if "handover" in data:
        h = data["handover"]
        if (
            not isinstance(h, dict)
            or h.keys()
            != {
                "skill_id",
                "checkpoint_verified",
                "ready_pose_reviewed",
                "ready_right_arm_joints",
            }
            or type(h["skill_id"]) is not str
            or h["skill_id"] not in skills
            or skills[h["skill_id"]].completion_action != "hold"
            or type(h["checkpoint_verified"]) is not bool
            or type(h["ready_pose_reviewed"]) is not bool
        ):
            raise ValueError("Invalid handover settings")
        joints = h["ready_right_arm_joints"]
        if not isinstance(joints, dict) or joints.keys() != RIGHT_ARM_BOUNDS.keys():
            raise ValueError("Ready pose requires seven named right-arm joints")
        for name, (low, high) in RIGHT_ARM_BOUNDS.items():
            if not finite_number(joints[name]) or not low <= joints[name] <= high:
                raise ValueError(f"Ready joint outside G1 bounds: {name}")
        data["handover"] = HandoverSettings(**h)
    del data["schema_version"]
    data.update(
        registry_sha256=hashlib.sha256(raw).hexdigest(),
        limits=HarnessLimits(**limits),
        skills=skills,
    )
    return G1Profile(**data)


def validate_skill_call(
    profile: G1Profile, call: SkillCall, *, locomotion_enabled: bool = False
) -> None:
    if (
        not isinstance(call, SkillCall)
        or type(call.skill_id) is not str
        or not isinstance(call.params, dict)
    ):
        raise ValueError("Invalid skill call")
    if call.skill_id in profile.skills:
        if call.params:
            raise ValueError("Manipulation takes no free-form arguments")
    elif call.skill_id == "reset_standing":
        if call.params.keys() != {"open_hands"}:
            raise ValueError("Standing reset requires only open_hands")
        validate_params("reset_standing", {"execution_id": "", **call.params})
    elif locomotion_enabled and call.skill_id in {"walk_for", "turn_by"}:
        validate_params(call.skill_id, call.params)
        lim = profile.limits
        if call.skill_id == "walk_for" and (
            call.params["speed_mps"] > lim.walk_max_speed_mps
            or call.params["duration_s"] > lim.walk_max_duration_s
        ):
            raise ValueError("Profile walk limits exceeded")
        if call.skill_id == "turn_by" and (
            abs(call.params["angle_rad"]) > lim.turn_max_angle_rad
            or call.params["rate_rps"] > lim.turn_max_rate_rps
        ):
            raise ValueError("Profile turn limits exceeded")
    else:
        raise ValueError(f"Unsupported skill: {call.skill_id}")
