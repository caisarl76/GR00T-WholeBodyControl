import re
import unittest
from xml.etree import ElementTree

import mujoco
import numpy as np

from gear_sonic.scripts import (
    pico_manager_thread_server as pico_server,
    render_dex3_pinch_primitive as pinch_renderer,
)

CANONICAL_MODEL = pinch_renderer.REPO_ROOT / "gear_sonic_deploy" / "g1" / "g1_29dof_with_hand.xml"
RIGHT_MODEL_JOINT_ORDER = (
    "thumb0",
    "thumb1",
    "thumb2",
    "index0",
    "index1",
    "middle0",
    "middle1",
)


def _semantic_name(name: str) -> str:
    if name in {"wrist", "right_wrist_yaw_link"}:
        return "wrist"

    match = re.fullmatch(r"right_hand_(thumb|index|middle)_([0-2])_(?:link|joint)", name)
    if match is None:
        match = re.fullmatch(r"(thumb|index|middle)([0-2])(?:_body)?", name)
    if match is None:
        raise ValueError(f"unrecognized Dex3 name: {name}")
    return "".join(match.groups())


def _vector(element: ElementTree.Element, attribute: str) -> tuple[float, ...]:
    return tuple(float(value) for value in element.get(attribute, "0 0 0").split())


def _quaternion(element: ElementTree.Element) -> tuple[float, ...]:
    return tuple(float(value) for value in element.get("quat", "1 0 0 0").split())


def _canonical_mesh_name(name: str) -> str:
    semantic_name = _semantic_name(name)
    match = re.fullmatch(r"(thumb|index|middle)([0-2])", semantic_name)
    if match is None:
        raise ValueError(f"unrecognized Dex3 mesh: {name}")
    return f"right_hand_{match.group(1)}_{match.group(2)}_link"


def _visual_mesh(body: ElementTree.Element) -> str:
    meshes = {_canonical_mesh_name(geom.attrib["mesh"]) for geom in body.findall("geom") if "mesh" in geom.attrib}
    if len(meshes) != 1:
        raise ValueError(f"Dex3 body {body.get('name')!r} must reference exactly one visual mesh")
    return meshes.pop()


def _joint_contracts(root: ElementTree.Element) -> dict[str, dict[str, object]]:
    parent_by_child = {child: parent for parent in root.iter() for child in parent}
    contracts = {}
    for joint in root.iter("joint"):
        joint_name = joint.get("name")
        if joint_name is None:
            continue
        try:
            semantic_name = _semantic_name(joint_name)
        except ValueError:
            continue
        if semantic_name not in RIGHT_MODEL_JOINT_ORDER:
            continue

        body = parent_by_child[joint]
        parent_body = parent_by_child[body]
        contracts[semantic_name] = {
            "axis": _vector(joint, "axis"),
            "range": _vector(joint, "range"),
            "joint_position": _vector(joint, "pos"),
            "body_position": _vector(body, "pos"),
            "body_orientation": _quaternion(body),
            "parent_body": _semantic_name(parent_body.attrib["name"]),
            "visual_mesh": _visual_mesh(body),
        }
    return contracts


class MotorToModelMappingTest(unittest.TestCase):
    def test_production_motor_names_derive_exact_model_mapping(self):
        self.assertEqual(
            getattr(pinch_renderer, "RIGHT_MODEL_JOINT_ORDER", None),
            RIGHT_MODEL_JOINT_ORDER,
        )
        self.assertIs(
            getattr(pinch_renderer, "DEX3_MOTOR_ORDER", None),
            pico_server.DEX3_MOTOR_ORDER,
        )

        derived = np.array([pico_server.DEX3_MOTOR_ORDER.index(name) for name in RIGHT_MODEL_JOINT_ORDER])
        np.testing.assert_array_equal(derived, [0, 1, 2, 5, 6, 3, 4])
        np.testing.assert_array_equal(pinch_renderer.RIGHT_MODEL_FROM_MOTOR, derived)

    def test_qpos_assignment_resolves_reordered_joint_addresses_by_name(self):
        joint_bodies = "".join(
            f'<body><joint name="{name}"/><geom size="0.001"/></body>'
            for name in reversed(RIGHT_MODEL_JOINT_ORDER)
        )
        model = mujoco.MjModel.from_xml_string(f"<mujoco><worldbody>{joint_bodies}</worldbody></mujoco>")
        data = mujoco.MjData(model)
        motor_q = np.arange(7, dtype=np.float64) + 0.25

        assign_qpos = getattr(pinch_renderer, "_set_right_hand_qpos", None)
        self.assertIsNotNone(assign_qpos)
        assign_qpos(model, data, motor_q)

        for model_joint_name in RIGHT_MODEL_JOINT_ORDER:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, model_joint_name)
            qpos_address = model.jnt_qposadr[joint_id]
            motor_index = pico_server.DEX3_MOTOR_ORDER.index(model_joint_name)
            self.assertEqual(data.qpos[qpos_address], motor_q[motor_index])


class RendererPoseStageTest(unittest.TestCase):
    def test_renderer_uses_all_four_production_pinch_stages(self):
        pose_stages = getattr(pinch_renderer, "_pose_stages", None)
        self.assertIsNotNone(pose_stages)

        stages = pose_stages()
        self.assertEqual(
            [(title, subtitle) for title, subtitle, _ in stages],
            [
                ("ALL OPEN", "Grip/Squeeze 0.0"),
                ("PINCH OPEN", "thumb-middle Grip/Squeeze 0.2"),
                ("PINCH MID", "thumb-middle Grip/Squeeze 0.6"),
                ("PINCH CLOSED", "thumb-middle Grip/Squeeze 1.0"),
            ],
        )
        for (_, _, actual), grip in zip(stages, (0.0, 0.2, 0.6, 1.0), strict=True):
            np.testing.assert_allclose(
                actual,
                pico_server.compute_pinch_joints("right", grip),
            )

    def test_rendered_closed_transition_moves_middle_not_index(self):
        pose_stages = getattr(pinch_renderer, "_pose_stages", None)
        self.assertIsNotNone(pose_stages)
        stages = pose_stages()
        open_pose = stages[1][2]
        closed_pose = stages[-1][2]
        delta = closed_pose - open_pose

        middle0 = pico_server.DEX3_MOTOR_ORDER.index("middle0")
        index_joints = [
            pico_server.DEX3_MOTOR_ORDER.index("index0"),
            pico_server.DEX3_MOTOR_ORDER.index("index1"),
        ]
        self.assertGreater(abs(delta[middle0]), 0.8)
        self.assertTrue(np.all(np.abs(delta[index_joints]) < 5e-5))


class RendererModelDriftTest(unittest.TestCase):
    def setUp(self):
        self.renderer_root = ElementTree.fromstring(pinch_renderer._mjcf())
        self.canonical_root = ElementTree.parse(CANONICAL_MODEL).getroot()

    def test_joint_kinematics_and_hierarchy_match_canonical_model(self):
        renderer_contracts = _joint_contracts(self.renderer_root)
        canonical_contracts = _joint_contracts(self.canonical_root)

        self.assertEqual(set(renderer_contracts), set(RIGHT_MODEL_JOINT_ORDER))
        self.assertEqual(renderer_contracts, canonical_contracts)

    def test_joint_contract_covers_anchor_orientation_and_visual_mesh(self):
        required_fields = {
            "axis",
            "range",
            "joint_position",
            "body_position",
            "body_orientation",
            "parent_body",
            "visual_mesh",
        }

        for joint_name, contract in _joint_contracts(self.renderer_root).items():
            with self.subTest(joint=joint_name):
                self.assertEqual(set(contract), required_fields)

    def test_visual_mesh_names_are_normalized_to_canonical_links(self):
        for joint_name, contract in _joint_contracts(self.renderer_root).items():
            match = re.fullmatch(r"(thumb|index|middle)([0-2])", joint_name)
            self.assertIsNotNone(match)
            expected_mesh = f"right_hand_{match.group(1)}_{match.group(2)}_link"
            with self.subTest(joint=joint_name):
                self.assertEqual(contract["visual_mesh"], expected_mesh)

    def test_palm_geom_offset_matches_canonical_model(self):
        renderer_palms = [geom for geom in self.renderer_root.iter("geom") if geom.get("mesh") == "palm"]
        canonical_offsets = {
            _vector(geom, "pos")
            for geom in self.canonical_root.iter("geom")
            if geom.get("mesh") == "right_hand_palm_link"
        }

        self.assertEqual(len(renderer_palms), 1)
        self.assertEqual(len(canonical_offsets), 1)
        self.assertEqual(_vector(renderer_palms[0], "pos"), canonical_offsets.pop())

    def test_mesh_geoms_are_visual_only(self):
        mesh_geoms = [geom for geom in self.renderer_root.iter("geom") if geom.get("mesh")]

        self.assertEqual(len(mesh_geoms), 8)
        for geom in mesh_geoms:
            with self.subTest(mesh=geom.get("mesh")):
                self.assertEqual(geom.get("contype"), "0")
                self.assertEqual(geom.get("conaffinity"), "0")
                self.assertEqual(geom.get("group"), "1")
                self.assertEqual(geom.get("density"), "0")


if __name__ == "__main__":
    unittest.main()
