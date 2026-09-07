import json
import threading

from gear_sonic.scripts.run_vla_adapter_sim import run_simulation


class FakeBridge:
    def __init__(self, received=False):
        self.low_cmd_received = received
        self.left_hand_cmd_received = False
        self.right_hand_cmd_received = False
        self.low_cmd_lock = threading.Lock()
        self.left_hand_cmd_lock = threading.Lock()
        self.right_hand_cmd_lock = threading.Lock()
        self.low_cmd = type("Command", (), {"motor_cmd": [type("Motor", (), {"q": 1.25})()]})()
        self.left_hand_cmd = type("Command", (), {"motor_cmd": [type("Motor", (), {"q": 2.5})()]})()
        self.right_hand_cmd = type("Command", (), {"motor_cmd": [type("Motor", (), {"q": 3.75})()]})()


class FakeEnv:
    def __init__(self, bad=False, fall_after_step=False):
        self.steps = 0
        self.bad = bad
        self.fall_after_step = fall_after_step
        self.fall = False
        self.elastic_band = type("Band", (), {"enable": True})()

    def sim_step(self):
        self.steps += 1
        if self.fall_after_step and self.steps >= 1:
            self.fall = True

    def update_render_caches(self):
        pass


class FakeSim:
    sim_dt = 0.01
    image_dt = 0.02

    def __init__(self, received=False, bad=False, fall_after_step=False):
        self.sim_env = FakeEnv(bad, fall_after_step)
        self.sim_env.prepare_obs = self.prepare_obs
        self.unitree_bridge = FakeBridge(received)
        self.bad = bad
        self.closed = False

    def get_privileged_obs(self):
        # The repository's minimal BaseSimulator exposes an empty privileged
        # observation hook; measured joint state comes from prepare_obs.
        return {}

    def prepare_obs(self):
        value = float("nan") if self.bad else 0.0
        return {
            "body_q": [value],
            "body_dq": [0.0],
            "left_hand_q": [0.0],
            "left_hand_dq": [0.0],
            "right_hand_q": [0.0],
            "right_hand_dq": [0.0],
            "floating_base_pose": [0.0],
        }

    def close(self):
        self.closed = True

    def handle_keyboard_button(self, key):
        self.sim_env.keyboard_events.append((key, self.sim_env.steps))
        if key == "9":
            self.sim_env.elastic_band.enable = not self.sim_env.elastic_band.enable
        elif key == "backspace":
            self.sim_env.reset_completed = True


def test_bounded_loop_and_release_gate(tmp_path):
    release = tmp_path / "release"
    release.touch()
    out = tmp_path / "out"
    summary = run_simulation(FakeSim(received=True), out, 0.025, release)
    assert 2 <= summary["steps"] <= 3
    assert summary["released"] is True
    assert summary["standing_verified"] is False
    records = [line for line in (out / "observations.jsonl").read_text().splitlines() if '"event"' not in line]
    assert '"body_command_target": [1.25]' in records[0]


def test_release_requires_low_command(tmp_path):
    out = tmp_path / "out"
    release = tmp_path / "release"
    release.touch()
    summary = run_simulation(FakeSim(received=False), out, 0.01, release)
    assert summary["released"] is False
    assert summary["mode"] == "supported"


def test_nonfinite_observation_fails(tmp_path):
    summary = run_simulation(FakeSim(bad=True), tmp_path / "out", 0.01)
    assert summary["error"] == "nonfinite observation"
    assert summary["finite"] is False


def test_fall_after_release_fails_and_logs_event(tmp_path):
    release = tmp_path / "release"
    release.touch()
    output = tmp_path / "out"
    sim = FakeSim(received=True, fall_after_step=True)
    summary = run_simulation(sim, output, 0.01, release)
    assert summary["fall_detected"] is True
    assert summary["finite"] is False
    assert "fall detected" in summary["error"]
    assert sim.closed is True
    events = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events if event.get("event") == "fall_detected"] == ["fall_detected"]


def test_startup_unhangs_then_resets_before_first_step(tmp_path):
    request = tmp_path / "startup.request"
    request.touch()
    sim = FakeSim(received=True)
    sim.sim_env.keyboard_events = []
    sim.sim_env.reset_completed = False
    output = tmp_path / "out"
    summary = run_simulation(sim, output, 0.025, startup_file=request)
    assert sim.sim_env.keyboard_events == [("9", 0), ("backspace", 0)]
    assert not sim.sim_env.elastic_band.enable
    assert sim.sim_env.reset_completed
    assert summary["startup_reset_completed"]
    acknowledgement = json.loads((output / "startup_reset.json").read_text())
    assert acknowledgement["keyboard_sequence"] == ["9", "backspace"]
    assert acknowledgement["elastic_band_enabled"] is False
    events = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
    assert [row["key"] for row in events if row.get("event") == "keyboard"] == ["9", "backspace"]


def test_startup_waits_for_native_initialization_command(tmp_path):
    request = tmp_path / "startup.request"
    request.touch()
    sim = FakeSim(received=False)
    sim.sim_env.keyboard_events = []
    output = tmp_path / "out"
    summary = run_simulation(sim, output, 0.01, startup_file=request)
    assert sim.sim_env.keyboard_events == []
    assert not summary["startup_reset_completed"]
    assert not (output / "startup_reset.json").exists()
