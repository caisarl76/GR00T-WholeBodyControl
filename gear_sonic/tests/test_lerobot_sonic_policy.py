from pathlib import Path
import threading
from types import SimpleNamespace

import cv2
import msgpack
import numpy as np
import pytest
import yaml
import zmq

from gear_sonic.scripts import run_vla_inference as runner
from gear_sonic.utils.inference.harness_profile import LeRobotPolicySettings, load_profile
from gear_sonic.utils.inference.lerobot_sonic_policy import (
    RIGHT_SWAP,
    LeRobotSonicPolicyClient,
    TokenStream,
    lerobot_state,
)
from gear_sonic.utils.teleop.xr_upperbody_bridge import unpack_bridge_message

FIXTURE = Path(__file__).parent / "fixtures/g1_profile.yaml"


def test_state_is_arms_then_hands_with_the_right_hand_in_dataset_order():
    msg = dict(body_q=np.arange(29.0), left_hand_q=np.arange(100.0, 107), right_hand_q=np.arange(200.0, 207))
    expected = np.r_[np.arange(15.0, 29), np.arange(100.0, 107), np.arange(200.0, 207)]
    np.testing.assert_array_equal(lerobot_state(msg, np.arange(7)), expected)
    swapped = lerobot_state(msg, RIGHT_SWAP)
    np.testing.assert_array_equal(swapped[21:], [200, 201, 202, 205, 206, 203, 204])
    np.testing.assert_array_equal(RIGHT_SWAP[RIGHT_SWAP], np.arange(7))  # Converts both ways.
    with pytest.raises(ValueError):
        lerobot_state(dict(msg, body_q=np.zeros(28)), np.arange(7))


INFO = dict(image_keys=["observation.images.egocentric"], n_obs=1,
            shapes={"observation.images.egocentric": [3, 48, 64]}, model="s10")


class FakeServer:
    """sonic_policy_server.py's protocol, recording each request."""

    def __init__(self, chunk, info=INFO):
        self.chunk, self.info, self.requests = chunk, info, []
        self.sock = zmq.Context.instance().socket(zmq.REP)
        self.sock.bind("tcp://127.0.0.1:*")
        self.port = int(self.sock.getsockopt(zmq.LAST_ENDPOINT).decode().rsplit(":", 1)[1])
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def serve(self):
        while True:
            if not self.sock.poll(2000):
                break
            request = msgpack.unpackb(self.sock.recv(), raw=False)
            self.requests.append(request)
            reply = self.info if request["op"] == "info" else dict(
                chunk=self.chunk.tobytes(), shape=list(self.chunk.shape), latency_s=0.1)
            self.sock.send(msgpack.packb(reply, use_bin_type=True))
        self.sock.close()


@pytest.mark.parametrize("order", ["dataset", "swap"])
def test_client_sends_the_streamer_request_and_splits_the_chunk(order):
    chunk = np.tile(np.arange(78, dtype=np.float32) / 100, (40, 1))
    server = FakeServer(chunk)
    client = LeRobotSonicPolicyClient("127.0.0.1", server.port, 2000, right_order=order)
    assert client.ping()
    image = np.zeros((48, 64, 3), np.uint8)
    image[..., 0] = 200  # Red stays in channel 0: no colour conversion either way.
    state = np.arange(28, dtype=np.float32)
    action, info = client.get_action({"lerobot": dict(state=state, image=image, task="pick drink bottle")})
    client.close()
    request = server.requests[-1]
    assert request["op"] == "chunk" and request["task"] == "pick drink bottle"
    np.testing.assert_array_equal(request["states"], [state])
    decoded = cv2.imdecode(np.frombuffer(request["images"][0]["observation.images.egocentric"], np.uint8),
                           cv2.IMREAD_COLOR)
    assert decoded[..., 0].mean() > 190 and decoded[..., 2].mean() < 10
    assert action["motion_token"].shape == (1, 40, 64) and info["model"] == "s10"
    np.testing.assert_allclose(action["left_hand_joints"][0, 0], np.arange(64, 71) / 100, rtol=1e-6)
    right = np.arange(71, 78) / 100
    np.testing.assert_allclose(action["right_hand_joints"][0, 0], right[RIGHT_SWAP] if order == "swap" else right,
                               rtol=1e-6)


def test_client_rejects_a_multi_observation_policy():
    server = FakeServer(np.zeros((40, 78), np.float32), dict(INFO, n_obs=2))
    with pytest.raises(ValueError, match="single-observation"):
        LeRobotSonicPolicyClient("127.0.0.1", server.port, 2000)


def ramp(start, stop):
    rows = np.zeros((40, 78), np.float32)
    rows[:, :64] = np.linspace(start, stop, 40)[:, None]
    rows[:, 64:] = np.linspace(start, stop, 40)[:, None]
    return rows


def test_token_stream_plays_rows_by_time_and_holds_the_ends():
    stream = TokenStream(50)
    stream.set_chunk(ramp(0, 39), t0=10.0, now=10.0)
    assert stream.at(9.0)[0][0] == 0  # Before t0: the first row.
    assert stream.at(10.1)[0][0] == pytest.approx(5)  # Row 5 at 0.1 s.
    assert stream.at(10.11)[1][0] == pytest.approx(5.5)  # Interpolated, hands too.
    assert stream.at(20.0)[0][0] == pytest.approx(39)  # Held after the last row.


def test_token_stream_cross_fades_a_new_chunk_from_the_current_output():
    stream = TokenStream(50, blend_s=0.3)
    stream.set_chunk(np.full((40, 78), 1.0, np.float32), t0=0.0, now=0.0)
    stream.set_chunk(np.zeros((40, 78), np.float32), t0=0.2, now=0.2)
    assert stream.at(0.2)[0][0] == pytest.approx(1.0)
    assert stream.at(0.35)[0][0] == pytest.approx(0.5)
    assert stream.at(0.5)[0][0] == pytest.approx(0.0)
    # A chunk arriving mid-fade fades from what was being output.
    stream.set_chunk(np.full((40, 78), 1.0, np.float32), t0=0.6, now=0.6)
    stream.set_chunk(np.full((40, 78), 2.0, np.float32), t0=0.75, now=0.75)
    assert stream.at(0.75)[0][0] == pytest.approx(0.5)


def test_token_stream_slew_limits_tokens_but_not_hands():
    stream = TokenStream(50, max_token_step=0.05)
    stream.reset(start_token=np.zeros(64))
    stream.set_chunk(np.ones((40, 78), np.float32), t0=0.0, now=0.0)
    tokens = [stream.at(i * 0.02) for i in range(25)]
    assert tokens[0][0][0] == pytest.approx(0.05) and tokens[0][1][0] == 1.0
    assert tokens[19][0][0] == pytest.approx(1.0) and tokens[24][0][0] == pytest.approx(1.0)
    unlimited = TokenStream(50, max_token_step=0.05)  # No start token: the first tick is not limited.
    unlimited.set_chunk(np.ones((40, 78), np.float32), t0=0.0, now=0.0)
    assert unlimited.at(0.0)[0][0] == 1.0


def write_profile(tmp_path, lerobot):
    data = yaml.safe_load(FIXTURE.read_text())
    if lerobot is not None:
        data["lerobot_policy"] = lerobot
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_profile_lerobot_section_is_optional_and_validated(tmp_path):
    assert load_profile(write_profile(tmp_path, None)).lerobot_policy is None
    assert load_profile(write_profile(tmp_path, {})).lerobot_policy == LeRobotPolicySettings()
    settings = dict(chunk_blend_s=0.3, max_token_step=0.05, dex3_right_order="swap")
    assert load_profile(write_profile(tmp_path, settings)).lerobot_policy == LeRobotPolicySettings(**settings)
    for bad in (dict(chunk_blend_s=0.9), dict(chunk_blend_s=-0.1), dict(max_token_step=float("nan")),
                dict(max_token_step=3.0), dict(dex3_right_order="mirror"), dict(rtc=True), [0.3]):
        with pytest.raises(ValueError, match="lerobot_policy"):
            load_profile(write_profile(tmp_path, bad))


def published_tokens(monkeypatch, tmp_path, lerobot):
    """Run the native loop's harness path with a scripted policy; return the published tokens."""
    from test_run_vla_reset import run_keys

    from gear_sonic.utils.inference import harness_rpc, observation_snapshot

    clock, started, pending = [0.0], [], []
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    snapshots = SimpleNamespace(start=lambda: None, close=lambda: None, latest=lambda now: dict(
        camera_key="ego_view", source_timestamp=now, frame_id=str(now), received_at=now, age_s=0.0,
        jpeg_rgb_b64="unused"))
    monkeypatch.setattr(observation_snapshot, "ObservationSnapshotCache", lambda *args: snapshots)

    class RPC:
        def __init__(self, *args):
            self.count = 0

        def start(self):
            pass

        def close(self):
            pass

        def drain(self, control, now):
            control.observation = snapshots.latest(now)
            self.count += 1
            if self.count < 2:
                return
            method, params = "heartbeat", {"sequence": self.count}
            if self.count == 2:
                method, params = "claim_control", {"registry_sha256": control.profile.registry_sha256}
            elif self.count == 3:
                method, params = "start_manipulation", {"skill_id": "bottle_to_right_table"}
            assert control.dispatch(dict(version=1, request_id=str(self.count), runtime_id=control.runtime_id,
                                         session_id="s", lease_id=control.owner["lease_id"] if control.owner else None,
                                         method=method, params=params), now)["error"] is None

    monkeypatch.setattr(harness_rpc, "HarnessRPCServer", RPC)
    measured = dict(index=0, body_q=np.zeros(29), body_q_measured_motor=np.zeros(29),
                    harness_planner_hold_enabled=[1], left_hand_q=np.zeros(7), right_hand_q=np.zeros(7),
                    left_hand_q_measured=np.zeros(7), right_hand_q_measured=np.zeros(7), base_quat=[1, 0, 0, 0],
                    token_state=np.zeros(64))

    def action(value):
        return dict(motion_token=np.full((1, 40, 64), value), left_hand_joints=np.zeros((1, 40, 7)),
                    right_hand_joints=np.zeros((1, 40, 7)))

    def on_key(index, args):
        clock[0] = (index + 1) * .02
        measured["index"] += 1
        requests, results, _, busy = args[:4]
        if pending and clock[0] >= pending[0][0]:
            _, captured_at, epoch, value = pending.pop(0)
            results.put((action(value), captured_at, epoch))
            busy.clear()
        if not requests.empty() and not busy.is_set():
            epoch = requests.get_nowait()
            started.append(epoch)
            if len(started) == 1:  # Prewarm.
                results.put((action(0.5), clock[0], epoch))
            else:  # Policy chunks alternate between token values 0.5 and 0.2.
                pending.append((clock[0] + 0.25, clock[0], epoch, 0.5 if len(started) % 2 == 0 else 0.2))
                busy.set()

    messages = run_keys(monkeypatch, ["k"] + [None] * 119, measured, on_key, runner.InferenceConfig(
        harness_endpoint="ipc:///tmp/unused-test.sock", harness_profile=str(write_profile(tmp_path, lerobot))))
    return np.array([unpack_bridge_message(raw, topic="pose")["token_state"][0, 0]
                     for raw in messages if raw.startswith(b"pose")])


def test_native_loop_plays_lerobot_chunks_like_the_streamer(monkeypatch, tmp_path):
    tokens = published_tokens(monkeypatch, tmp_path, dict(chunk_blend_s=0.3, max_token_step=0.05))
    assert len(tokens) > 75
    assert tokens[0] == pytest.approx(0.05)  # Slewed from the deploy's planner token (0), not a jump to 0.5.
    assert np.max(np.abs(np.diff(tokens))) <= 0.05 + 1e-6
    # A new chunk cross-fades over 0.3 s (15 ticks): steps of 0.02, below the 0.05 slew limit.
    fall = np.flatnonzero(np.diff(tokens) < -1e-6)
    assert len(fall) >= 14 and np.max(np.abs(np.diff(tokens)[fall])) <= 0.0201


def test_native_loop_without_lerobot_section_publishes_chunk_rows_unchanged(monkeypatch, tmp_path):
    tokens = published_tokens(monkeypatch, tmp_path, None)
    assert tokens[0] == pytest.approx(0.5)  # No slew from the planner token.
    assert np.all(np.isclose(tokens, 0.5) | np.isclose(tokens, 0.2))  # Rows switch without a cross-fade.
