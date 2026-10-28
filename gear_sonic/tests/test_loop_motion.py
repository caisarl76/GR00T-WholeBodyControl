import hashlib
import json

import numpy as np
import pytest

from gear_sonic.utils.inference.lerobot_sonic_policy import TokenStream
from gear_sonic.utils.inference.loop_motion import READY_POSE_PATH, PoseCommander, load_ready_pose

SHAPES = dict(
    token=(64,), hands=(14,), state=(28,), raise_tokens=(83, 64), raise_hands=(83, 14),
    lower_tokens=(83, 64), lower_hands=(83, 14),
)


def _good():
    return {k: np.zeros(s, np.float32) for k, s in SHAPES.items()}


def test_load_ready_pose_shapes_and_sha():
    pose = load_ready_pose()
    for name, shape in SHAPES.items():
        arr = getattr(pose, name)
        assert arr.shape == shape and arr.dtype == np.float32
    meta = json.loads(READY_POSE_PATH.with_suffix(".json").read_text())
    assert hashlib.sha256(READY_POSE_PATH.read_bytes()).hexdigest() == meta["sha256"]


def test_ready_pose_a_is_ep76_with_the_right_shoulder_yawed_out():
    a, ep76 = load_ready_pose(), load_ready_pose(READY_POSE_PATH.parent / "ready_pose_261020_ep76.npz")
    assert READY_POSE_PATH.name == "ready_pose_261026_candA.npz"
    diff = a.state - ep76.state
    assert np.isclose(diff[9], -0.40) and np.allclose(np.delete(diff, 9), 0.0)  # state 9: right shoulder yaw
    for name in ("hands", "raise_tokens", "raise_hands", "lower_tokens", "lower_hands"):
        assert np.array_equal(getattr(a, name), getattr(ep76, name))
    assert not np.array_equal(a.token, ep76.token)  # the SONIC encoding of the new static pose


def test_load_ready_pose_rejects_bad_shapes(tmp_path):
    load_ready_pose(_save(tmp_path, _good()))
    bad = _good()
    bad["raise_tokens"] = np.zeros((82, 64), np.float32)
    with pytest.raises(ValueError, match="raise_tokens"):
        load_ready_pose(_save(tmp_path, bad))
    nan = _good()
    nan["token"][3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        load_ready_pose(_save(tmp_path, nan))


def test_load_ready_pose_rejects_missing_key_and_large_token(tmp_path):
    missing = _good()
    del missing["state"]
    with pytest.raises(ValueError, match="state"):
        load_ready_pose(_save(tmp_path, missing))
    big = _good()
    big["token"][0] = 1.3
    with pytest.raises(ValueError, match="1.25"):
        load_ready_pose(_save(tmp_path, big))


def _save(tmp_path, arrays):
    path = tmp_path / "pose.npz"
    np.savez(path, **arrays)
    return path


def _cmd(v=0.0, step=0.05):
    c = PoseCommander(step)
    c.start(np.full(64, v, np.float32), np.zeros(14, np.float32))
    return c


def _rows(n, scale=0.001):
    return np.arange(1, n + 1, dtype=np.float32)[:, None] * scale * np.ones((1, 64), np.float32)


def test_blend_is_linear_from_last_sent():
    c = _cmd()
    c.blend_to(np.ones(64), np.ones(14), ticks=50)
    out = [c.tick(i / 50) for i in range(50)]
    assert out[0][0][0] == pytest.approx(0.02) and out[0][2] == "blend"
    assert out[49][0][0] == pytest.approx(1.0)
    assert not c.busy


def test_play_emits_one_row_per_tick():
    c = _cmd()
    toks, hands = _rows(5), np.arange(5, dtype=np.float32)[:, None] * np.ones((1, 14), np.float32)
    c.play(toks, hands, "raise")
    for i in range(5):
        t, h, lab = c.tick(0.0)
        assert t[0] == pytest.approx(toks[i, 0]) and h[0] == hands[i, 0] and lab == "raise"
    assert not c.busy
    assert c.tick(0.0)[2] == "hold"


def test_slew_limits_token_not_hands():
    c = _cmd()
    c.blend_to(np.ones(64), np.ones(14), ticks=1)
    t, h, _ = c.tick(0.0)
    assert t[0] == pytest.approx(0.05) and h[0] == 1.0


def test_freeze_mid_blend_holds_last_sent_token():
    c = _cmd()
    c.blend_to(np.ones(64), np.ones(14), ticks=50)
    for i in range(11):
        last = c.tick(i / 50)
    c.freeze()
    for i in range(20):
        t, h, lab = c.tick(1.0)
        assert lab == "frozen" and np.array_equal(t, last[0]) and np.array_equal(h, last[1])


def test_resume_continues_path_from_frozen_index():
    c = _cmd()
    toks = _rows(60)
    c.play(toks, np.zeros((60, 14), np.float32), "p")
    for _ in range(31):
        c.tick(0.0)
    c.freeze()
    assert c.frozen and c.busy
    c.tick(0.0)
    c.resume()
    assert c.tick(0.0)[0][0] == pytest.approx(toks[31, 0])


def test_follow_reads_stream_by_time():
    s = TokenStream(50.0, max_token_step=0.05)
    s.reset(np.zeros(64))
    rows = np.zeros((10, 78), np.float32)
    rows[:, :64] = np.linspace(0, 0.09, 10)[:, None]
    s.set_chunk(rows, 0.0, 0.0)
    c = _cmd()
    c.follow(s)
    t, _, lab = c.tick(5 / 50)
    assert lab == "episode" and t[0] == pytest.approx(0.05)  # target 0.05, stream slew from 0
    assert np.array_equal(c.last_token, t)


def test_follow_anchors_stream_to_sent_token_and_yields_to_blend():
    s = TokenStream(50.0, max_token_step=0.05)
    s.reset()
    s.set_chunk(np.zeros((10, 78), np.float32), 0.0, 0.0)
    c = _cmd(0.5)
    c.follow(s)
    assert c.tick(0.0)[0][0] == pytest.approx(0.45)  # stream slewed from the sent 0.5, not from None
    c.blend_to(np.zeros(64), np.zeros(14), ticks=2, label="to_ready")
    assert c.tick(0.0)[2] == "to_ready"  # follow ended; blend starts from last sent
    assert c.tick(0.0)[2] == "to_ready" and not c.busy


def test_start_and_hold_clear_queue_and_frozen_and_frozen_rejects_new():
    c = _cmd()
    c.blend_to(np.ones(64), np.ones(14))
    c.freeze()
    for fn in (lambda: c.blend_to(np.ones(64), np.ones(14)), lambda: c.play(_rows(2), np.zeros((2, 14)), "p"),
               lambda: c.follow(TokenStream(50.0))):
        with pytest.raises(RuntimeError):
            fn()
    c.hold()
    assert not c.frozen and not c.busy and c.tick(0.0)[2] == "hold"
    c.blend_to(np.ones(64), np.ones(14))
    c.freeze()
    c.start(np.zeros(64), np.zeros(14))
    assert not c.frozen and not c.busy


def test_resumed_blend_lands_on_same_path():
    a, b = _cmd(), _cmd()
    for c in (a, b):
        c.blend_to(np.ones(64), np.ones(14), ticks=40)
    ref = [a.tick(0.0)[0] for _ in range(40)]
    out = [b.tick(0.0)[0] for _ in range(10)]
    b.freeze()
    for _ in range(5):
        b.tick(0.0)
    b.resume()
    out += [b.tick(0.0)[0] for _ in range(30)]
    assert all(np.array_equal(x, y) for x, y in zip(ref, out))


def test_recent_tokens_keeps_last_25_sent():
    c = _cmd()
    c.play(_rows(30), np.zeros((30, 14), np.float32), "p")
    sent = [c.tick(0.0)[0] for _ in range(30)]
    assert len(c.recent_tokens) == 25
    assert all(np.array_equal(x, y) for x, y in zip(c.recent_tokens, sent[-25:]))


def test_blend_matches_lerobot_expression_exactly():
    ready = load_ready_pose()
    c = PoseCommander(0.0)
    z, zh = ready.lower_tokens[-1], ready.lower_hands[-1]  # a nonzero origin exercises the (1 - w) term
    c.start(z, zh)
    c.blend_to(ready.token, ready.hands, ticks=50)
    for i in range(50):
        w = (i + 1) / 50
        t, h, _ = c.tick(0.0)
        assert np.array_equal(t, np.asarray((1 - w) * z + w * ready.token, np.float32))
        assert np.array_equal(h, np.asarray((1 - w) * zh + w * ready.hands, np.float32))


def test_new_motion_ends_every_queued_follow():
    s = TokenStream(50.0)
    s.reset()
    s.set_chunk(np.zeros((10, 78), np.float32), 0.0, 0.0)
    c = _cmd()
    c.blend_to(np.ones(64) * 0.1, np.zeros(14), ticks=3, label="blend")
    c.follow(s)
    c.blend_to(np.zeros(64), np.zeros(14), ticks=3, label="to_ready")
    labels = [c.tick(0.0)[2] for _ in range(7)]
    assert labels == ["blend"] * 3 + ["to_ready"] * 3 + ["hold"]


def test_blend_rejects_zero_ticks_and_play_copies_rows():
    c = _cmd()
    with pytest.raises(ValueError):
        c.blend_to(np.zeros(64), np.zeros(14), ticks=0)
    hands = np.ones((2, 14), np.float32)
    c.play(_rows(2), hands, "p")
    hands[:] = 9
    assert c.tick(0.0)[1][0] == 1.0
