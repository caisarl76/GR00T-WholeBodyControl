"""Tests for the Step 8 displacement still-standing check (synthetic data)."""

import hashlib
import json

import numpy as np

from gear_sonic.scripts.gates import g1b_still2 as S
from gear_sonic.scripts.gates import g1b_still_thresholds as T

P = 0.02


def _samples(n=30.0, q=lambda t: 0.0):
    r = np.arange(0, n, P)
    s = T.make_samples(r, np.zeros((len(r), 12)), np.zeros((len(r), 3)), period_s=P)
    s["q"] = np.stack([np.full(12, q(x)) for x in r])
    s["w"] = np.stack([[0.01, 0.0, 0.0] for _ in r])
    return s


def test_disp_feature_uses_received_window():
    s = _samples(q=lambda t: 0.1 * (t > 10.003))  # sample at 10.004 jumps by 0.1 rad: received after t=10.0
    D, G = S.window_features(s, np.array([10.0, 10.5]))
    assert D[0] == 0.0  # newest sample at 10.0 predates the jump
    assert abs(D[1] - 0.1) < 1e-12  # newest 10.5 vs oldest used (<= 9.5): the jump is inside
    assert abs(G[0] - 0.01) < 1e-9
    assert np.isnan(S.window_features(s, np.array([0.5]))[0][0])  # history does not cover the window


class _R:  # minimal Rec stand-in
    def __init__(self, since, gt):
        n = len(since)
        self.since, self.static, self.walk = np.array(since), np.ones(n, bool), np.zeros(n, bool)
        self.run = type("R", (), {"gt": np.array(gt), "unk": np.zeros(n, bool)})()
        self.D, self.G = np.full(n, 0.001), np.full(n, 0.001)


def test_dev_scope_excludes_first_0_5s():
    r = _R([0.0, 0.3, 0.49, 0.5, 2.0, 5.0], [False] * 6)
    assert list(S.in_scope(r, S.DEV_SCOPE_S)) == [False, False, False, True, True, True]
    assert list(S.in_scope(r, S.VAL_SCOPE_S)) == [False] * 5 + [True]
    assert S.pooled_boundary([_R([0.0, 0.3, 0.6], [False, False, True])]) is not None  # early false-still not counted


def test_validate_refuses_changed_hash(tmp_path):
    script, frozen, ledger = tmp_path / "s.py", tmp_path / "f.json", tmp_path / "l.md"
    script.write_text("v1")
    h = hashlib.sha256(b"v1").hexdigest()
    frozen.write_text(json.dumps({"D_max": 1, "G_max": 1, "script_sha256": h}))
    ledger.write_text(f"Step 8 FREEZE: script {h} json {hashlib.sha256(frozen.read_bytes()).hexdigest()}\n")
    assert S.check_hashes(ledger, frozen, script) is None
    script.write_text("v2")
    assert "script sha256 changed" in S.check_hashes(ledger, frozen, script)
    script.write_text("v1")
    frozen.write_text(json.dumps({"D_max": 2, "G_max": 1, "script_sha256": h}))
    assert "frozen json sha256 changed" in S.check_hashes(ledger, frozen, script)
    ledger.write_text("nothing\n")
    assert "no 'Step 8 FREEZE:'" in S.check_hashes(ledger, frozen, script)


def _res(fa=0, holds=15, iso=10, gt=100):
    return {"false_acceptances": fa, "gt_still_instants": gt, "pre_walk_holds": {"n": 15, "accepted": holds},
            "isolated_holds": {"n": 10, "accepted": iso}}


def test_verdict_failed_on_any_false_accept():
    assert S.verdict(_res()) == "PASS"
    assert S.verdict(_res(fa=1)) == "FAILED"
    assert S.verdict(_res(holds=12)) == "FAILED" and S.verdict(_res(iso=8)) == "FAILED"
    r = _res()
    r["isolated_holds"]["n"] = 0
    assert S.verdict(r) == "INCONCLUSIVE"  # missing data only
