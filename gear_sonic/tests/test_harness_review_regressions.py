import queue
import threading

from test_harness_control import make, request, start, state

from gear_sonic.scripts import run_vla_inference as runner


def test_planner_timeout_invalidates_reset_and_preserves_hands():
    c, h = make()
    eid = start(c)
    request(c, "pause_manipulation", {"execution_id": eid}, 0.1)
    request(c, "reset_standing", {"execution_id": eid, "open_hands": True}, 0.2)
    request(c, "heartbeat", {}, 1.9)
    old_epoch = c.epoch
    c.tick(2.3, state(2, 0), 2.3)
    assert c.phase == "FAULT" and c.reason == "planner_activation_timeout"
    assert c.reset is None and c.epoch > old_epoch
    assert h.holds[-1][1] is False


def test_telemetry_restart_requires_advancement_and_operator_preparation():
    c, _ = make()
    c.tick(0.1, state(10000), 0.1)
    c.tick(0.2, state(0), 0.2)
    assert c.phase == "FAULT" and not c._fresh(0.2)
    c.tick(0.3, state(1), 0.3)
    assert c.phase == "FAULT" and c.feedback_index == 1
    c.operator_override("i")
    assert c.phase == "IDLE"
    assert request(c, "claim_control", {"registry_sha256": c.profile.registry_sha256}, 0.3)["error"] is None


def test_operator_toggle_without_owner_preserves_native_pause_state():
    c, h = make()
    start(c)
    c.owner = None
    h.enabled = True
    c.operator_override("p")
    assert h.enabled


def test_worker_reports_epoch_bound_policy_failure():
    requests, results = queue.Queue(), queue.Queue()
    stop, busy = threading.Event(), threading.Event()
    requests.put(7)

    def failed(observation):
        stop.set()
        return None

    runner._inference_worker_loop(
        requests, results, stop, busy, lambda: {"_harness_captured_at": 1.0}, failed, report_failures=True
    )
    action, captured, epoch = results.get_nowait()
    assert action is None and captured == 1.0 and epoch == 7 and not busy.is_set()
