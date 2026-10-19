import importlib

import pytest
from test_harness_control import make, request, state
from gear_sonic.utils.inference.harness_control import monotonic_clock_id


def measured_state(index, at):
    return {**state(index), 'right_hand_received_at': [at],
            'controller_clock_id': list(monotonic_clock_id()),
            'right_hand_q_measured': [0.] * 7}


def test_native_pose_uses_thumb_first_measured_joints():
    classifier = importlib.import_module('gear_sonic.utils.inference.harness_hand_pose')
    assert classifier.classify_right_hand([.028, -.341, -.538, .533, 1.188, .653, .795]) == 'closed'
    assert classifier.classify_right_hand([0.] * 7) == 'open'


def test_observation_exposes_measured_pose_not_commanded_hand():
    c, _ = make()
    measured = measured_state(2, .1)
    measured['right_hand_q'] = [1.] * 7
    c.tick(.1, measured, .1)
    c.observation = {'received_at': .1, 'frame_id': 'aligned'}
    result = request(c, 'observe', now=.12)['result']
    assert result['right_hand_state'] == 'open'
    assert result['right_hand_sample_at'] == .1
    assert result['right_hand_sample_index'] == 100_000_000


@pytest.mark.parametrize('frame_time,feedback_time,now', [
    (0., .2, .21), (.2, 0., .21), (0., 0., .51), (.2, .3, .21),
])
def test_stale_misaligned_or_future_feedback_cannot_label_image(frame_time, feedback_time, now):
    c, _ = make()
    measured = measured_state(2, feedback_time)
    c.tick(feedback_time, measured, feedback_time)
    c.observation = {'received_at': frame_time, 'frame_id': 'image'}
    result = request(c, 'observe', now=now)
    if result['error']:
        assert result['error']['code'] == 'UNAVAILABLE'
    else:
        assert result['result']['right_hand_state'] == 'unknown'
        assert result['result']['right_hand_sample_at'] is None
        assert result['result']['right_hand_sample_index'] is None


def test_repeated_dds_sample_keeps_identity_despite_new_packet_index():
    c, _ = make()
    ids = []
    for index, now in [(20, 1.01), (21, 1.05)]:
        c.tick(now, measured_state(index, 1.), now)
        c.observation = {'received_at': now, 'frame_id': str(index)}
        result = request(c, 'observe', now=now)['result']
        assert result['right_hand_sample_at'] == 1.
        ids.append(result['right_hand_sample_index'])
    assert ids == [1_000_000_000] * 2


@pytest.mark.parametrize('now', [1.35, 1.45, 1.55])
def test_old_dds_receipt_is_not_refreshed_by_new_status_packet(now):
    c, _ = make()
    c.tick(now, measured_state(20, 1.), now)
    c.observation = {'received_at': now, 'frame_id': 'new-camera'}
    result = request(c, 'observe', now=now)['result']
    assert result['right_hand_state'] == 'unknown'


@pytest.mark.parametrize('change', ['missing_receipt', 'missing_clock', 'remote_clock'])
def test_missing_or_other_host_clock_evidence_cannot_label_hand(change):
    c, _ = make()
    feedback = measured_state(2, .1)
    if change == 'missing_receipt': del feedback['right_hand_received_at']
    if change == 'missing_clock': del feedback['controller_clock_id']
    if change == 'remote_clock': feedback['controller_clock_id'] = [99]
    c.tick(.1, feedback, .1)
    c.observation = {'received_at': .1, 'frame_id': 'new-camera'}
    assert request(c, 'observe', now=.1)['result']['right_hand_state'] == 'unknown'
