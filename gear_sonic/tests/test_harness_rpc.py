import importlib
import importlib.util
import json
import threading
import time

import pytest
import zmq

from test_harness_control import make, request, start, state


def test_duplicate_start_is_idempotent():
    c, h = make(); eid = start(c); epoch = h.epoch
    duplicate = request(c, 'start_manipulation', {'skill_id': 'bottle_to_right_table'}, rid='start_manipulation-0.0', now=.1)
    assert duplicate['result']['execution_id'] == eid
    assert h.epoch == epoch


def test_request_id_payload_conflict():
    c, _ = make(); start(c)
    result = request(c, 'start_manipulation', {'skill_id': 'invented'}, rid='start_manipulation-0.0', now=.1)
    assert result['error']['code'] == 'REQUEST_CONFLICT'


def test_old_runtime_id_rejected():
    c, _ = make()
    result = c.dispatch(dict(version=1, request_id='r', runtime_id='old', session_id=None, lease_id=None, method='get_status', params={}), 0.)
    assert result['error']['code'] == 'STALE_RUNTIME'


def test_only_one_owner():
    c, _ = make(); start(c)
    result = request(c, 'claim_control', {'registry_sha256': c.profile.registry_sha256}, session='other', rid='other-claim')
    assert result['error']['code'] == 'BUSY'


def test_lease_expires_without_motion_replay():
    c, h = make(); start(c)
    request(c, 'heartbeat', now=.1, rid='hb')
    request(c, 'heartbeat', now=1.9, rid='hb')
    c.tick(2.11, state(2), 2.11)
    assert c.owner is None
    assert not h.enabled


def rpc_module():
    assert importlib.util.find_spec('gear_sonic.utils.inference.harness_rpc'), 'RPC server missing'
    return importlib.import_module('gear_sonic.utils.inference.harness_rpc')


def test_native_rpc_contract():
    from pathlib import Path
    from gear_sonic.utils.inference.harness_contract import decode_request
    fixtures = json.loads((Path(__file__).parent / 'fixtures/g1_rpc_v1.json').read_text())
    for entry in fixtures['valid_requests']:
        assert decode_request(json.dumps(entry).encode()).method == entry['method']
    for entry in fixtures['invalid_requests']:
        with pytest.raises(ValueError):
            decode_request(json.dumps(entry).encode())


def test_local_rpc_and_second_server_does_not_unlink_live_socket(tmp_path):
    module = rpc_module(); c, _ = make()
    endpoint = 'ipc://' + str(tmp_path / 'h.sock')
    server = module.HarnessRPCServer(endpoint, None)
    server.start()
    other = module.HarnessRPCServer(endpoint, None)
    try:
        with pytest.raises((OSError, ValueError)):
            other.start()
        replies = []
        def client():
            with zmq.Context() as context:
                with context.socket(zmq.REQ) as socket:
                    socket.setsockopt(zmq.LINGER, 0)
                    socket.setsockopt(zmq.RCVTIMEO, 1000)
                    socket.connect(endpoint)
                    socket.send_json(dict(version=1, request_id='read', runtime_id=None, session_id=None, lease_id=None, method='get_status', params={}))
                    replies.append(socket.recv_json())
        thread = threading.Thread(target=client); thread.start()
        deadline = time.monotonic() + 1.
        while thread.is_alive() and time.monotonic() < deadline:
            server.drain(c, time.monotonic()); time.sleep(.005)
        thread.join(1.)
        assert replies[0]['runtime_id'] == 'boot'
    finally:
        server.close()
