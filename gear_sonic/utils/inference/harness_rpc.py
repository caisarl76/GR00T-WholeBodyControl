"""Local IPC worker. Control dispatch always runs in the existing action loop."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import fcntl
import json
import os
from pathlib import Path
import queue
import threading
import time

import zmq

from .harness_contract import decode_request


@dataclass
class PendingRequest:
    request: dict
    expires_at: float
    done: threading.Event = field(default_factory=threading.Event)
    reply: dict | None = None


class HarnessRPCServer:
    def __init__(self, endpoint, snapshots):
        if not endpoint.startswith('ipc:///') or '\x00' in endpoint or endpoint.startswith('ipc://@'):
            raise ValueError('Only absolute local IPC endpoints are supported')
        self.endpoint, self.snapshots = endpoint, snapshots
        self.path = Path(endpoint[6:])
        self.pending = queue.Queue(maxsize=16)
        self._stop, self._ready = threading.Event(), threading.Event()
        self._thread, self._error = None, None
        self.runtime_id = ''

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(1.):
            self.close()
            raise OSError('IPC worker startup timed out')
        if self._error:
            raise OSError(str(self._error))

    def _run(self):
        lock = None
        bound = False
        try:
            lock = open(str(self.path) + '.lock', 'a')
            os.chmod(lock.name, 0o600)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.path.exists():
                raise OSError('IPC path already exists; verify the previous executor stopped before removing its stale socket')
            with zmq.Context() as context:
                with context.socket(zmq.REP) as socket:
                    socket.setsockopt(zmq.LINGER, 0)
                    socket.setsockopt(zmq.MAXMSGSIZE, 65536)
                    socket.bind(self.endpoint)
                    bound = True
                    os.chmod(self.path, 0o600)
                    self._ready.set()
                    while not self._stop.is_set():
                        if not socket.poll(50):
                            continue
                        raw = socket.recv()
                        try:
                            req = asdict(decode_request(raw))
                            item = PendingRequest(req, time.monotonic() + .5)
                            self.pending.put_nowait(item)
                            while not item.done.wait(.02) and not self._stop.is_set() and time.monotonic() < item.expires_at:
                                pass
                            reply = item.reply or self._error_reply(req['request_id'], 'UNAVAILABLE', 'Action loop did not respond before deadline')
                        except (ValueError, queue.Full) as exc:
                            reply = self._error_reply('', 'INVALID_REQUEST', str(exc))
                        socket.send(json.dumps(reply, allow_nan=False).encode())
        except Exception as exc:
            self._error = exc
            self._ready.set()
        finally:
            if bound:
                self.path.unlink(missing_ok=True)
            if lock:
                lock.close()

    def _error_reply(self, request_id, code, message):
        return dict(request_id=request_id, runtime_id=self.runtime_id, result=None, error=dict(code=code, message=message))

    def drain(self, control, now, max_requests=1):
        self.runtime_id = control.runtime_id
        if self.snapshots is not None:
            control.observation = self.snapshots.latest(now)
        for _ in range(max_requests):
            try:
                item = self.pending.get_nowait()
            except queue.Empty:
                break
            if now > item.expires_at:
                item.reply = self._error_reply(item.request['request_id'], 'UNAVAILABLE', 'Expired queued request')
            else:
                item.reply = control.dispatch(item.request, now)
            item.done.set()

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(1.)
