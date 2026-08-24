import pytest

from gear_sonic.camera import composed_camera


def test_run_composed_camera_server_closes_on_keyboard_interrupt(monkeypatch, capsys):
    sensors = []

    class FakeComposedCameraSensor:
        def __init__(self, config):
            self.config = config
            self.close_calls = 0
            sensors.append(self)

        def run_server(self):
            raise KeyboardInterrupt

        def close(self):
            self.close_calls += 1

    monkeypatch.setattr(composed_camera, "ComposedCameraSensor", FakeComposedCameraSensor)
    config = composed_camera.ComposedCameraConfig(ego_view_camera=None)

    composed_camera.run_composed_camera_server(config)

    assert len(sensors) == 1
    assert sensors[0].close_calls == 1
    assert capsys.readouterr().out.splitlines() == [
        "Running composed camera server...",
        "Stopping composed camera server...",
        "Composed camera server stopped.",
    ]


def test_run_composed_camera_server_handles_keyboard_interrupt_during_construction(
    monkeypatch, capsys
):
    sensors = []

    class FakeComposedCameraSensor:
        def __init__(self, config):
            self.config = config
            self.close_calls = 0
            sensors.append(self)
            self.close()
            raise KeyboardInterrupt

        def close(self):
            self.close_calls += 1

    monkeypatch.setattr(composed_camera, "ComposedCameraSensor", FakeComposedCameraSensor)
    config = composed_camera.ComposedCameraConfig(ego_view_camera=None)

    try:
        composed_camera.run_composed_camera_server(config)
    except KeyboardInterrupt:
        pytest.fail("construction KeyboardInterrupt propagated")

    assert len(sensors) == 1
    assert sensors[0].close_calls == 1
    assert capsys.readouterr().out.splitlines() == [
        "Running composed camera server...",
        "Stopping composed camera server...",
        "Composed camera server stopped.",
    ]


def test_run_composed_camera_server_closes_and_propagates_unexpected_error(monkeypatch):
    sensors = []

    class FakeComposedCameraSensor:
        def __init__(self, config):
            self.config = config
            self.close_calls = 0
            sensors.append(self)

        def run_server(self):
            raise RuntimeError("server failed")

        def close(self):
            self.close_calls += 1

    monkeypatch.setattr(composed_camera, "ComposedCameraSensor", FakeComposedCameraSensor)
    config = composed_camera.ComposedCameraConfig(ego_view_camera=None)

    with pytest.raises(RuntimeError, match="server failed"):
        composed_camera.run_composed_camera_server(config)

    assert len(sensors) == 1
    assert sensors[0].close_calls == 1


@pytest.mark.parametrize(
    ("failure", "socket_created"),
    [(KeyboardInterrupt(), False), (RuntimeError("bind failed"), True)],
    ids=["keyboard-interrupt", "bind-error"],
)
def test_composed_camera_sensor_cleans_partial_server_on_base_exception(
    monkeypatch, failure, socket_created
):
    class FakeSocket:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    class FakeContext:
        def __init__(self):
            self.term_calls = 0

        def term(self):
            self.term_calls += 1

    socket = FakeSocket()
    context = FakeContext()
    sensors = []

    def interrupting_start_server(self, port):
        sensors.append(self)
        self.context = context
        if socket_created:
            self.socket = socket
        raise failure

    monkeypatch.setattr(
        composed_camera.ComposedCameraSensor,
        "_wait_for_all_cameras_ready",
        lambda self, timeout: None,
    )
    monkeypatch.setattr(
        composed_camera.ComposedCameraSensor, "start_server", interrupting_start_server
    )
    config = composed_camera.ComposedCameraConfig(ego_view_camera=None)

    with pytest.raises(type(failure)) as raised:
        composed_camera.ComposedCameraSensor(config)

    assert raised.value is failure
    assert len(sensors) == 1
    assert socket.close_calls == int(socket_created)
    assert context.term_calls == 1

    sensors[0].close()
    assert socket.close_calls == int(socket_created)
    assert context.term_calls == 1


def test_composed_camera_sensor_cleans_worker_when_thread_start_is_interrupted(monkeypatch):
    threads = []

    class FakeThread:
        def __init__(self, *, target, args):
            self.sensor = target.__self__
            self.args = args
            self.alive = False
            self.join_calls = 0
            threads.append(self)

        def start(self):
            self.alive = True
            raise KeyboardInterrupt

        def is_alive(self):
            return self.alive

        def join(self, timeout):
            self.join_calls += 1
            self.alive = False

    monkeypatch.setattr(composed_camera.threading, "Thread", FakeThread)
    config = composed_camera.ComposedCameraConfig(ego_view_camera="usb", server=False)

    with pytest.raises(KeyboardInterrupt):
        composed_camera.ComposedCameraSensor(config)

    assert len(threads) == 1
    assert threads[0].args[4].is_set()
    assert threads[0].join_calls == 1

    threads[0].sensor.close()
    assert threads[0].join_calls == 1
