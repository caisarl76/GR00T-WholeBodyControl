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
    assert "Composed camera server stopped." in capsys.readouterr().out


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
