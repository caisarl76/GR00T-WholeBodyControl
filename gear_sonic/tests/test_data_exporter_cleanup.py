"""Focused tests for shutdown reporting without importing the data stack."""

import ast
from pathlib import Path


def _save_and_cleanup():
    source = Path(__file__).parents[1] / "scripts" / "run_data_exporter.py"
    tree = ast.parse(source.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "save_and_cleanup"
    )
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["save_and_cleanup"]


class _Exporter:
    def __init__(self, size, save_error=None):
        self.episode_buffer = {"size": size}
        self.meta = type("Meta", (), {"root": "/tmp/dataset"})()
        self.saved = 0
        self.save_error = save_error

    def save_episode(self):
        if self.save_error:
            raise self.save_error
        self.saved += 1


class _Collector:
    save_and_cleanup = _save_and_cleanup()

    def __init__(self, exporter):
        self.data_exporter = exporter
        self.messages = []
        self._state_subscriber = _Closable()
        self._sonic_zmq_socket = None
        self._sonic_zmq_ctx = None

    def _print_and_say(self, message, **kwargs):
        self.messages.append(message)


class _Closable:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_empty_buffer_reports_no_new_episode_and_cleans_up():
    exporter = _Exporter(0)
    collector = _Collector(exporter)

    collector.save_and_cleanup()

    assert exporter.saved == 0
    assert any("No new episode saved" in message for message in collector.messages)
    assert any("Existing saved episodes preserved" in message for message in collector.messages)
    assert collector._state_subscriber.closed


def test_nonempty_buffer_reports_saved_episode_after_success():
    exporter = _Exporter(3)
    collector = _Collector(exporter)

    collector.save_and_cleanup()

    assert exporter.saved == 1
    assert collector.messages[:2] == ["Saving episode", "Saved episode. Recording complete: /tmp/dataset"]
    assert collector._state_subscriber.closed


def test_save_failure_reports_error_without_recording_complete():
    exporter = _Exporter(1, RuntimeError("disk full"))
    collector = _Collector(exporter)

    collector.save_and_cleanup()

    assert collector.messages[:2] == ["Saving episode", "Error saving episode: disk full"]
    assert not any("Recording complete" in message for message in collector.messages)
    assert collector._state_subscriber.closed
