import shlex
import subprocess
from unittest.mock import patch

import pytest

from gear_sonic.scripts import launch_inference


def _completed(returncode=0):
    return subprocess.CompletedProcess([], returncode)


def test_remote_mode_builds_module_commands_and_skips_deploy():
    commands = []

    def run(command, *args, **kwargs):
        commands.append(command)
        if command[:2] == ["tmux", "has-session"]:
            return _completed(1)
        if command[:2] == ["tmux", "list-panes"]:
            return subprocess.CompletedProcess(command, 0, stdout="0")
        return _completed()

    config = launch_inference.InferenceLaunchConfig(
        deploy=False,
        state_zmq_host="192.168.0.223",
        action_zmq_host="192.168.0.62",
        prompt="pick O'Reilly's cup",
        task_prompt="pick O'Reilly's cup",
    )
    with (
        patch.object(launch_inference, "_check_prerequisites"),
        patch.object(launch_inference, "_get_local_ip", return_value="localhost"),
        patch.object(launch_inference.time, "sleep"),
        patch.object(launch_inference.subprocess, "run", side_effect=run),
    ):
        launch_inference.main(config)

    sent = [command[4] for command in commands if command[:2] == ["tmux", "send-keys"]]
    assert len(sent) == 4
    assert all("deploy.sh" not in command for command in sent)
    inference = next(command for command in sent if "run_vla_inference" in command)
    exporter = next(command for command in sent if "run_data_exporter" in command)
    assert "python -m gear_sonic.scripts.run_vla_inference" in inference
    assert "--state-zmq-host 192.168.0.223" in inference
    assert "--action-zmq-host 192.168.0.62" in inference
    assert "python -m gear_sonic.scripts.run_data_exporter" in exporter
    assert "--legacy-recording-controls" in exporter
    assert "--state-zmq-host 192.168.0.223" in exporter
    assert "--sonic-zmq-host 192.168.0.62" in exporter
    assert shlex.quote("pick O'Reilly's cup") in inference


def test_existing_session_is_rejected_without_kill():
    calls = []

    def run(command, *args, **kwargs):
        calls.append(command)
        if command[:2] == ["tmux", "has-session"]:
            return _completed(0)
        return _completed()

    with (
        patch.object(launch_inference, "_check_prerequisites"),
        patch.object(launch_inference.subprocess, "run", side_effect=run),
    ):
        with pytest.raises(SystemExit) as exc:
            launch_inference.main(launch_inference.InferenceLaunchConfig(deploy=False))

    assert exc.value.code == 1
    assert calls == [["tmux", "has-session", "-t", launch_inference.SESSION_NAME]]
