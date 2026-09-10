"""Check the table evaluation preset without starting the evaluation stack."""

from pathlib import Path
import runpy
import shlex
import sys
from unittest.mock import patch


def test_table_preset_dry_run(capsys):
    root = Path(__file__).resolve().parents[2]
    script = root / "gear_sonic/scripts/launch_pnp_table_eval.py"
    for argv, port, prompt, dataset_prefix, instructions in (
        ([str(script), "--dry-run"], "15554", "approach the table", "pnp_table_260908_subtask_eval_", True),
        (
            [str(script), "full", "--dry-run"],
            "15555",
            "approach the table and pick the bottle",
            "pnp_table_260908_full_task_eval_",
            False,
        ),
        (
            [str(script), "full", "--policy-port", "15600", "--dry-run"],
            "15600",
            "approach the table and pick the bottle",
            "pnp_table_260908_full_task_eval_",
            False,
        ),
    ):
        with (
            patch.object(sys, "argv", argv),
            patch("subprocess.call", side_effect=AssertionError("Dry-run must not launch anything")),
        ):
            runpy.run_path(str(script), run_name="__main__")
        output = capsys.readouterr().out
        command = shlex.split(output.splitlines()[-1])
        assert command[command.index("--policy-port") + 1] == port
        assert command[command.index("--prompt") + 1] == prompt
        assert command[command.index("--dataset-name") + 1].startswith(dataset_prefix)
        assert "--no-deploy" in command
        for flag, value in {
            "--policy-host": "192.168.75.173",
            "--camera-host": "192.168.0.223",
            "--camera-port": "5555",
            "--state-zmq-host": "192.168.0.223",
            "--action-zmq-host": "192.168.0.62",
            "--embodiment-tag": "unitree_g1_sonic",
            "--action-horizon": "40",
            "--action-publish-rate": "50",
            "--task-prompt": "approach the table and pick the bottle",
        }.items():
            assert command[command.index(flag) + 1] == value
        assert ("t grasp the bottle" in output) is instructions
        assert ("t pick up the bottle" in output) is instructions
