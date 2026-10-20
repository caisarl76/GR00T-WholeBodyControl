"""Check preset selection through the CLI without starting the evaluation stack."""

from pathlib import Path
import runpy
import shlex
import sys
from unittest.mock import patch


def test_left_only_presets_use_matching_servers_and_dataset_prompts(capsys):
    root = Path(__file__).resolve().parents[2]
    script = root / "gear_sonic/scripts/launch_pnp_trash_left_eval.py"
    for variant, port, prompt in (
        (
            "full",
            "15552",
            "approach the table, pick the cup, turn left and approach the trash bin, put it in to the trash bin",
        ),
        ("subtasks", "15553", "approach brown table"),
    ):
        with (
            patch.object(sys, "argv", [str(script), variant, "--object", "cup", "--dry-run"]),
            patch("subprocess.call", side_effect=AssertionError("Dry-run must not launch anything")),
        ):
            runpy.run_path(str(script), run_name="__main__")
        output = capsys.readouterr().out
        command = shlex.split(output.splitlines()[-1])
        assert "--no-deploy" in command
        for flag, value in {
            "--policy-host": "192.168.75.173",
            "--policy-port": port,
            "--state-zmq-host": "192.168.0.223",
            "--camera-host": "192.168.0.223",
            "--action-zmq-host": "192.168.0.62",
            "--action-horizon": "40",
            "--action-publish-rate": "50",
            "--prompt": prompt,
        }.items():
            assert command[command.index(flag) + 1] == value
        if variant == "subtasks":
            assert "t pick the cup" in output
            assert "t turn left and approach the trash bin" in output
        assert command[command.index("--dataset-name") + 1].startswith(f"pnp_trash_left_eval_{variant}_cup_")
