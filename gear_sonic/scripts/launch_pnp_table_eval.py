"""Four-pane workstation preset for the completed pnp table evaluation model."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("subtask", "full"), nargs="?", default="subtask")
    parser.add_argument("--policy-host", default="192.168.75.173")
    parser.add_argument("--policy-port", type=int, default=None)
    parser.add_argument("--robot-host", default="192.168.0.223", help="PC2 camera and SONIC state host.")
    parser.add_argument("--action-host", default="192.168.0.62", help="Workstation IP that PC2 subscribes to.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without launching anything.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    policy_port = (
        args.policy_port if args.policy_port is not None else (15555 if args.variant == "full" else 15554)
    )
    initial_prompt = (
        "approach the table" if args.variant == "subtask" else "approach the table and pick the bottle"
    )
    task_prompt = "approach the table and pick the bottle"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dataset_variant = "full_task" if args.variant == "full" else "subtask"
    dataset_name = f"pnp_table_260908_{dataset_variant}_eval_{stamp}"
    command = [
        sys.executable,
        str(repo_root / "gear_sonic/scripts/launch_inference.py"),
        "--no-deploy",
        "--policy-host",
        args.policy_host,
        "--policy-port",
        str(policy_port),
        "--camera-host",
        args.robot_host,
        "--camera-port",
        "5555",
        "--state-zmq-host",
        args.robot_host,
        "--action-zmq-host",
        args.action_host,
        "--embodiment-tag",
        "unitree_g1_sonic",
        "--action-horizon",
        "40",
        "--action-publish-rate",
        "50",
        "--prompt",
        initial_prompt,
        "--task-prompt",
        task_prompt,
        "--dataset-name",
        dataset_name,
    ]

    print(f"Policy server: {args.policy_host}:{policy_port}")
    print(f"PC2 {args.robot_host}: SONIC must subscribe to {args.action_host}:5556.")
    print(f"Initial prompt: {initial_prompt}")
    print(f"Task prompt: {task_prompt}")
    if args.variant == "subtask":
        print("Keyboard commands:")
        print("  t grasp the bottle")
        print("  t pick up the bottle")
    print(f"Recording destination: {repo_root / 'outputs' / dataset_name}")
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        raise SystemExit(subprocess.call(command, cwd=repo_root))


if __name__ == "__main__":
    main()
