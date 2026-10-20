"""Four-pane workstation presets for the left-only checkpoint-20000 models.

SONIC v1.1 and the camera run separately on PC2. This script starts the
workstation client, keyboard publisher, and recorder only when executed.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("full", "subtasks"))
    parser.add_argument(
        "--object",
        choices=("apple", "cup", "pill bottle", "pill box", "red bottle"),
        default="pill bottle",
        help="Object to evaluate (default: pill bottle). Uses the dataset's exact labels.",
    )
    parser.add_argument("--policy-host", default="192.168.75.173")
    parser.add_argument("--robot-host", default="192.168.0.223", help="PC2 camera and SONIC state host.")
    parser.add_argument("--action-host", default="192.168.0.62", help="Workstation IP that PC2 subscribes to.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without launching anything.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    full_prompt = (
        f"approach the table, pick the {args.object}, turn left and approach the trash bin, "
        "put it in to the trash bin"
    )
    prompt = full_prompt if args.variant == "full" else "approach brown table"
    port = 15552 if args.variant == "full" else 15553
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dataset_name = f"pnp_trash_left_eval_{args.variant}_{args.object.replace(' ', '_')}_{stamp}"
    command = [
        sys.executable,
        str(repo_root / "gear_sonic/scripts/launch_inference.py"),
        "--no-deploy",
        "--policy-host",
        args.policy_host,
        "--policy-port",
        str(port),
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
        prompt,
        "--task-prompt",
        full_prompt,
        "--dataset-name",
        dataset_name,
    ]

    print(f"Left-only {args.variant}: checkpoint-20000 at {args.policy_host}:{port}")
    print(f"PC2 {args.robot_host}: SONIC v1.1 must subscribe to {args.action_host}:5556.")
    print(f"Initial prompt: {prompt}")
    print(f"Recording destination: {repo_root / 'outputs' / dataset_name}")
    if args.variant == "subtasks":
        print("In the keyboard pane, change prompts as each real subtask completes:")
        for text in (
            f"pick the {args.object}",
            "turn left and approach the trash bin",
            "put it in to the trash bin",
        ):
            print(f"  t {text}")
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        raise SystemExit(subprocess.call(command, cwd=repo_root))


if __name__ == "__main__":
    main()
