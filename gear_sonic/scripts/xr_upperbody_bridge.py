"""CLI entry point for the GEAR-SONIC XR upper-body bridge."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.teleop.xr_upperbody_bridge import main


if __name__ == "__main__":
    raise SystemExit(main())
