#!/usr/bin/env python3
"""Inspect atomic optical hand samples; raw joint 0 is palm and 1 is wrist."""

import time

import xrobotoolkit_sdk as xrt


def main():
    xrt.init()
    try:
        while True:
            for side, getter in (
                ("left", xrt.get_left_hand_snapshot),
                ("right", xrt.get_right_hand_snapshot),
            ):
                sample = getter()
                if sample["binding_generation"] == 0:
                    print(f"{side}: no complete optical hand sample (timestamp/flags required)")
                    continue
                print(
                    f"{side}: active={sample['is_active']} "
                    f"timestamp_ns={sample['source_timestamp_ns']} "
                    f"generation={sample['binding_generation']} "
                    f"wrist={sample['pose'][1, :3]} "
                    f"wrist_flags={sample['location_flags'][1]}"
                )
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        xrt.close()


if __name__ == "__main__":
    main()
