"""PICO button chords and nonblocking manager-terminal T/C/S input."""

import math
import os
import select
import sys
import termios
import tty


def read_controllers(sdk):
    """Read each atomic controller once; absent/stale controllers are neutral."""
    neutral = {
        "primary_button": False,
        "secondary_button": False,
        "grip": 0.0,
        "trigger": 0.0,
        "axis": (0.0, 0.0),
        "axis_click": False,
        "menu_button": False,
    }
    snapshots, healthy = [], []
    for side in ("left", "right"):
        try:
            sample = getattr(sdk, f"get_{side}_controller_snapshot")()
            generation, age = sample["binding_generation"], sample["receipt_age_ns"]
            valid = (
                type(generation) is int
                and generation > 0
                and type(age) is int
                and 0 <= age < 100_000_000
                and all(
                    type(sample[k]) is bool
                    for k in ("primary_button", "secondary_button", "axis_click", "menu_button")
                )
                and len(sample["axis"]) == 2
                and all(math.isfinite(v) for v in (*sample["axis"], sample["grip"], sample["trigger"]))
            )
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError, OverflowError):
            sample, valid = neutral, False
        snapshots.append(sample if valid else neutral)
        healthy.append(valid)
    left, right = snapshots
    a, b, x, y = (
        right["primary_button"],
        right["secondary_button"],
        left["primary_button"],
        left["secondary_button"],
    )
    return {
        "buttons": {"a": a, "b": b, "x": x, "y": y, "grip": left["grip"] > 0.5},
        "abxy": (a, b, x, y),
        "fresh": all(healthy),
        "inputs": (left["menu_button"], left["trigger"], right["trigger"], left["grip"], right["grip"]),
        "axes": (*left["axis"], *right["axis"]),
        "clicks": (left["axis_click"], right["axis_click"]),
    }


class ControllerChords:
    """Timed debounce followed by one priority action and mandatory release."""

    BUTTONS = ("a", "b", "x", "y", "grip")
    CHORDS = (
        (frozenset("abxy"), "sonic"),
        (frozenset(("grip", "b")), "abort"),
        (frozenset(("grip", "a")), "record"),
        (frozenset("ax"), "tracking"),
        (frozenset("by"), "freeze"),
        (frozenset("ab"), "locomotion_next"),
        (frozenset("xy"), "locomotion_previous"),
    )

    def __init__(self):
        self.raw = dict.fromkeys(self.BUTTONS, False)
        self.stable = self.raw.copy()
        self.changed_at = dict.fromkeys(self.BUTTONS, 0)
        self.wait_release = True
        self.neutral_since = None
        self.deadline = None

    def poll(self, buttons, now_ns, *, fresh=True):
        if not fresh:
            self.wait_release = True
            self.neutral_since = self.deadline = None
            self.raw = dict.fromkeys(self.BUTTONS, False)
            self.stable = self.raw.copy()
            self.changed_at = dict.fromkeys(self.BUTTONS, now_ns)
            return None
        raw = {key: bool(buttons.get(key, False)) for key in self.BUTTONS}
        if self.wait_release:
            if any(raw.values()):
                self.neutral_since = None
            elif self.neutral_since is None:
                self.neutral_since = now_ns
            elif now_ns - self.neutral_since >= 40_000_000:
                self.wait_release = False
                self.raw = raw
                self.stable = raw.copy()
                self.changed_at = dict.fromkeys(self.BUTTONS, now_ns)
            return None
        for key, value in raw.items():
            if self.raw[key] != value:
                self.raw[key] = value
                self.changed_at[key] = now_ns
            if now_ns - self.changed_at[key] >= 40_000_000:
                self.stable[key] = value
        down = {key for key, value in self.stable.items() if value}
        if self.deadline is None and down:
            self.deadline = now_ns + 200_000_000
        if self.deadline is None:
            return None
        candidate = next((action for chord, action in self.CHORDS if chord <= down), None)
        if candidate == "sonic" or now_ns >= self.deadline:
            self.wait_release = True
            self.neutral_since = self.deadline = None
            return candidate
        return None


class ManagerKeyboard:
    """Read available terminal characters without intercepting SIGINT."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stdin
        self.saved = None
        self.fd = None

    def open(self):
        if self.stream.isatty():
            self.fd = self.stream.fileno()
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)

    def poll(self):
        if self.fd is None or not select.select([self.fd], [], [], 0)[0]:
            return []
        # C/S are state-setting commands; repeated characters are idempotent.
        return [c for c in os.read(self.fd, 128).decode(errors="ignore").lower() if c in "tcs"]

    def close(self):
        if self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
            self.saved = None
