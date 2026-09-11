"""Exercise the production cursor helper without robot, CUDA, or DDS dependencies."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_streamed_playback_preserves_legacy_and_bounds_live_backlog(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the standalone playback regression")
    include_dir = Path(__file__).resolve().parents[2] / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include"
    source = tmp_path / "streamed_playback.cpp"
    source.write_text(
        r"""
#include "streamed_motion_playback.hpp"
#include <cassert>

int main() {
  // Match the old increment/hold branch, including short windows and a
  // cursor already beyond the safe horizon. The default is sequential.
  for (int timesteps = 1; timesteps <= 100; ++timesteps) {
    for (int window : {0, 1, 5, 9, 101}) {
      for (int frame = 0; frame <= timesteps + 3; ++frame) {
        int legacy = frame + 1;
        if (legacy >= timesteps - window) --legacy;
        assert(SelectStreamedPlaybackFrame(frame, timesteps, window) == legacy);
        assert(SelectStreamedPlaybackFrame(frame, timesteps, window, false) == legacy);
        int live = SelectStreamedPlaybackFrame(frame, timesteps, window, true);
        assert(live >= frame);
        if (frame <= timesteps - window - 1) {
          assert(live + window < timesteps);
          assert(live == timesteps - window - 1);
        } else {
          assert(live == frame);
        }
      }
    }
  }

  // A 60 Hz producer against a 50 Hz consumer accumulates 100 extra frames
  // in ten seconds. Live playback keeps only the required future horizon.
  constexpr int window = 5;
  int timesteps = window + 1;
  int sequential = 0;
  int live = 0;
  for (int tick = 0; tick < 500; ++tick) {
    timesteps += 1 + (tick % 5 == 0);
    sequential = SelectStreamedPlaybackFrame(sequential, timesteps, window);
    int previous_live = live;
    live = SelectStreamedPlaybackFrame(live, timesteps, window, true);
    assert(live >= previous_live);
    assert(timesteps - 1 - live == window);
    assert(live + window < timesteps);
  }
  assert(timesteps - 1 - sequential == window + 100);

  // A stalled producer holds the cursor. A larger observation horizon or
  // shortened buffer must not rewind a cursor already beyond the safe edge.
  assert(SelectStreamedPlaybackFrame(live, timesteps, window, true) == live);
  assert(SelectStreamedPlaybackFrame(live, timesteps, window + 20, true) == live);
  assert(SelectStreamedPlaybackFrame(live, timesteps - 20, window, true) == live);
  assert(SelectStreamedPlaybackFrame(0, 1, window, true) == 0);
}
"""
    )
    executable = tmp_path / "streamed_playback"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(include_dir),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
