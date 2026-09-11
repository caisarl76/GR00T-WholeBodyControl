#pragma once

#include <algorithm>

// Called only for a playing, nonempty streamed motion. Live playback skips
// queued history while retaining the observation window; neither mode rewinds.
inline int SelectStreamedPlaybackFrame(int current_frame, int timesteps,
                                      int observation_window,
                                      bool live_pose_playback = false) {
  if (live_pose_playback) {
    return std::max(current_frame,
                    std::max(0, timesteps - observation_window - 1));
  }
  return current_frame + 1 >= timesteps - observation_window
             ? current_frame
             : current_frame + 1;
}
