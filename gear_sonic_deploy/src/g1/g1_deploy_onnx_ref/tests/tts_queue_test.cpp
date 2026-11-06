#include <iostream>
#include <string>

#include "tts_queue.hpp"

using gear_sonic::deploy::AppendTts;

int main() {
  // High-temp warning and the manual (F) temperature report in the same tick.
  std::string pending;
  AppendTts(pending, "Warning! High temperature at Left Ankle Roll(88)");
  AppendTts(pending, "Hottest motors: Left Ankle Roll 88");
  const std::string expected =
      "Warning! High temperature at Left Ankle Roll(88). Hottest motors: Left Ankle Roll 88";
  if (pending != expected) {
    std::cerr << "overlap lost speech: '" << pending << "'\n";
    return 1;
  }
  AppendTts(pending, "");
  if (pending != expected) {
    std::cerr << "empty message changed pending speech\n";
    return 1;
  }
  return 0;
}
