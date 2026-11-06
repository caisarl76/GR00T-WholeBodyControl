#pragma once

#include <string>

namespace gear_sonic::deploy {

// Queue one-shot speech without dropping what is already pending this tick.
inline void AppendTts(std::string& pending, const std::string& message) {
  if (message.empty()) { return; }
  if (!pending.empty()) { pending += ". "; }
  pending += message;
}

}  // namespace gear_sonic::deploy
