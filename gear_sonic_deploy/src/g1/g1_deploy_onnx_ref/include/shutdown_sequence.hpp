#ifndef SHUTDOWN_SEQUENCE_HPP
#define SHUTDOWN_SEQUENCE_HPP

#include <exception>
#include <iostream>

/// Stop control first, which joins the loops and sends the damping command, then
/// flush diagnostic output. A trace write failure is reported as exit status 1 but
/// can no longer skip the stop.
template <typename Deploy>
int StopThenFlushDiagnostics(Deploy& deploy) {
  deploy.Stop();
  try {
    deploy.FlushPlannerTrace();
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << std::endl;
    return 1;
  }
  return 0;
}

#endif  // SHUTDOWN_SEQUENCE_HPP
