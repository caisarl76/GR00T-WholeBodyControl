#ifndef REFERENCE_HEADING_HANDOFF_HPP
#define REFERENCE_HEADING_HANDOFF_HPP

#include <array>
#include <cmath>
#include "math_utils.hpp"
#include "robot_parameters.hpp"

// Preserve the commanded world yaw when changing the reference's yaw origin.
// The pending pair is guarded by the same mutex as the current reference.
struct ReferenceHeadingHandoff {
    std::array<double, 4> outgoing_root;
    std::array<double, 4> incoming_root;

    HeadingState Rebase(HeadingState heading, std::array<double, 4>& initial_reference) const {
        const auto offset = quat_mul_d(calc_heading_quat_d(outgoing_root),
                                       calc_heading_quat_inv_d(initial_reference));
        heading.delta_heading += 2.0 * std::atan2(offset[3], offset[0]);
        initial_reference = incoming_root;
        return heading;
    }
};

#endif
