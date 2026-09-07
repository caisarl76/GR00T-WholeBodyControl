#include "../include/motion_data_reader.hpp"
#include "../include/input_interface/streamed_motion_merger.hpp"
#include <cassert>
#include <cmath>

static StreamedMotionMerger::IncomingData MakeData(int start, int count, int chunk, bool tagged = true) {
    StreamedMotionMerger::IncomingData d;
    d.protocol_version = 1; d.num_frames = count; d.num_joints = 29; d.num_quat_bodies = 1;
    d.joint_pos.resize(count, std::vector<double>(29)); d.joint_vel.resize(count, std::vector<double>(29));
    d.body_quat.resize(count, std::vector<std::array<double, 4>>(1)); d.frame_indices.resize(count);
    if (tagged) d.adapter_reference_frames.resize(count);
    for (int f = 0; f < count; ++f) {
        const int global = start + f; d.frame_indices[f] = global;
        for (int j = 0; j < 29; ++j) { d.joint_pos[f][j] = 1000.0 * chunk + global + j / 100.0; d.joint_vel[f][j] = -d.joint_pos[f][j]; }
        d.body_quat[f][0] = {1.0, 0.0, 0.0, 0.0};
        if (tagged) { auto& row = d.adapter_reference_frames[f]; row.valid = true; row.session_id = 1; row.chunk_id = chunk; row.reference_frame = global; row.expires_at_ns = 1000000000000LL; row.left[0] = global; row.right[0] = -global; }
    }
    return d;
}

static void AssertRow(const MotionSequence& m, int i, int global, int chunk) {
    assert(std::abs(m.JointPositions(i)[0] - (1000.0 * chunk + global)) < 1e-9);
    assert(m.adapter_reference_frames[i].reference_frame == global);
    assert(m.adapter_reference_frames[i].left[0] == global && m.adapter_reference_frames[i].right[0] == -global);
}

int main() {
    StreamedMotionMerger merger;
    auto first = merger.MergeIncomingData(MakeData(0, 100, 1), 0);
    assert(first.motion && first.motion->timesteps == 100);
    for (int i = 0; i < 100; ++i) AssertRow(*first.motion, i, i, 1);
    auto second = merger.MergeIncomingData(MakeData(70, 100, 2), 60);
    assert(second.motion && second.window_start < 70);
    for (int i = 0; i < second.motion->timesteps; ++i) { int global = second.window_start + i; AssertRow(*second.motion, i, global, global < 70 ? 1 : 2); }

    StreamedMotionMerger capped;
    assert(capped.MergeIncomingData(MakeData(0, 1000, 3), 100).motion);
    assert(!capped.MergeIncomingData(MakeData(295, 14900, 4), 100).motion);
    auto followup = capped.MergeIncomingData(MakeData(110, 1000, 5), 100);
    assert(followup.motion && followup.window_start == 95);
    for (int i = 0; i < followup.motion->timesteps; ++i) {
        const int global = followup.window_start + i;
        AssertRow(*followup.motion, i, global, global < 110 ? 3 : 5);
    }

    StreamedMotionMerger legacy;
    auto result = legacy.MergeIncomingData(MakeData(0, 2, 0, false), 0);
    assert(result.motion && result.motion->adapter_reference_frames.empty());
}
