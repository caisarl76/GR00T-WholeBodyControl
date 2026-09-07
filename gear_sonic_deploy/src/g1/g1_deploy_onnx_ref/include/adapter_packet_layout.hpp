#ifndef ADAPTER_PACKET_LAYOUT_HPP
#define ADAPTER_PACKET_LAYOUT_HPP

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

template <typename Fields, typename Buffers>
bool ValidateAdapterPacketLayout(const Fields& fields, const Buffers& buffers) {
    if (fields.size() != 9 || buffers.size() != fields.size()) return false;
    std::vector<std::string> names;
    size_t frames = 0;
    for (const auto& f : fields) {
        if (std::find(names.begin(), names.end(), f.name) != names.end()) return false;
        names.push_back(f.name);
        if (f.name == "joint_pos") {
            if (f.shape.size() != 2 || f.shape[0] < 1 || f.shape[0] > 14900 || f.shape[1] != 29) return false;
            frames = f.shape[0];
        }
    }
    if (!frames) return false;
    for (size_t i = 0; i < fields.size(); ++i) {
        const auto& f = fields[i];
        size_t count = 0, width = 0;
        if (f.name == "adapter_session_id" || f.name == "adapter_chunk_id" || f.name == "adapter_deadline_ns") {
            if (f.shape != std::vector<size_t>{1} || f.dtype != "i64") return false;
            count = 1; width = 8;
        } else if (f.name == "frame_index") {
            if (f.shape != std::vector<size_t>{frames}) return false;
            count = frames;
            width = f.dtype == "i64" ? 8 : f.dtype == "i32" ? 4 : 0;
        } else {
            size_t channels = 0;
            if (f.name == "joint_pos" || f.name == "joint_vel") channels = 29;
            else if (f.name == "left_hand_joints" || f.name == "right_hand_joints") channels = 7;
            else if (f.name == "body_quat") channels = 4;
            else return false;
            if (f.shape != std::vector<size_t>{frames, channels} &&
                !(f.name == "body_quat" && f.shape == std::vector<size_t>{frames, 1, 4})) return false;
            count = frames * channels;
            width = f.dtype == "f32" ? 4 : f.dtype == "f64" ? 8 : 0;
        }
        if (!width || buffers[i].size() != count * width) return false;
    }
    return true;
}

#endif
