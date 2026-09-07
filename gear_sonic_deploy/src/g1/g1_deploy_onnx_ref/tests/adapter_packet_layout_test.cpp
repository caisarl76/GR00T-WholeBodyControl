#include "adapter_packet_layout.hpp"

#include <cassert>
#include <cstdint>

struct Field { std::string name, dtype; std::vector<size_t> shape; };
using Bytes = std::vector<std::uint8_t>;

static std::pair<std::vector<Field>, std::vector<Bytes>> Layout(
    size_t n = 3, bool f64 = false, bool quat3d = false) {
    const auto real = f64 ? "f64" : "f32";
    std::vector<Field> f = {
        {"joint_pos", real, {n, 29}}, {"joint_vel", real, {n, 29}},
        {"body_quat", real, quat3d ? std::vector<size_t>{n, 1, 4} : std::vector<size_t>{n, 4}},
        {"frame_index", "i32", {n}}, {"left_hand_joints", real, {n, 7}},
        {"right_hand_joints", real, {n, 7}}, {"adapter_session_id", "i64", {1}},
        {"adapter_chunk_id", "i64", {1}}, {"adapter_deadline_ns", "i64", {1}}};
    std::vector<Bytes> b;
    for (const auto& x : f) {
        size_t count = 1; for (size_t d : x.shape) count *= d;
        const size_t width = x.dtype == "i64" || x.dtype == "f64" ? 8 : 4;
        b.emplace_back(count * width);
    }
    return {f, b};
}

int main() {
    auto [fields, buffers] = Layout();
    assert(ValidateAdapterPacketLayout(fields, buffers));
    buffers[1].pop_back(); assert(!ValidateAdapterPacketLayout(fields, buffers));
    std::tie(fields, buffers) = Layout(); fields[2].shape = {3, 3};
    assert(!ValidateAdapterPacketLayout(fields, buffers));
    std::tie(fields, buffers) = Layout(); fields[0].dtype = "i16";
    assert(!ValidateAdapterPacketLayout(fields, buffers));
    std::tie(fields, buffers) = Layout(); buffers[0].pop_back();
    assert(!ValidateAdapterPacketLayout(fields, buffers));
    std::tie(fields, buffers) = Layout(); fields[1].name = fields[0].name;
    assert(!ValidateAdapterPacketLayout(fields, buffers));
    std::tie(fields, buffers) = Layout(); fields[1].name = "extra";
    assert(!ValidateAdapterPacketLayout(fields, buffers));
    std::tie(fields, buffers) = Layout(); fields[0].shape[0] = 14901;
    assert(!ValidateAdapterPacketLayout(fields, buffers));
    std::tie(fields, buffers) = Layout(2, true, true);
    assert(ValidateAdapterPacketLayout(fields, buffers));
}
