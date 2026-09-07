#ifndef ADAPTER_REFERENCE_SCHEDULE_HPP
#define ADAPTER_REFERENCE_SCHEDULE_HPP

#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <vector>
#include <cstring>

struct AdapterReferenceFrame {
    bool valid = false;
    int64_t session_id = 0;
    int64_t chunk_id = 0;
    int64_t reference_frame = 0;
    int64_t expires_at_ns = 0;
    std::array<double, 7> left{};
    std::array<double, 7> right{};
};

inline bool IsFiniteAdapterValue(double value) {
    uint64_t raw = 0;
    std::memcpy(&raw, &value, sizeof(raw));
    volatile uint64_t bits = raw;
    return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}

inline bool ValidateAdapterReferenceFrames(
    const std::vector<AdapterReferenceFrame>& rows,
    size_t expected_count,
    const std::vector<int64_t>& frame_indices) {
    if (rows.empty()) return true;
    if (expected_count == 0 || rows.size() != expected_count || frame_indices.size() != expected_count) return false;
    const auto& first = rows.front();
    constexpr int64_t max_frame = static_cast<int64_t>(std::numeric_limits<int>::max()) - 15000;
    if (!first.valid || first.session_id <= 0 || first.chunk_id < 0 || first.reference_frame < 0 || first.reference_frame > max_frame || first.expires_at_ns <= 0) return false;
    for (const auto& value : first.left) if (!IsFiniteAdapterValue(value)) return false;
    for (const auto& value : first.right) if (!IsFiniteAdapterValue(value)) return false;
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& row = rows[i];
        if (!row.valid || row.session_id != first.session_id || row.chunk_id != first.chunk_id || row.expires_at_ns != first.expires_at_ns
            || row.reference_frame < 0 || row.reference_frame > max_frame || frame_indices[i] != row.reference_frame
            || (i > 0 && (row.reference_frame != rows[i - 1].reference_frame + 1 || frame_indices[i] != frame_indices[i - 1] + 1))) return false;
        for (const auto& value : row.left) if (!IsFiniteAdapterValue(value)) return false;
        for (const auto& value : row.right) if (!IsFiniteAdapterValue(value)) return false;
    }
    return true;
}

#endif
