#include "../include/adapter_reference_schedule.hpp"
#include <cassert>
#include <cmath>
#include <bit>

int main() {
    std::vector<AdapterReferenceFrame> rows(2);
    std::vector<int64_t> indices{10, 11};
    for (int i = 0; i < 2; ++i) {
        rows[i].valid = true; rows[i].session_id = 1; rows[i].chunk_id = 2;
        rows[i].reference_frame = indices[i]; rows[i].expires_at_ns = 99;
    }
    assert(ValidateAdapterReferenceFrames(rows, 2, indices));
    assert(ValidateAdapterReferenceFrames({}, 0, {}));
    rows[1].reference_frame = 13;
    assert(!ValidateAdapterReferenceFrames(rows, 2, indices));
    rows[1].reference_frame = 11;
    rows[0].left[0] = std::bit_cast<double>(uint64_t{0x7ff8000000000001ULL});
    assert(!ValidateAdapterReferenceFrames(rows, 2, indices));
}
