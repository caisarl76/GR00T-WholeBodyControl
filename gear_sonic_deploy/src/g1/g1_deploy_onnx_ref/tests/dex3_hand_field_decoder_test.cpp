#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include "input_interface/dex3_hand_field_decoder.hpp"

namespace {

using gear_sonic::deploy::DecodeDex3HandField;
using gear_sonic::deploy::HandFieldDecodeStatus;

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
  }
  return condition;
}

}  // namespace

int main() {
  const std::array<float, 6> inspire_values{0.0F, 0.1F, 0.2F, 0.3F, 0.4F, 0.5F};

  const auto six_value_result = DecodeDex3HandField(
      "f32", std::vector<std::size_t>{6}, inspire_values.data(),
      sizeof(inspire_values), false);
  if (!Expect(six_value_result.status == HandFieldDecodeStatus::kShapeMismatch,
              "six-value Inspire field must be ignored") ||
      !Expect(!six_value_result.values.has_value(),
              "six-value Inspire field must not produce a Dex3 command")) {
    return 1;
  }

  const auto short_buffer_result = DecodeDex3HandField(
      "f32", std::vector<std::size_t>{7}, inspire_values.data(),
      sizeof(inspire_values), false);
  if (!Expect(short_buffer_result.status == HandFieldDecodeStatus::kBufferTooSmall,
              "declared seven-value field must validate its byte count") ||
      !Expect(!short_buffer_result.values.has_value(),
              "short buffer must not produce a Dex3 command")) {
    return 1;
  }

  const auto unsupported_dtype_result = DecodeDex3HandField(
      "i32", std::vector<std::size_t>{7}, inspire_values.data(),
      sizeof(inspire_values), false);
  if (!Expect(
          unsupported_dtype_result.status == HandFieldDecodeStatus::kUnsupportedDtype,
          "Dex3 field must reject non-floating dtype") ||
      !Expect(!unsupported_dtype_result.values.has_value(),
              "unsupported dtype must not produce a Dex3 command")) {
    return 1;
  }

  const std::array<float, 7> dex3_values{0.0F, 0.1F, 0.2F, 0.3F,
                                         0.4F, 0.5F, 0.6F};
  const auto short_chunk_result = DecodeDex3HandField(
      "f32", std::vector<std::size_t>{2, 7}, dex3_values.data(),
      sizeof(dex3_values), false);
  if (!Expect(short_chunk_result.status == HandFieldDecodeStatus::kBufferTooSmall,
              "chunked Dex3 field must validate the full declared byte count") ||
      !Expect(!short_chunk_result.values.has_value(),
              "short chunk must not produce a Dex3 command")) {
    return 1;
  }

  const auto valid_result = DecodeDex3HandField(
      "f32", std::vector<std::size_t>{7}, dex3_values.data(),
      sizeof(dex3_values), false);
  if (!Expect(valid_result.status == HandFieldDecodeStatus::kDecoded,
              "valid Dex3 field must decode") ||
      !Expect(valid_result.values.has_value(),
              "valid Dex3 field must return values")) {
    return 1;
  }
  for (std::size_t index = 0; index < dex3_values.size(); ++index) {
    if (!Expect(std::abs(valid_result.values.value()[index] - dex3_values[index]) < 1e-12,
                "decoded Dex3 value differs from input")) {
      return 1;
    }
  }

  return 0;
}
