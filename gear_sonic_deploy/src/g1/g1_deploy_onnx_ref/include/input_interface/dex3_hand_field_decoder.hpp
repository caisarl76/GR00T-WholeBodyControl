#ifndef DEX3_HAND_FIELD_DECODER_HPP
#define DEX3_HAND_FIELD_DECODER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string_view>
#include <vector>

namespace gear_sonic::deploy {

enum class HandFieldDecodeStatus {
  kDecoded,
  kShapeMismatch,
  kUnsupportedDtype,
  kBufferTooSmall,
  kNullData,
};

struct HandFieldDecodeResult {
  HandFieldDecodeStatus status;
  std::optional<std::array<double, 7>> values;
};

namespace detail {

template <typename T>
T ByteSwap(T value) {
  std::array<std::uint8_t, sizeof(T)> source{};
  std::array<std::uint8_t, sizeof(T)> destination{};
  std::memcpy(source.data(), &value, sizeof(T));
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    destination[index] = source[sizeof(T) - 1 - index];
  }
  std::memcpy(&value, destination.data(), sizeof(T));
  return value;
}

template <typename T>
std::array<double, 7> DecodeValues(const void* data, bool needs_byte_swap) {
  std::array<double, 7> decoded{};
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  for (std::size_t index = 0; index < decoded.size(); ++index) {
    T value{};
    std::memcpy(&value, bytes + index * sizeof(T), sizeof(T));
    if (needs_byte_swap) {
      value = ByteSwap(value);
    }
    decoded[index] = static_cast<double>(value);
  }
  return decoded;
}

}  // namespace detail

inline HandFieldDecodeResult DecodeDex3HandField(
    std::string_view dtype, const std::vector<std::size_t>& shape,
    const void* data, std::size_t buffer_size, bool needs_byte_swap) {
  const bool valid_shape =
      (shape.size() == 1 && shape[0] == 7) ||
      (shape.size() == 2 && shape[0] > 0 && shape[1] == 7);
  if (!valid_shape) {
    return {HandFieldDecodeStatus::kShapeMismatch, std::nullopt};
  }

  std::size_t element_size = 0;
  if (dtype == "f32") {
    element_size = sizeof(float);
  } else if (dtype == "f64") {
    element_size = sizeof(double);
  } else {
    return {HandFieldDecodeStatus::kUnsupportedDtype, std::nullopt};
  }

  std::size_t element_count = 1;
  for (const std::size_t dimension : shape) {
    if (element_count > std::numeric_limits<std::size_t>::max() / dimension) {
      return {HandFieldDecodeStatus::kBufferTooSmall, std::nullopt};
    }
    element_count *= dimension;
  }
  if (element_count > std::numeric_limits<std::size_t>::max() / element_size ||
      buffer_size < element_count * element_size) {
    return {HandFieldDecodeStatus::kBufferTooSmall, std::nullopt};
  }
  if (data == nullptr) {
    return {HandFieldDecodeStatus::kNullData, std::nullopt};
  }

  if (dtype == "f32") {
    return {HandFieldDecodeStatus::kDecoded,
            detail::DecodeValues<float>(data, needs_byte_swap)};
  }
  return {HandFieldDecodeStatus::kDecoded,
          detail::DecodeValues<double>(data, needs_byte_swap)};
}

}  // namespace gear_sonic::deploy

#endif  // DEX3_HAND_FIELD_DECODER_HPP
