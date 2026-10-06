#ifndef PLANNER_TRACE_HPP
#define PLANNER_TRACE_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <nlohmann/json.hpp>

inline double PlannerTraceTime(std::chrono::steady_clock::time_point time =
                               std::chrono::steady_clock::now()) {
  return std::chrono::duration<double>(time.time_since_epoch()).count();
}

inline void ValidatePlannerTraceOptions(const std::string& path,
                                       const std::string& interface,
                                       bool disable_crc,
                                       const std::string& input,
                                       const std::string& planner) {
  if (!path.empty() && (interface != "lo" || !disable_crc ||
                       input != "zmq_manager" || planner.empty())) {
    throw std::invalid_argument("--planner-trace-file requires lo, --disable-crc-check, "
                                "zmq_manager and --planner-file");
  }
}

// Opt-in simulation diagnostic. Each stream has exactly one writer. Storage is
// allocated before controller threads start; Record only copies fixed records.
// Freeze seals all streams and waits for copies already in progress. Only the
// main thread serializes, after the trial. Overflow retains the oldest records.
class PlannerTrace {
 public:
  static constexpr int kRawFrames = 64;
  static constexpr int kMotionFrames = 128;
  using Quat = std::array<double, 4>;  // w,x,y,z

  struct Plan {
    std::uint64_t generation = 0;
    bool initial = false;
    int generation_frame = 0, mode = 0, seed = 0;
    double begin = 0, end = 0, movement_buffer_at = 0;
    double gather_us = 0, inference_us = 0, resample_us = 0;
    double speed = 0, height = 0;
    std::array<double, 3> received_facing{}, facing{}, movement{};
    std::array<double, 144> context{};  // 4 x 36 qpos, planner joint order
    int raw_frames = 0, resampled_frames = 0;
    std::array<double, kRawFrames * 36> raw_qpos{};
    std::array<Quat, kMotionFrames> resampled_quat{};
  };
  struct Merge {
    double at = 0;
    std::uint64_t generation = 0;
    int published_generation_frame = 0, used_generation_frame = 0;
    int previous_frame = 0, frames = 0;
    bool first = false;
    std::array<Quat, kMotionFrames> quat{};
  };
  struct Control {
    std::uint64_t generation = 0;
    int tick = 0, frame = 0, observation_window_frames = 0;
    bool planner_motion = false;
    double begin = 0, observations_end = 0, policy_end = 0, telemetry_end = 0;
    double measurement_at = 0;
    Quat active_quat{}, heading_quat{}, target_quat{}, measured_quat{};
    std::array<double, 29> motor_target{}, motor_measured{};
  };

  explicit PlannerTrace(std::size_t plans = 512, std::size_t merges = 512,
                        std::size_t controls = 8192)
      : plans_(plans), merges_(merges), controls_(controls) {}

  bool Record(const Plan& row) { return RecordInto(plans_, row); }
  bool Record(const Merge& row) { return RecordInto(merges_, row); }
  bool Record(const Control& row) { return RecordInto(controls_, row); }

  void Freeze() {
    sealed_.store(true);
    while (writers_.load() != 0) std::this_thread::yield();
    frozen_.store(true);
  }

  void Write(std::ostream& out) const {
    if (!frozen_.load())
      throw std::logic_error("Freeze planner trace before writing");
    using nlohmann::json;
    out << json{{"event", "metadata"}, {"schema", 1},
                {"clock", "steady_clock_seconds"}, {"quaternion_order", "wxyz"},
                {"plan_count", plans_.count}, {"plan_dropped", plans_.dropped},
                {"plan_capacity", plans_.capacity},
                {"merge_count", merges_.count}, {"merge_dropped", merges_.dropped},
                {"merge_capacity", merges_.capacity},
                {"control_count", controls_.count}, {"control_dropped", controls_.dropped},
                {"control_capacity", controls_.capacity}}.dump() << '\n';
    for (std::size_t i = 0; i < plans_.count; ++i) {
      const auto& r = plans_.rows[i];
      json raw = json::array(), resampled = json::array();
      for (int j = 0; j < std::min(r.raw_frames, kRawFrames) * 36; ++j)
        raw.push_back(r.raw_qpos[j]);
      for (int j = 0; j < std::min(r.resampled_frames, kMotionFrames); ++j)
        resampled.push_back(r.resampled_quat[j]);
      out << json{{"event", "plan"}, {"generation", r.generation},
                  {"initial", r.initial},
                  {"generation_frame", r.generation_frame}, {"mode", r.mode},
                  {"seed", r.seed}, {"begin", r.begin}, {"end", r.end},
                  {"movement_buffer_at", r.movement_buffer_at},
                  {"gather_us", r.initial ? json(nullptr) : json(r.gather_us)},
                  {"inference_us", r.initial ? json(nullptr) : json(r.inference_us)},
                  {"resample_us", r.initial ? json(nullptr) : json(r.resample_us)}, {"speed", r.speed},
                  {"height", r.height}, {"received_facing", r.received_facing},
                  {"facing", r.facing}, {"movement", r.movement},
                  {"context", r.context}, {"raw_frames", r.raw_frames},
                  {"resampled_frames", r.resampled_frames},
                  {"trajectory_truncated", r.raw_frames > kRawFrames || r.resampled_frames > kMotionFrames},
                  {"raw_qpos", raw}, {"resampled_quat", resampled}}.dump() << '\n';
    }
    for (std::size_t i = 0; i < merges_.count; ++i) {
      const auto& r = merges_.rows[i];
      json quat = json::array();
      for (int j = 0; j < std::min(r.frames, kMotionFrames); ++j) quat.push_back(r.quat[j]);
      out << json{{"event", "merge"}, {"at", r.at}, {"generation", r.generation},
                  {"published_generation_frame", r.published_generation_frame},
                  {"used_generation_frame", r.used_generation_frame},
                  {"previous_frame", r.previous_frame}, {"first", r.first},
                  {"frames", r.frames}, {"trajectory_truncated", r.frames > kMotionFrames},
                  {"quat", quat}}.dump() << '\n';
    }
    for (std::size_t i = 0; i < controls_.count; ++i) {
      const auto& r = controls_.rows[i];
      out << json{{"event", "control"}, {"generation", r.generation},
                  {"tick", r.tick}, {"frame", r.frame},
                  {"observation_window_frames", r.observation_window_frames},
                  {"planner_motion", r.planner_motion}, {"begin", r.begin},
                  {"observations_end", r.observations_end}, {"policy_end", r.policy_end},
                  {"telemetry_end", r.telemetry_end}, {"measurement_at", r.measurement_at},
                  {"active_quat", r.active_quat}, {"heading_quat", r.heading_quat},
                  {"target_quat", r.target_quat}, {"measured_quat", r.measured_quat},
                  {"motor_target", r.motor_target}, {"motor_measured", r.motor_measured}}.dump() << '\n';
    }
    if (!out) throw std::runtime_error("Failed to write planner trace");
  }

 private:
  template <class T> struct Stream {
    explicit Stream(std::size_t size) : rows(std::make_unique<T[]>(size)), capacity(size) {}
    std::unique_ptr<T[]> rows;
    std::size_t capacity, count = 0, dropped = 0;
  };
  template <class T> bool RecordInto(Stream<T>& stream, const T& row) {
    if (sealed_.load()) return false;
    writers_.fetch_add(1);
    if (sealed_.load()) { writers_.fetch_sub(1); return false; }
    const bool retained = stream.count < stream.capacity;
    if (retained) stream.rows[stream.count++] = row;
    else ++stream.dropped;
    writers_.fetch_sub(1);
    return retained;
  }
  Stream<Plan> plans_;
  Stream<Merge> merges_;
  Stream<Control> controls_;
  std::atomic<bool> sealed_{false};
  std::atomic<bool> frozen_{false};
  std::atomic<int> writers_{0};
};
#endif
