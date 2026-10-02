#include <gtest/gtest.h>
#include <zmq.h>
#include <cstring>
#include <thread>

#include "../include/input_interface/zmq_manager.hpp"
#include "../include/output_interface/output_interface.hpp"

namespace {
class TestOutput : public OutputInterface {
 public:
  explicit TestOutput(StateLogger& logger) : OutputInterface(logger) {}
  void publish(const std::array<double, 9>& a, const std::array<double, 12>& b,
               const std::array<double, 3>& c, const std::array<double, 7>& d,
               const std::array<double, 7>& e, const std::array<double, 4>& f,
               DataBuffer<HeadingState>& g, std::shared_ptr<const MotionSequence> h,
               int i) override { create_output_data_map(a, b, c, d, e, f, g, h, i); }
  const auto& data() const { return output_data_map_; }
};

struct Publisher {
  void* context = zmq_ctx_new();
  void* socket = zmq_socket(context, ZMQ_PUB);
  int port;
  Publisher() {
    if (zmq_bind(socket, "tcp://127.0.0.1:*") != 0) throw std::runtime_error("bind failed");
    char endpoint[128]; size_t size = sizeof(endpoint);
    zmq_getsockopt(socket, ZMQ_LAST_ENDPOINT, endpoint, &size);
    port = std::stoi(std::string(endpoint).substr(std::string(endpoint).rfind(':') + 1));
  }
  ~Publisher() { zmq_close(socket); zmq_ctx_term(context); }
  void planner() {
    const std::string header = R"({"v":1,"endian":"le","count":1,"fields":[{"name":"mode","dtype":"i32","shape":[1]},{"name":"movement","dtype":"f32","shape":[3]},{"name":"facing","dtype":"f32","shape":[3]},{"name":"speed","dtype":"f32","shape":[1]},{"name":"height","dtype":"f32","shape":[1]},{"name":"upper_body_position","dtype":"f32","shape":[17]},{"name":"upper_body_velocity","dtype":"f32","shape":[17]},{"name":"left_hand_joints","dtype":"f32","shape":[7]},{"name":"right_hand_joints","dtype":"f32","shape":[7]}]})";
    std::string message = "planner" + header + std::string(1280-header.size(), '\0');
    int32_t mode = 1;
    message.append(reinterpret_cast<const char*>(&mode), sizeof(mode));
    std::vector<float> values{1,0,0, 0.8,0.6,0, 0.2,-1};
    values.insert(values.end(), 17, 0.35f);
    values.insert(values.end(), 17, 0.25f);
    values.insert(values.end(), 7, 0.45f);
    values.insert(values.end(), 7, -0.55f);
    message.append(reinterpret_cast<const char*>(values.data()), values.size()*sizeof(float));
    zmq_send(socket, message.data(), message.size(), 0);
  }
  void command(bool stop, bool planner) {
    const std::string header = R"({"v":1,"endian":"le","count":1,"fields":[{"name":"start","dtype":"bool","shape":[1]},{"name":"stop","dtype":"bool","shape":[1]},{"name":"planner","dtype":"bool","shape":[1]}]})";
    std::string message = "command" + header + std::string(1280-header.size(), '\0');
    message.push_back(0); message.push_back(stop); message.push_back(planner);
    zmq_send(socket, message.data(), message.size(), 0);
  }
};
}

TEST(HarnessTelemetry, AbsoluteMotorOrderingAndEffectiveCapability) {
  StateLogger logger("", 2, 29, 29, 0.02, false);
  std::array<double, 29> body{}, zeros{};
  for (int i = 0; i < 29; ++i) body[i] = i * 0.01;
  std::array<double, 7> hands{0.1,0.2,0.3,0.4,0.5,0.6,0.7};
  std::span<double> q(body), z(zeros), hand(hands), empty;
  const std::array<double, 4> identity{1,0,0,0};
  const auto now = std::chrono::steady_clock::now();
  logger.LogFullState(identity, {}, {}, identity, {}, {}, q,z,z,empty,empty,empty,
                      hand,hand,hand,hand,hand,hand,0.0,now,now);
  TestOutput output(logger);
  DataBuffer<HeadingState> heading;
  output.publish({}, {}, {}, {}, {}, identity, heading, nullptr, 0);
  const auto& actual = output.data().at("body_q_measured_motor");
  ASSERT_EQ(actual.size(), 29);
  const std::array<double, 29> expected{
    -0.312,0.03,0.06,0.759,-0.233,0.17,-0.302,0.04,0.07,0.769,-0.223,0.18,
    0.02,0.05,0.08,0.31,0.35,0.19,0.81,0.23,0.25,0.27,0.32,-0.04,0.20,
    0.82,0.24,0.26,0.28};
  for (int i = 0; i < 29; ++i) EXPECT_NEAR(actual[i], expected[i], 1e-12) << i;
  EXPECT_EQ(output.data().at("left_hand_q_measured"), (std::vector<double>(hands.begin(), hands.end())));
  EXPECT_EQ(output.data().at("right_hand_q_measured"), (std::vector<double>(hands.begin(), hands.end())));
  EXPECT_EQ(output.data().at("harness_planner_hold_enabled"), std::vector<double>{0});
  output.SetHarnessPlannerHold(true);
  output.publish({}, {}, {}, {}, {}, identity, heading, nullptr, 0);
  EXPECT_EQ(output.data().at("harness_planner_hold_enabled"), std::vector<double>{1});
  // Disabled/absent and stale hands cannot be reported as measured-open.
  for (bool missing_left : {false, true}) {
    for (bool stale : {false, true}) {
      const auto absent = stale ? now - std::chrono::seconds(1) : std::chrono::steady_clock::time_point{};
      logger.LogFullState(identity, {}, {}, identity, {}, {}, q,z,z,empty,empty,empty,
                          hand,hand,hand,hand,hand,hand,0.0,
                          missing_left ? absent : now, missing_left ? now : absent);
      output.publish({}, {}, {}, {}, {}, identity, heading, nullptr, 0);
      EXPECT_TRUE(output.data().at(missing_left ? "left_hand_q_measured" : "right_hand_q_measured").empty());
      EXPECT_EQ(output.data().at(missing_left ? "right_hand_q_measured" : "left_hand_q_measured").size(), 7);
    }
  }
  logger.LogFullState(identity, {}, {}, identity, {}, {}, q,z,z,empty,empty,empty,
                      hand,hand,hand,hand,hand,hand);  // Disabled Dex3: no receipts.
  output.publish({}, {}, {}, {}, {}, identity, heading, nullptr, 0);
  EXPECT_TRUE(output.data().at("left_hand_q_measured").empty());
  EXPECT_TRUE(output.data().at("right_hand_q_measured").empty());
}

TEST(HarnessTimeout, RetainsOnlyOptedInTargetsAndZerosVelocity) {
  for (bool enabled : {false, true}) {
    Publisher publisher;
    ZMQManager manager("127.0.0.1", publisher.port, "pose", "command", "planner",
                       false, false, Vr3PtSafetyFilter::Config{}, enabled);
    MotionDataReader reader;
    auto motion = std::make_shared<MotionSequence>(); motion->name = "planner_motion";
    std::shared_ptr<const MotionSequence> current = motion;
    int frame = 0; bool reinitialize = false, report = false;
    OperatorState state; state.start = true;
    PlannerState planner; planner.enabled = planner.initialized = true;
    DataBuffer<HeadingState> heading; DataBuffer<MovementState> movement;
    movement.SetData(MovementState()); std::mutex mutex;
    auto tick = [&] { manager.handle_input(reader, current, frame, state, reinitialize,
                         heading, true, planner, movement, mutex, report); };
    // No previous command must not manufacture hand/arm overrides.
    tick(); EXPECT_FALSE(manager.GetHandPose(true).first);
    for (int i = 0; i < 100 && !manager.GetHandPose(true).first; ++i) {
      publisher.planner(); std::this_thread::sleep_for(std::chrono::milliseconds(10)); tick();
    }
    ASSERT_TRUE(manager.GetHandPose(true).first);
    ASSERT_TRUE(manager.GetUpperBodyJointPositions().first);
    EXPECT_NEAR(manager.GetUpperBodyJointVelocities().second[0], 0.25, 1e-6);
    std::this_thread::sleep_for(std::chrono::milliseconds(1250)); tick();
    EXPECT_EQ(movement.GetDataWithTime().data->locomotion_mode, 0);
    EXPECT_EQ(movement.GetDataWithTime().data->movement_direction, (std::array<double,3>{0,0,0}));
    EXPECT_EQ(manager.GetHandPose(true).first, enabled);
    EXPECT_EQ(manager.GetUpperBodyJointPositions().first, enabled);
    if (enabled) {
      EXPECT_NEAR(manager.GetHandPose(true).second[0], 0.45, 1e-6);
      EXPECT_NEAR(manager.GetHandPose(false).second[0], -0.55, 1e-6);
      EXPECT_NEAR(manager.GetUpperBodyJointPositions().second[0], 0.35, 1e-6);
      EXPECT_EQ(manager.GetUpperBodyJointVelocities().second, (std::array<double,17>{}));
    }
  }
}

TEST(HarnessTimeout, ModeBoundariesAndStopsClearRetainedTargets) {
  for (int scenario = 0; scenario < 3; ++scenario) {
    Publisher publisher;
    ZMQManager manager("127.0.0.1", publisher.port, "pose", "command", "planner",
                       false, false, Vr3PtSafetyFilter::Config{}, true);
    MotionDataReader reader;
    auto motion = std::make_shared<MotionSequence>(); motion->name = "planner_motion";
    std::shared_ptr<const MotionSequence> current = motion;
    int frame = 0; bool reinitialize = false, report = false;
    OperatorState state; state.start = true;
    PlannerState planner; planner.enabled = planner.initialized = true;
    DataBuffer<HeadingState> heading; DataBuffer<MovementState> movement;
    movement.SetData(MovementState()); std::mutex mutex;
    auto handle = [&] { manager.handle_input(reader, current, frame, state,
                         reinitialize, heading, true, planner, movement, mutex, report); };
    auto tick = [&] { manager.update(); handle(); };
    auto send_command = [&](bool stop, bool planner_mode) {
      publisher.command(stop, planner_mode);
      std::this_thread::sleep_for(std::chrono::milliseconds(30)); manager.update();
      if (planner_mode && !stop) {
        // Stand in for the planner thread after streamed mode changed motion.
        motion->name = "planner_motion";
        current = motion;
        planner.enabled = planner.initialized = true;
      }
      handle();
    };
    for (int i = 0; i < 100 && !manager.GetHandPose(true).first; ++i) {
      publisher.planner(); std::this_thread::sleep_for(std::chrono::milliseconds(10)); tick();
    }
    ASSERT_TRUE(manager.GetUpperBodyJointPositions().first);
    std::this_thread::sleep_for(std::chrono::milliseconds(1250)); tick();
    ASSERT_TRUE(manager.GetUpperBodyJointPositions().first);
    if (scenario == 0) {
      send_command(false, false);
      EXPECT_FALSE(manager.GetUpperBodyJointPositions().first);
      EXPECT_FALSE(manager.GetHandPose(true).first);
      send_command(false, true);
    } else if (scenario == 1) {
      send_command(true, true);
      EXPECT_TRUE(state.stop);
    } else {
      manager.PushStdinChar('o'); tick();
      EXPECT_TRUE(state.stop);
    }
    EXPECT_FALSE(manager.GetUpperBodyJointPositions().first);
    EXPECT_FALSE(manager.GetHandPose(true).first);
    tick();
    EXPECT_FALSE(manager.GetUpperBodyJointPositions().first);
    EXPECT_FALSE(manager.GetHandPose(true).first);
    if (scenario == 0) {
      // An intentional fresh packet may prime entry to the next planner session.
      send_command(false, false);
      publisher.planner(); std::this_thread::sleep_for(std::chrono::milliseconds(20));
      send_command(false, true);
      tick();
      EXPECT_TRUE(manager.GetUpperBodyJointPositions().first);
      EXPECT_TRUE(manager.GetHandPose(true).first);
      EXPECT_NEAR(manager.GetUpperBodyJointVelocities().second[0], 0.25, 1e-6);
    }
  }
}
