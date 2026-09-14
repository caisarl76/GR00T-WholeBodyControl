// Real input headers; -fno-access-control permits deterministic callback delivery
// without a publisher, robot, or planner inference thread.
#include "input_interface/zmq_manager.hpp"
#include <stdexcept>

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

struct Fixture {
  ZMQManager manager{"127.0.0.1", 59993};
  MotionDataReader reader;
  std::shared_ptr<MotionSequence> original = std::make_shared<MotionSequence>();
  std::shared_ptr<const MotionSequence> current;
  int frame = 12;
  OperatorState op;
  PlannerState planner;
  bool heading = false, temperature = false;
  DataBuffer<HeadingState> heading_buffer;
  DataBuffer<MovementState> movement_buffer;
  std::mutex mutex;

  Fixture() {
    manager.command_subscriber_->Stop();
    manager.planner_subscriber_->Stop();
    manager.pose_interface_->subscriber_->Stop();
    original->name = "planner_motion";
    original->ReserveCapacity(30);
    original->timesteps = 30;
    original->SetEncodeMode(0);
    for (int f = 0; f < 30; ++f) {
      original->BodyQuaternions(f)[0] = {1, 0, 0, 0};
      std::fill_n(original->JointPositions(f), 29, 0.1);
    }
    reader.motions.push_back(original);
    current = original;
    op.start = op.play = true;
    planner.enabled = planner.initialized = true;
    heading_buffer.SetData(HeadingState({1, 0, 0, 0}, 0.8));
    movement_buffer.SetData(MovementState(0, {0, 0, 0}, {0, 1, 0}, -1, -1));
  }

  void command(bool use_planner, bool stop = false) {
    using Sub = ZMQPackedMessageSubscriber;
    Sub::DecodedHeader h;
    h.version = 1;
    h.fields = {{"start", "bool", {1}}, {"stop", "bool", {1}}, {"planner", "bool", {1}}};
    uint8_t start = 1, stop_value = stop, p = use_planner;
    manager.OnCommandReceived("command", h, {{&start, 1}, {&stop_value, 1}, {&p, 1}});
  }

  void pose(int version = 3) {
    using Sub = ZMQPackedMessageSubscriber;
    Sub::DecodedHeader h;
    h.version = version;
    h.endian = "le";
    h.fields = {{"joint_pos", "f64", {5, 29}}, {"joint_vel", "f64", {5, 29}},
                {"body_quat", "f64", {5, 1, 4}}, {"frame_index", "i32", {5}},
                {"smpl_joints", "f64", {5, 24, 3}}, {"smpl_pose", "f64", {5, 24, 3}}};
    double q[145], dq[145] = {}, quat[20] = {}, smpl[360] = {};
    int32_t frames[5] = {0, 1, 2, 3, 4};
    std::fill_n(q, 145, 0.6);
    for (int i = 0; i < 5; ++i) quat[4 * i] = 1;
    manager.pose_interface_->OnPoseDataReceived("pose", h,
      {{q, sizeof(q)}, {dq, sizeof(dq)}, {quat, sizeof(quat)},
       {frames, sizeof(frames)}, {smpl, sizeof(smpl)}, {smpl, sizeof(smpl)}});
  }

  void tick() {
    manager.update();
    manager.handle_input(reader, current, frame, op, heading, heading_buffer,
                         true, planner, movement_buffer, mutex, temperature);
  }
};

// Only the model backend is synthetic; context setup, resampling and generation
// stamping run through LocalMotionPlannerBase exactly as in deployment.
struct PlannerBackend : LocalMotionPlannerBase {
  float context[144] = {}, qpos[72] = {}, movement[3] = {}, facing[3] = {};
  int mode = 0;
  bool cancel_during_inference = false;
  PlannerBackend() { qpos[3] = qpos[39] = 1; }
  bool InitializeSpecific() override { return true; }
  void RunInference() override {
    if (cancel_during_inference) ++planner_state_.generation;
  }
  void UpdateInputTensors(int m, float, float, const std::array<float, 3>& move,
                         const std::array<float, 3>& face, int) override {
    mode = m;
    std::copy(move.begin(), move.end(), movement);
    std::copy(face.begin(), face.end(), facing);
  }
  float* GetContextBuffer() override { return context; }
  int32_t GetNumPredFrames() override { return 2; }
  const float* GetMujocoQposBuffer() override { return qpos; }
  float* GetMovementDirectionValues() override { return movement; }
  float* GetFacingDirectionValues() override { return facing; }
  float GetTargetVelValue() override { return 0; }
  float GetHeightValue() override { return 0; }
  int32_t GetRandomSeedValue() override { return 0; }
  int32_t GetModeValue() override { return mode; }
};

int main() {
  try {
    {
      Fixture f;
      f.command(false);
      f.tick();
      require(f.current == f.original && f.frame == 12 && f.op.play && f.planner.enabled,
              "pose preparation must retain the active planner reference and playback");
      require(!f.heading, "pose preparation must preserve heading");
      f.pose();
      f.tick();
      require(f.current->name == "streamed" && f.current->GetEncodeMode() == 2 && f.frame == 0,
              "first v3 pose must commit with its own frame origin and SMPL encoder");
      require(f.op.play && !f.planner.enabled && !f.heading,
              "pose commit must disable planner without pausing or resetting heading");
      require(f.planner.heading_handoff.has_value(),
              "pose commit must queue reference heading rebasing for control");
    }
    for (int version : {1, 3}) {
      Fixture f;
      f.pose(version);
      f.command(false);
      f.tick();
      require(f.current->name == "streamed" && f.current->JointPositions(0)[0] == 0.6,
              "mode command must retain and consume the buffered first pose");
    }
    {
      Fixture f;
      f.pose();
      f.manager.pose_interface_->last_receive_time_ =
          std::chrono::steady_clock::now() - std::chrono::seconds(1);
      f.command(false);
      f.tick();
      require(f.current == f.original && f.planner.enabled && f.op.play,
              "stale prebuffer must not replace the active planner");
      f.pose();
      f.manager.pose_interface_->buffered_header_.fields[4].name = "missing_smpl_joints";
      f.tick();
      require(f.current == f.original && f.planner.enabled && f.op.play,
              "malformed v3 first payload must not replace the planner");
      f.command(true);
      f.tick();
      require(f.current == f.original && f.op.play && !f.heading,
              "cancelling pose preparation must preserve the original reference");
    }
    {
      Fixture f;
      f.pose(); f.command(false); f.tick();
      auto streamed = f.current;
      f.manager.latest_planner_message_.valid = true;
      f.manager.latest_planner_message_.timestamp = std::chrono::steady_clock::now();
      f.manager.latest_planner_message_.mode = static_cast<int>(LocomotionMode::WALK);
      f.manager.latest_planner_message_.movement = {1, 0, 0};
      f.manager.latest_planner_message_.facing = {0, 1, 0};
      f.manager.latest_planner_message_.speed = 0.3;
      f.command(true);
      auto started = std::chrono::steady_clock::now();
      f.tick();
      require(std::chrono::steady_clock::now() - started < std::chrono::milliseconds(50),
              "planner initialization must not block an input tick");
      require(f.current == streamed && f.op.play && f.planner.enabled && !f.planner.initialized,
              "planner preparation must preserve the streamed reference");
      require(f.movement_buffer.GetDataWithTime().data->locomotion_mode == static_cast<int>(LocomotionMode::WALK) &&
              f.movement_buffer.GetDataWithTime().data->facing_direction[1] == 1,
              "fresh destination command must be available before planner inference");
      f.planner.initialized = true;
      f.tick();
      require(f.current == streamed && !f.manager.is_planner_ready_,
              "inference initialized alone is not a committed planner reference");
      f.current = f.original; f.frame = 0;  // Control thread accepts generated reference.
      f.manager.OnPlannerReferenceCommitted();
      f.tick();
      require(f.manager.is_planner_ready_ && f.op.play && !f.heading,
              "planner handoff must complete without resetting heading");
    }
    {
      Fixture f;
      f.pose(); f.command(false); f.tick();
      f.command(true); f.tick();
      require(f.current->GetEncodeMode() == 2,
              "pending planner must not change the active SMPL encoder");
      f.manager.pose_interface_->has_hand_joints_ = true;
      f.manager.has_hand_joints_ = false;
      f.command(false); f.tick();
      require(f.manager.HasHandJoints(),
              "rapid reversal must retain the controls of the committed pose source");
    }
    {
      Fixture f;
      f.pose(); f.command(false); f.tick();
      f.manager.pose_interface_->has_hand_joints_ = true;
      f.command(true); f.tick();
      require(f.manager.HasHandJoints(), "pose controls must remain active while planner initializes");
      f.current = f.original; f.planner.initialized = true;
      f.manager.OnPlannerReferenceCommitted();  // Before the next input poll.
      require(!f.manager.HasHandJoints(), "controls must change at the actual planner commit");
      f.command(false); f.tick();
      require(!f.manager.HasHandJoints(),
              "reversal before readiness poll must preserve committed planner controls");
    }
    for (bool destination_planner : {false, true}) {
      Fixture f;
      if (destination_planner) { f.pose(); f.command(false); f.tick(); }
      f.command(destination_planner); f.tick();
      auto before_stop = f.current;
      f.command(destination_planner, true); f.tick();
      require(f.op.stop && !f.planner.enabled && f.current == before_stop,
              "stop during preparation must not enable or replace the reference");
    }
    for (bool destination_planner : {false, true}) {
      Fixture f;
      if (destination_planner) { f.pose(); f.command(false); f.tick(); }
      f.command(destination_planner); f.tick();
      if (destination_planner)
        f.manager.planner_init_started_ = std::chrono::steady_clock::now() - std::chrono::seconds(6);
      else
        f.manager.pose_handoff_started_ = std::chrono::steady_clock::now() - std::chrono::seconds(6);
      auto before_timeout = f.current;
      f.tick();
      require(f.op.stop && !f.planner.enabled && f.current == before_timeout,
              "handoff deadline must stop without replacing the reference");
    }
    for (const auto& yaws : {std::array<double, 3>{0.0, M_PI / 4, -0.3},
                             std::array<double, 3>{0.4, -0.5, 0.9},
                             std::array<double, 3>{3.1, -3.1, 2.9}}) {
      auto initial_reference = euler_z_to_quat_d(yaws[0]);
      HeadingState heading(euler_z_to_quat_d(0.6), 0.25);
      const double old_world_yaw = 0.6 + 0.25 + yaws[1] - yaws[0];
      PlannerState state;
      state.QueueHeadingHandoff(euler_z_to_quat_d(yaws[1]), euler_z_to_quat_d(1.2));
      state.QueueHeadingHandoff(euler_z_to_quat_d(1.2), euler_z_to_quat_d(yaws[2]));
      heading = state.heading_handoff->Rebase(heading, initial_reference);
      const double new_world_yaw = 0.6 + heading.delta_heading;
      require(std::abs(std::sin(new_world_yaw - old_world_yaw)) < 1e-9 &&
              std::cos(new_world_yaw - old_world_yaw) > 0.999999,
              "heading handoff must preserve world yaw, including wraparound and chained commits");
      require(initial_reference == euler_z_to_quat_d(yaws[2]),
              "heading handoff must use the latest destination origin");
      require(heading.init_base_quat == euler_z_to_quat_d(0.6),
              "heading handoff must preserve the captured robot base quaternion");
    }
    {
      PlannerBackend backend;
      backend.planner_state_.enabled = true;
      backend.cancel_during_inference = true;
      const MovementState command(static_cast<int>(LocomotionMode::WALK), {1, 0, 0}, {0, 1, 0}, 0.3, -1);
      require(backend.Initialize({1, 0, 0, 0}, {}, command), "synthetic inference must succeed");
      require(backend.mode == static_cast<int>(LocomotionMode::WALK) && backend.facing[1] == 1,
              "initial inference must use the prepared destination command and facing");
      require(!backend.planner_state_.initialized,
              "cancelled initialization must not publish initialized state");
      require(backend.motion_generation_ != backend.planner_state_.generation.load(),
              "late inference output must retain its obsolete generation");
      backend.cancel_during_inference = false;
      require(backend.Initialize({1, 0, 0, 0}, {}, command), "replacement inference must succeed");
      require(backend.motion_generation_ == backend.planner_state_.generation.load(),
              "replacement inference must publish the current generation");
    }
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "PASS: managed ZMQ mode handoffs\n";
}
