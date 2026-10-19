#include <gtest/gtest.h>
#include <sstream>
#include <thread>
#include "../include/planner_trace.hpp"

TEST(PlannerTrace, RetainsFirstRecordsAndReportsOverflow) {
  PlannerTrace trace(1, 1, 2);
  PlannerTrace::Plan plan;
  plan.generation = 7;
  plan.context[12] = 0.75;
  EXPECT_TRUE(trace.Record(plan));
  plan.generation = 8;
  EXPECT_FALSE(trace.Record(plan));
  trace.Freeze();
  std::ostringstream out;
  trace.Write(out);
  std::istringstream lines(out.str());
  nlohmann::json metadata, retained;
  lines >> metadata >> retained;
  EXPECT_EQ(metadata["plan_dropped"], 1);
  EXPECT_EQ(metadata["plan_count"], 1);
  EXPECT_EQ(retained["generation"], 7);
  EXPECT_EQ(retained["context"][12], 0.75);
}

TEST(PlannerTrace, RequiresFreezeAndRejectsLaterWrites) {
  PlannerTrace trace(1, 1, 1);
  std::ostringstream out;
  EXPECT_THROW(trace.Write(out), std::logic_error);
  EXPECT_TRUE(out.str().empty());
  PlannerTrace::Control control;
  control.tick = 19;
  EXPECT_TRUE(trace.Record(control));
  trace.Freeze();
  control.tick = 20;
  EXPECT_FALSE(trace.Record(control));
  trace.Write(out);
  EXPECT_EQ(out.str().find("\"tick\":20"), std::string::npos);
  EXPECT_NE(out.str().find("\"tick\":19"), std::string::npos);
}

TEST(PlannerTrace, RecordsGeneratedMergedAndSelectedQuaternionsSeparately) {
  PlannerTrace trace(1, 1, 1);
  PlannerTrace::Plan plan;
  plan.generation = 3;
  plan.raw_frames = 1;
  plan.raw_qpos[3] = 0.6;
  plan.resampled_frames = 1;
  plan.resampled_quat[0][0] = 0.7;
  PlannerTrace::Merge merge;
  merge.generation = 3;
  merge.frames = 1;
  merge.quat[0][0] = 0.8;
  PlannerTrace::Control control;
  control.generation = 3;
  control.active_quat[0] = 0.9;
  control.target_quat[0] = 1.0;
  control.telemetry_end = 15.25;
  trace.Record(plan);
  trace.Record(merge);
  trace.Record(control);
  trace.Freeze();
  std::ostringstream out;
  trace.Write(out);
  std::istringstream lines(out.str());
  nlohmann::json metadata, generated, blended, selected;
  lines >> metadata >> generated >> blended >> selected;
  EXPECT_DOUBLE_EQ(generated["raw_qpos"][3], 0.6);
  EXPECT_DOUBLE_EQ(generated["resampled_quat"][0][0], 0.7);
  EXPECT_DOUBLE_EQ(blended["quat"][0][0], 0.8);
  EXPECT_DOUBLE_EQ(selected["active_quat"][0], 0.9);
  EXPECT_DOUBLE_EQ(selected["target_quat"][0], 1.0);
  EXPECT_DOUBLE_EQ(selected["telemetry_end"], 15.25);
}

TEST(PlannerTrace, FreezeProducesConsistentSnapshotWithConcurrentWriter) {
  PlannerTrace trace(1, 1, 5000);
  std::atomic<bool> started{false};
  std::thread writer([&] {
    PlannerTrace::Control control;
    for (int i = 0; i < 5000; ++i) {
      control.tick = i;
      control.target_quat.fill(i);
      trace.Record(control);
      if (i == 20) started.store(true);
    }
  });
  while (!started.load()) std::this_thread::yield();
  trace.Freeze();
  std::ostringstream out;
  trace.Write(out);
  writer.join();
  std::istringstream lines(out.str());
  nlohmann::json row;
  lines >> row;
  const int count = row["control_count"];
  int observed = 0;
  std::string line;
  std::getline(lines, line);  // Consume the metadata line's newline.
  while (std::getline(lines, line)) {
    row = nlohmann::json::parse(line);
    for (double value : row["target_quat"]) EXPECT_EQ(value, observed);
    EXPECT_EQ(row["tick"], observed++);
  }
  EXPECT_EQ(observed, count);
  EXPECT_GE(observed, 21);
}

TEST(PlannerTrace, OptionIsLimitedToLoopbackSimulation) {
  EXPECT_NO_THROW(ValidatePlannerTraceOptions("", "eth0", false, "keyboard", ""));
  EXPECT_NO_THROW(ValidatePlannerTraceOptions("trace.jsonl", "lo", true, "zmq_manager", "model.onnx"));
  EXPECT_THROW(ValidatePlannerTraceOptions("trace.jsonl", "eth0", true, "zmq_manager", "model.onnx"), std::invalid_argument);
  EXPECT_THROW(ValidatePlannerTraceOptions("trace.jsonl", "lo", false, "zmq_manager", "model.onnx"), std::invalid_argument);
  EXPECT_THROW(ValidatePlannerTraceOptions("trace.jsonl", "lo", true, "keyboard", "model.onnx"), std::invalid_argument);
  EXPECT_THROW(ValidatePlannerTraceOptions("trace.jsonl", "lo", true, "zmq_manager", ""), std::invalid_argument);
}

TEST(PlannerTrace, InitializationDoesNotInventPhaseTiming) {
  PlannerTrace trace(1, 1, 1);
  PlannerTrace::Plan plan;
  plan.initial = true;
  plan.inference_us = 999;  // Stale timing must not be presented as initialization.
  trace.Record(plan);
  trace.Freeze();
  std::ostringstream out;
  trace.Write(out);
  std::istringstream lines(out.str());
  nlohmann::json metadata, row;
  lines >> metadata >> row;
  EXPECT_TRUE(row["gather_us"].is_null());
  EXPECT_TRUE(row["inference_us"].is_null());
  EXPECT_TRUE(row["resample_us"].is_null());
}
