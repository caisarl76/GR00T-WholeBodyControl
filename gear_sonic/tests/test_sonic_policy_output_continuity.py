"""Regression test for policy output continuity across teleop transitions.

The test compiles the production method bodies directly into a deliberately
small harness.  This keeps the regression coupled to the real C++ mapping and
exposes the old output ramp, which incorrectly changed policy targets.
"""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile

SOURCE = Path(__file__).parents[2] / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp"
PARAMETERS = SOURCE.parent.parent / "include/policy_parameters.hpp"


def _extract_method(name: str) -> str:
    text = SOURCE.read_text()
    match = re.search(rf"\b(?:bool|void)\s+{name}\s*\([^)]*\)\s*\{{", text)
    assert match, f"{name} was not found in production source"
    opening = text.index("{", match.start())
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index]
    raise AssertionError(f"unterminated {name} body")


def test_create_policy_command_preserves_balance_feedback_after_handoff():
    compiler = shutil.which("g++")
    assert compiler, "g++ is required for this source-level regression test"
    create_body = _extract_method("CreatePolicyCommand")
    ramp_body = (
        _extract_method("ApplyTeleopOutputRamp") if "void ApplyTeleopOutputRamp(" in SOURCE.read_text() else ""
    )
    parameter_include = PARAMETERS.as_posix()
    source = f'''#include <array>
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <atomic>
#include <cmath>
#include <memory>
#include <vector>
#include "{parameter_include}"
using namespace std;
constexpr int G1_NUM_MOTOR = 29;
struct MotorCommand {{ array<float,29> q_target{{}};
  array<double,29> tau_ff{{}};
  array<double,29> kp{{}};
  array<double,29> kd{{}};
  array<double,29> dq_target{{}};
  }};
struct MotorGainScaleConfig {{}};
void apply_motor_gain_scales(const MotorGainScaleConfig&, MotorCommand&) {{}}
struct Policy {{ vector<float> actions{{}};
  vector<float> input{{}};
  vector<float>& GetInputBuffer() {{ return input;
  }} bool Infer() {{ return true;
  }} vector<float>& GetActionBuffer() {{ return actions;
  }} }};
struct Buffer {{ shared_ptr<MotorCommand> data;
  void SetData(const MotorCommand& value) {{ data = make_shared<MotorCommand>(value);
  }} }};
struct Harness {{
  vector<double> obs_buffer_{{0.0}}; unique_ptr<Policy> policy_engine_{{new Policy}}; Buffer motor_command_buffer_;
  array<double,29> last_action{{}}; array<double,29> teleop_output_ramp_start_q_{{}};
  bool teleop_output_ramp_active_ = true; double teleop_output_ramp_elapsed_ = 0.0; double control_dt_ = 0.02;
      MotorGainScaleConfig motor_gain_scales_;
      static constexpr double kTeleopOutputRampDuration = 1.0;
      void ApplyTeleopOutputRamp(MotorCommand& motor_command) {{ {ramp_body} }}
      bool CreatePolicyCommand() {{ {create_body} }}
}};
int main() {{
  Harness h;
  h.policy_engine_->actions.resize(29); h.policy_engine_->input.resize(1);
  for (int i=0; i<29; ++i) h.teleop_output_ramp_start_q_[i] = 0.0;
  for (int frame=0; frame<51; ++frame) {{
    for (int i=0; i<29; ++i) h.policy_engine_->actions[i] = 0.1f * (i + 1) + 0.01f * frame;
    if (!h.CreatePolicyCommand()) return 2;
    const auto& command = *h.motor_command_buffer_.data;
    for (int i=0; i<29; ++i) {{
      const double expected = default_angles[i] +
          h.policy_engine_->actions[isaaclab_to_mujoco[i]] * g1_action_scale[i];
      if (abs(command.q_target[i] - expected) > 1e-5) {{ cerr << "ramped q_target at " << i << "\\n"; return 10; }}
      if (abs(h.last_action[i] - h.policy_engine_->actions[i]) > 1e-6) return 11;
      const double executed = (command.q_target[i] - default_angles[i]) / g1_action_scale[i];
      if (abs(executed - h.last_action[isaaclab_to_mujoco[i]]) > 1e-5) return 12;
    }}
  }}
  return 0;
}}
'''
    with tempfile.TemporaryDirectory() as directory:
        cpp = Path(directory) / "regression.cpp"
        binary = Path(directory) / "regression"
        cpp.write_text(source)
        build = subprocess.run(
            [compiler, "-std=c++20", "-O0", str(cpp), "-o", str(binary)], capture_output=True, text=True
        )
        assert build.returncode == 0, build.stderr
        run = subprocess.run([str(binary)], capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
