"""Exercise the real state logger and ZMQ publisher with production fast-math flags.

The wire check uses a temporary TCP port; no DDS, robot, or CSV writer is started.
Run: PYTHONPATH=. pytest gear_sonic/tests/test_pico_dex3_feedback.py -q
"""

import os
from pathlib import Path
import socket
import subprocess

import msgpack
import zmq

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref"


def test_dds_feedback_age_survives_logging_and_republication(tmp_path):
    source = tmp_path / "test_dex3_feedback.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <limits>
#include <mutex>
#include <thread>
#include "robot_parameters.hpp"
#include "utils.hpp"
#include "output_interface/zmq_output_handler.hpp"

using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;
using Fields = std::map<std::string, msgpack::object>;

void log(StateLogger& logger, const std::array<Clock::time_point, 2>& stamps = {}) {
    std::array<double, 4> quat{1, 0, 0, 0};
    std::array<double, 3> vec{};
    std::array<double, 29> body{};
    std::array<double, 7> hand{};
    logger.LogFullState(quat, vec, vec, quat, vec, vec,
        std::span(body), std::span(body), std::span(body), std::span(body),
        std::span(body), std::span(body), std::span(hand), std::span(hand),
        std::span(hand), std::span(hand), std::span(hand), std::span(hand), 0, stamps);
}

msgpack::object_handle pack(const StateLogger::Entry& state, Clock::time_point now,
                            const HeadingState* heading = nullptr) {
    msgpack::sbuffer buffer;
    ZMQOutputHandler::pack_state(buffer, state, {{"viz_test", {1, 2, 3}}}, heading, now);
    size_t offset = 0;
    auto result = msgpack::unpack(buffer.data(), buffer.size(), offset);
    // An incorrect map field count either leaves trailing bytes or fails unpacking.
    assert(offset == buffer.size());
    assert(result.get().type == msgpack::type::MAP);
    assert(result.get().via.map.size == (heading ? 25 : 23));
    return result;
}

int64_t check(const StateLogger::Entry& state, Clock::time_point now,
              bool left_valid, bool right_valid, int64_t left_age, int64_t right_age) {
    auto owner = pack(state, now);
    const auto fields = owner.get().as<Fields>();
    assert(fields.at("left_hand_feedback_valid").as<bool>() == left_valid);
    assert(fields.at("right_hand_feedback_valid").as<bool>() == right_valid);
    const auto actual_left_age = fields.at("left_hand_feedback_age_ns").as<int64_t>();
    assert(actual_left_age == left_age);
    assert(fields.at("right_hand_feedback_age_ns").as<int64_t>() == right_age);
    return actual_left_age;
}

int main(int argc, char** argv) {
    if (argc == 2) {
        StateLogger logger("", 8, 29, 29, .02, false, {{"control_frequency", 50.0}});
        ZMQOutputHandler publisher(logger, std::stoi(argv[1]), "g1_debug");
        DataBuffer<HeadingState> heading;
        heading.SetData(HeadingState{});
        std::array<double, 9> position{};
        std::array<double, 12> orientation{};
        std::array<double, 3> compliance{};
        std::array<double, 7> hand{};
        std::array<double, 4> root{1, 0, 0, 0};
        std::array<double, 2> tokens{0.25, 0.5};
        for (int tick = 0; tick < 200; ++tick) {
            const auto received = Clock::now();
            log(logger, {received, received - 1s});
            assert(logger.LogPostState(std::span(tokens), 0, "planner_motion", true));
            publisher.publish(position, orientation, compliance, hand, hand,
                              root, heading, nullptr, 0);
            std::this_thread::sleep_for(10ms);
        }
        return 0;
    }
    const auto now = Clock::now();
    StateLogger logger("", 8, 29, 29, .02, false);
    log(logger);
    auto state = logger.GetLatest(1).at(0);
    // Default/missing DDS data cannot make zero-valued fallback positions ready.
    check(state, now, false, false, -1, -1);

    // DataBuffer uses the DDS callback's receive time and does not restamp reads.
    DataBuffer<std::array<double, 7>> dds;
    assert(!dds.GetDataWithTime().HasData());
    dds.SetData(std::array<double, 7>{});
    const auto receive = dds.GetDataWithTime().timestamp;
    log(logger, {receive, receive - 100ms});
    state = logger.GetLatest(1).at(0);
    assert(state.hand_feedback_time[0] == receive);
    assert(state.hand_feedback_time[1] == receive - 100ms);
    check(state, receive, true, false, 0, 100000000);
    check(state, receive + 99999999ns, true, false, 99999999, 199999999);
    check(state, receive + 100ms, false, false, 100000000, 200000000);

    // Re-logging/re-publishing cached source data must not refresh its source age.
    assert(dds.GetDataWithTime().timestamp == receive);
    log(logger, {receive, receive});
    state = logger.GetLatest(1).at(0);
    check(state, receive + 101ms, false, false, 101000000, 101000000);
    const auto before = check(state, receive + 102ms, false, false, 102000000, 102000000);
    const auto after = check(state, receive + 103ms, false, false, 103000000, 103000000);
    assert(after > before);

    HeadingState heading;
    const auto heading_owner = pack(state, receive + 103ms, &heading);
    const auto heading_fields = heading_owner.get().as<Fields>();
    assert(heading_fields.contains("init_base_quat"));
    assert(heading_fields.at("delta_heading").as<double>() == 0);
    assert(heading_fields.at("viz_test").as<std::vector<double>>().size() == 3);

    state.hand_feedback_time = {receive + 1ns, receive};
    check(state, receive, false, true, -1, 0);
    state.hand_feedback_time = {receive, receive};
    state.left_hand_q.resize(6);
    check(state, receive, false, true, -1, 0);
    state.left_hand_q.resize(7);
    for (const uint64_t bits : {UINT64_C(0x7ff8000000000000), UINT64_C(0x7ff0000000000000),
                                UINT64_C(0xfff0000000000000)}) {
        state.left_hand_q[2] = std::bit_cast<double>(bits);
        check(state, receive, false, true, -1, 0);
    }
    state.left_hand_q[2] = 0.0;
    state.right_hand_q[5] = std::bit_cast<double>(UINT64_C(0x7ff8000000000000));
    check(state, receive, true, false, 0, -1);

    // Compile and exercise the old caller signature without either optional argument.
    std::array<double, 4> quat{};
    std::array<double, 3> vec{};
    std::span<double> empty;
    logger.LogFullState(quat, vec, vec, quat, vec, vec, empty, empty, empty,
                       empty, empty, empty, empty, empty, empty, empty, empty, empty);
    state = logger.GetLatest(1).at(0);
    check(state, receive, false, false, -1, -1);
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / "test_dex3_feedback"
    subprocess.run(
        [
            os.environ.get("CXX", "c++"),
            "-std=c++20",
            "-O3",
            "-ffast-math",
            "-pthread",
            f"-I{DEPLOY / 'include'}",
            str(source),
            str(DEPLOY / "src/state_logger.cpp"),
            str(DEPLOY / "src/file_sink.cpp"),
            "-lzmq",
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)

    # Reserve an OS-selected port instead of connecting to any robot endpoint.
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    with zmq.Context() as context, context.socket(zmq.SUB) as subscriber:
        subscriber.setsockopt(zmq.LINGER, 0)
        subscriber.setsockopt(zmq.SUBSCRIBE, b"g1_debug")
        subscriber.connect(f"tcp://127.0.0.1:{port}")
        with subprocess.Popen(
            [str(binary), str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as publisher:
            try:
                for _ in range(5):
                    if not subscriber.poll(3000):
                        stdout, stderr = publisher.communicate(timeout=5)
                        raise AssertionError(
                            "Actual ZMQOutputHandler.publish() emitted no g1_debug feedback: "
                            f"exit={publisher.returncode}, stdout={stdout!r}, stderr={stderr!r}"
                        )
                    packet = subscriber.recv()
                    assert not subscriber.getsockopt(zmq.RCVMORE), "Feedback must be one topic-prefixed frame"
                    assert packet.startswith(b"g1_debug")
                    # unpackb rejects extra bytes from an incorrect msgpack map length.
                    fields = msgpack.unpackb(packet[len(b"g1_debug") :], raw=False)
                    assert fields["control_loop_type"] == "cpp"
                    assert fields["token_state"] == [0.25, 0.5]
                    assert len(fields["body_q_measured"]) == 29
                    for side in ("left", "right"):
                        assert fields[f"{side}_hand_q"] == [0.0] * 7
                        assert fields[f"{side}_hand_q_measured"] == [0.0] * 7
                        assert type(fields[f"{side}_hand_feedback_valid"]) is bool
                        assert type(fields[f"{side}_hand_feedback_age_ns"]) is int
                    assert fields["left_hand_feedback_valid"]
                    assert 0 <= fields["left_hand_feedback_age_ns"] < 100_000_000
                    assert not fields["right_hand_feedback_valid"]
                    assert fields["right_hand_feedback_age_ns"] >= 1_000_000_000
                stdout, stderr = publisher.communicate(timeout=5)
                assert publisher.returncode == 0, (stdout, stderr)
            finally:
                if publisher.poll() is None:
                    publisher.terminate()
                    publisher.communicate(timeout=5)
