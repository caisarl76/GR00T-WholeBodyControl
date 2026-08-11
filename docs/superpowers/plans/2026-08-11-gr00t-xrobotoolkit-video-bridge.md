# GR00T XRoboToolkit Video Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to execute this plan task by task. When a task is delegated within the same session, use `subagent-driven-development` for that task and preserve the review checkpoints below.

**Goal:** Add a reproducible pc2 service that subscribes to the existing GR00T RealSense ZMQ stream, serves the XRoboToolkit v1.1.1 command protocol, and sends fresh, correctly colored 2160x810 side-by-side H.264 to PICO4U without disrupting episode collection.

**Architecture:** Keep `composed_camera` as the only RealSense owner. A native C++ bridge parses fragmented TCP commands, receives newest-only GR00T frames, corrects the legacy RealSense JPEG color contract, constructs side-by-side frames, encodes Annex-B access units with Jetson GStreamer, and performs deadline-bounded TCP delivery. Python-side collection diagnostics independently verify that exporter timestamps and recorded images continue to progress.

**Tech Stack:** C++17, ZeroMQ, MessagePack, OpenCV, GStreamer 1.0, Jetson `nvv4l2h264enc`, POSIX sockets, Python 3.10, pytest, uv, systemd, FFmpeg/ffprobe.

**Approved Design:** `docs/superpowers/specs/2026-08-11-gr00t-xrobotoolkit-video-bridge-design.md` at commit `0c7bc6c` is authoritative if this plan is ambiguous.

---

## Execution Constraints

- Start implementation from a branch/worktree that contains approved design commit `0c7bc6c` and this plan commit.
- Do not modify or stage the unrelated untracked file `docs/option_b_vla_sonic_pipeline_spec.md`.
- Treat these local clones as read-only protocol references:
  - `/home/jihun/work/XRoboToolkit-Orin-Video-Sender` at `8bedc7bb225628235f97b6e51bfde5a2a3704031`.
  - `/home/jihun/work/XRoboToolkit-Unity-Client` tag `v1.1.1` at `9f775b535d781618bd2bb7ef8d6c414c0531387c`.
- Do not copy or link the upstream ZED sender. In particular, do not reuse its one-`recv()` command handling.
- Run hardware-independent tests on the development PC. Run GStreamer encoder, service, camera, and headset gates on pc2 because the development PC does not provide the Jetson encoder plugin.
- Follow test-driven development: add a focused failing test, observe the expected failure, make the smallest implementation, then rerun the focused and aggregate suites.
- Commit after each task. Stage only the paths listed in that task.

## Locked Production Contracts

Place the fixed compatibility values in `bridge_constants.hpp`; there must be no CLI overrides for them:

```cpp
inline constexpr int kSourceWidth = 640;
inline constexpr int kSourceHeight = 480;
inline constexpr int kEyeWidth = 1080;
inline constexpr int kOutputWidth = 2160;
inline constexpr int kOutputHeight = 810;
inline constexpr int kEncoderFps = 15;
inline constexpr int kEncoderBitrate = 4'000'000;
inline constexpr auto kMaxEncoderInputAge = std::chrono::milliseconds(250);
inline constexpr auto kMaxFutureSourceSkew = std::chrono::milliseconds(50);
inline constexpr auto kMaxSendStartAge = std::chrono::milliseconds(250);
inline constexpr auto kPerAccessUnitSendBudget = std::chrono::milliseconds(100);
inline constexpr auto kMaxLocalHandoffAge = std::chrono::milliseconds(350);
inline constexpr auto kVideoConnectBudget = std::chrono::seconds(2);
inline constexpr std::uint16_t kCommandPort = 13579;
inline constexpr std::uint16_t kVideoPort = 12345;
inline constexpr std::size_t kMaxGr00tMessageBytes = 4U * 1024U * 1024U;
inline constexpr std::size_t kMaxAccessUnitBytes = 4U * 1024U * 1024U;
```

The accepted request is exactly camera `VR`, `2160x810`, requested 60 FPS, requested bitrate `20'971'520`, HEVC flag `0`, render mode `2`, and port `12345`. The request IP, TCP peer, and `--allowed-headset-ip` must all be the same IPv4 address.

Use these shared values throughout the implementation rather than parallel ad-hoc representations:

```cpp
struct FrameMetadata {
    std::uint64_t sequence;
    double source_wall_time_seconds;
    std::chrono::steady_clock::time_point received_at;
};

struct DecodedFrame {
    FrameMetadata metadata;
    cv::Mat bgr_sbs;
};

struct EncodedAccessUnit {
    FrameMetadata metadata;
    std::vector<std::uint8_t> annex_b;
};

struct CameraRequest {
    std::int32_t width;
    std::int32_t height;
    std::int32_t fps;
    std::int32_t bitrate;
    std::int32_t enable_mv_hevc;
    std::int32_t render_mode;
    std::int32_t port;
    std::string camera;
    std::string ip;
};
```

## Planned File Map

Create the following maintained native files:

```text
gear_sonic/camera/xr_video_bridge/
  bridge_constants.hpp
  bridge_types.hpp
  latest_value_slot.hpp
  xr_control_protocol.hpp
  xr_control_protocol.cpp
  gr00t_frame_decoder.hpp
  gr00t_frame_decoder.cpp
  gr00t_subscriber.hpp
  gr00t_subscriber.cpp
  bounded_tcp_sender.hpp
  bounded_tcp_sender.cpp
  h264_pipeline.hpp
  h264_pipeline.cpp
  control_server.hpp
  control_server.cpp
  bridge_app.hpp
  bridge_app.cpp
  main.cpp
  Makefile
  tests/test_support.hpp
  tests/test_main.cpp
  tests/test_latest_value_slot.cpp
  tests/test_xr_control_protocol.cpp
  tests/compare_xr_control_capture.cpp
  tests/test_gr00t_frame_decoder.cpp
  tests/test_gr00t_subscriber.cpp
  tests/test_bounded_tcp_sender.cpp
  tests/test_h264_pipeline_contract.cpp
  tests/test_bridge_app.cpp
  tests/test_h264_pipeline_jetson.cpp
```

Add collection validation and deployment files:

```text
gear_sonic/utils/data_collection/camera_freshness.py
gear_sonic/scripts/validate_episode_camera_progression.py
gear_sonic/tests/test_composed_camera_client_freshness.py
gear_sonic/tests/test_camera_freshness.py
gear_sonic/tests/test_run_data_exporter_camera_freshness.py
gear_sonic/tests/test_validate_episode_camera_progression.py
gear_sonic/tests/test_install_xr_video_bridge.py
install_scripts/install_xr_video_bridge.sh
systemd/gr00t_xr_video_bridge.service.in
```

Modify:

```text
gear_sonic/scripts/run_data_exporter.py
gear_sonic/camera/composed_camera.py
install_scripts/install_camera_server.sh
docs/source/tutorials/data_collection.md
```

## Task 0: Create an Isolated Implementation Worktree

**Files:** none

- [ ] Verify the approved commits and preserve the user's current workspace state.

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
git status --short
git merge-base --is-ancestor 0c7bc6c HEAD
git show --no-patch --oneline HEAD
git -C /home/jihun/work/XRoboToolkit-Orin-Video-Sender rev-parse HEAD
git -C /home/jihun/work/XRoboToolkit-Unity-Client rev-list -n 1 v1.1.1
```

Expected: the GR00T ancestry check succeeds; the two reference SHAs match the execution constraints; the unrelated untracked document remains untouched.

- [ ] Create and enter the feature worktree.

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
git worktree add /home/jihun/work/GR00T-WholeBodyControl-xr-video-bridge -b feat/gr00t-xr-video-bridge
cd /home/jihun/work/GR00T-WholeBodyControl-xr-video-bridge
git status --short
```

Expected: clean feature worktree. Do not make an empty setup commit.

## Task 1: Establish the Native Test Harness and Newest-Only Slot

**Files:**

- Create: `gear_sonic/camera/xr_video_bridge/bridge_constants.hpp`
- Create: `gear_sonic/camera/xr_video_bridge/bridge_types.hpp`
- Create: `gear_sonic/camera/xr_video_bridge/latest_value_slot.hpp`
- Create: `gear_sonic/camera/xr_video_bridge/tests/test_support.hpp`
- Create: `gear_sonic/camera/xr_video_bridge/tests/test_main.cpp`
- Create: `gear_sonic/camera/xr_video_bridge/tests/test_latest_value_slot.cpp`
- Create: `gear_sonic/camera/xr_video_bridge/Makefile`

- [ ] Add a minimal dependency-free test runner with `TEST_CASE`, `CHECK`, and `CHECK_EQ` macros. Add tests proving that an empty slot yields no value, the first publish is consumable, a second publish replaces an unconsumed first value, and `close()` wakes a waiting consumer.

- [ ] Add the constants and shared types shown above. Implement `LatestValueSlot<T>` with one mutex, one condition variable, `std::optional<T>`, an overwrite counter, and these operations:

```cpp
bool publish(T value); // false after close; increments overwrite count on replacement
std::optional<T> take_latest(std::chrono::milliseconds timeout);
void close();
std::uint64_t overwritten() const;
```

- [ ] Add Makefile targets that initially compile only the hardware-independent test binary:

```make
.PHONY: native-tests clean
native-tests: build/native_tests
	./build/native_tests
```

Use `-std=c++17 -Wall -Wextra -Wpedantic -Werror -pthread`. Keep `pkg-config` dependency groups separate so native tests that do not use GStreamer can build without the Jetson plugin.

- [ ] Run the failing test before implementing `LatestValueSlot`, then run it again after implementation.

```bash
make -C gear_sonic/camera/xr_video_bridge native-tests
```

Expected final output: all newest-only slot tests pass.

- [ ] Commit.

```bash
git add gear_sonic/camera/xr_video_bridge
git commit -m "test: establish XR bridge native harness"
```

## Task 2: Implement Fragment-Safe XR Command Parsing

**Files:**

- Create: `gear_sonic/camera/xr_video_bridge/xr_control_protocol.hpp`
- Create: `gear_sonic/camera/xr_video_bridge/xr_control_protocol.cpp`
- Create: `gear_sonic/camera/xr_video_bridge/tests/test_xr_control_protocol.cpp`
- Create: `gear_sonic/camera/xr_video_bridge/tests/compare_xr_control_capture.cpp`
- Modify: `gear_sonic/camera/xr_video_bridge/Makefile`

- [ ] Build canonical byte fixtures for v1.1.1 `OPEN_CAMERA` and empty `CLOSE_CAMERA`. The fixture builder must write the four-byte outer length in network byte order and all inner signed 32-bit values in little-endian order. Keep the fixture builder in test code so production parsing is not used to test itself.

Also build `build/compare_xr_control_capture`, a small test utility that independently constructs the canonical request for a supplied expected IPv4 address and byte-compares it with one raw outer frame read from a file. On mismatch, print the first differing offset and both byte values.

- [ ] Add failing accumulator tests that feed:

  - each fixture at every possible single split boundary;
  - each fixture one byte at a time;
  - `OPEN_CAMERA || CLOSE_CAMERA` in one chunk;
  - a partial frame followed by multiple complete frames;
  - zero, less-than-eight, greater-than-65,536, and accumulator-overflow lengths.

- [ ] Add failing body/parser tests for negative-equivalent lengths, integer overflow, truncated command/data, command length over 64, data length over 4,096, invalid UTF-8, trailing bytes, bad magic/version, and unknown commands. Confirm oversized outer lengths are rejected before a body allocation.

- [ ] Add one-field-at-a-time profile rejection tests for camera, dimensions, requested FPS, requested bitrate, HEVC, render mode, port, request IP, command peer, and allowlisted IP. Add the matching exact-profile acceptance test.

- [ ] Implement a bounded `ControlFrameAccumulator` with these invariants:

```text
outer body length: 8..65,536 bytes
total retained accumulator: <=131,080 bytes
command length: 0..64 bytes
data length: 0..4,096 bytes
exact body and payload consumption
```

Use checked size arithmetic before adding offsets. Protocol errors return a typed error that the server can log and use to close the client; do not attempt stream resynchronization.

- [ ] Implement explicit `parse_command_body`, `parse_camera_request`, and `validate_pico4u_v1_1_1`. Keep peer/allowlist validation separate from syntax parsing. `CLOSE_CAMERA` accepts only an empty payload.

- [ ] Run the focused tests, then all native tests.

```bash
make -C gear_sonic/camera/xr_video_bridge test-xr-control-protocol
make -C gear_sonic/camera/xr_video_bridge native-tests
```

- [ ] Compare the fixture field ordering against the pinned local Unity call site without editing that checkout.

```bash
git -C /home/jihun/work/XRoboToolkit-Unity-Client show v1.1.1:Assets/Scripts/UI/UICameraCtrl.cs | sed -n '150,215p'
git -C /home/jihun/work/XRoboToolkit-Unity-Client show v1.1.1:Assets/StreamingAssets/video_source.yml | sed -n '1,120p'
```

- [ ] Commit.

```bash
git add gear_sonic/camera/xr_video_bridge
git commit -m "feat: parse XRoboToolkit control stream safely"
```

## Task 3: Decode GR00T Frames with Correct Color and SBS Layout

**Files:**

- Create: `gear_sonic/camera/xr_video_bridge/gr00t_frame_decoder.hpp`
- Create: `gear_sonic/camera/xr_video_bridge/gr00t_frame_decoder.cpp`
- Create: `gear_sonic/camera/xr_video_bridge/tests/test_gr00t_frame_decoder.cpp`
- Modify: `gear_sonic/camera/xr_video_bridge/Makefile`

- [ ] Add failing synthetic color tests for both wire variants:

  1. Make saturated RGB red, green, and blue patches, pass the RGB matrix directly to `cv::imencode`, base64 it, and pack it as a MessagePack string under `images["ego_view"]`. This reproduces the legacy RealSense bug. After bridge decode, the displayed BGR output must have the intended dominant channel within a JPEG tolerance.
  2. Encode a normal BGR color chart and pack the JPEG as MessagePack binary. It must decode without the corrective red/blue exchange.

- [ ] Add failing shape and payload tests for missing `timestamps`, missing mount, invalid/non-finite timestamp, missing image, invalid base64, invalid JPEG, decoded dimensions other than `640x480`, and malformed MessagePack. Include a payload with optional `ego_view_depth` and prove it is ignored.

- [ ] Add failing temporal tests using an injected wall clock. Reject timestamps that are missing, non-monotonic, more than 50 ms in the future, or older than 250 ms. Allow a publisher restart only after explicitly resetting decoder session state; do not silently accept regression within one open session.

- [ ] Implement `Gr00tFrameDecoder` so it:

  - unpacks only the requested mount;
  - distinguishes MessagePack string from binary before decoding;
  - applies exactly one `cv::cvtColor(..., cv::COLOR_RGB2BGR)` correction to the legacy string path after `imdecode`;
  - performs no channel exchange for binary JPEG;
  - resizes `640x480` to `1080x810` with `cv::INTER_LINEAR`;
  - duplicates the result horizontally into a contiguous `CV_8UC3` `2160x810` matrix;
  - returns a typed drop reason rather than throwing through the receive loop.

- [ ] Verify the two `1080x810` halves are pixel-identical, input and per-eye aspect ratios are 4:3, output is contiguous, and sequence/source metadata are preserved.

- [ ] Run tests.

```bash
make -C gear_sonic/camera/xr_video_bridge test-gr00t-frame-decoder
make -C gear_sonic/camera/xr_video_bridge native-tests
```

- [ ] Commit.

```bash
git add gear_sonic/camera/xr_video_bridge
git commit -m "feat: decode GR00T camera frames for XR"
```

## Task 4: Add the Newest-Only ZMQ Subscriber

**Files:**

- Create: `gear_sonic/camera/xr_video_bridge/gr00t_subscriber.hpp`
- Create: `gear_sonic/camera/xr_video_bridge/gr00t_subscriber.cpp`
- Create: `gear_sonic/camera/xr_video_bridge/tests/test_gr00t_subscriber.cpp`
- Modify: `gear_sonic/camera/xr_video_bridge/Makefile`

- [ ] Add a localhost/in-process PUB-SUB test that overproduces numbered valid MessagePack frames faster than the consumer can decode. Assert the consumer observes the newest available sequence, never a growing FIFO, and reports overwrites.

- [ ] Add shutdown, publisher-absent, malformed-frame-continuation, and publisher-resume tests. Tests must use finite poll deadlines and must not hang when the publisher disappears.

- [ ] Implement `Gr00tSubscriber` with a dedicated receive thread and these socket settings before `connect()`:

```cpp
socket.set(zmq::sockopt::subscribe, "");
socket.set(zmq::sockopt::conflate, 1);
socket.set(zmq::sockopt::rcvhwm, 1);
socket.set(zmq::sockopt::linger, 0);
socket.set(zmq::sockopt::maxmsgsize, kMaxGr00tMessageBytes);
```

Poll with a bounded interval so stop requests are observed. Receive complete MessagePack messages, invoke `Gr00tFrameDecoder`, and publish only valid fresh frames into `LatestValueSlot<DecodedFrame>`. The subscriber must never open or import RealSense.

- [ ] Expose bounded counters for received, decoded, stale-dropped, decode-dropped, and overwritten. Rate-limit repeated failure logs in the later app layer; the subscriber itself returns structured events.

- [ ] Run tests.

```bash
make -C gear_sonic/camera/xr_video_bridge test-gr00t-subscriber
make -C gear_sonic/camera/xr_video_bridge native-tests
```

- [ ] Commit.

```bash
git add gear_sonic/camera/xr_video_bridge
git commit -m "feat: subscribe to newest GR00T camera frame"
```

## Task 5: Implement Deadline-Bounded Video TCP Delivery

**Files:**

- Create: `gear_sonic/camera/xr_video_bridge/bounded_tcp_sender.hpp`
- Create: `gear_sonic/camera/xr_video_bridge/bounded_tcp_sender.cpp`
- Create: `gear_sonic/camera/xr_video_bridge/tests/test_bounded_tcp_sender.cpp`
- Modify: `gear_sonic/camera/xr_video_bridge/Makefile`

- [ ] Define a narrow injected socket/clock seam so tests can script connect results, partial writes, `EAGAIN`, poll readiness, and time advancement without real delays. Production uses POSIX nonblocking sockets; tests use `FakeSocketOps` and `FakeClock`.

- [ ] Add failing tests for:

  - video destination forced to the allowlisted command peer at port 12345;
  - a two-second nonblocking connect deadline;
  - `TCP_NODELAY` and a bounded `SO_SNDBUF` being configured;
  - four-byte big-endian access-unit length followed by the complete payload;
  - partial prefix and payload writes;
  - drop before send when source age is over 250 ms;
  - an absolute send deadline of `min(send_start + 100 ms, source_time + 350 ms)`;
  - socket closure after a partial packet misses its deadline;
  - no reuse of a stream after truncated framing;
  - a stalled receiver causing bounded disconnect, not queued access units.

- [ ] Implement `BoundedTcpSender` with a worker consuming `LatestValueSlot<EncodedAccessUnit>`. It must re-check age immediately before the first byte, write prefix and access unit under one absolute deadline, and increment transmitted only after the kernel has accepted every byte.

Reject empty access units and units larger than `kMaxAccessUnitBytes` before constructing the prefix or entering the send loop.

- [ ] Convert source wall time to age using a sampled wall/steady-clock relationship. Do not compare a wall timestamp directly to `steady_clock`. Detect unreasonable wall-clock jumps and close the current stream rather than claiming the 350 ms invariant.

- [ ] Use `poll()`/`ppoll()` for writable readiness and handle `EINTR`; treat zero writes and fatal socket errors as session failure. Cap the socket send buffer to a documented small multiple of the largest expected access unit, and verify the effective value with `getsockopt` for diagnostics.

- [ ] Add a real loopback slow-receiver integration test in the same test binary: accept a video connection without draining it, continuously replace large synthetic access units for at least five seconds, and sample process RSS plus slot depth. Assert the sender disconnects on its deadline, slot depth never exceeds one, and RSS growth remains within a fixed test tolerance after startup warmup.

- [ ] Run tests, including sanitizers.

```bash
make -C gear_sonic/camera/xr_video_bridge test-bounded-tcp-sender
make -C gear_sonic/camera/xr_video_bridge sanitize-tests
```

- [ ] Commit.

```bash
git add gear_sonic/camera/xr_video_bridge
git commit -m "feat: bound XR video socket latency"
```

## Task 6: Build the Jetson H.264 Pipeline Contract

**Files:**

- Create: `gear_sonic/camera/xr_video_bridge/h264_pipeline.hpp`
- Create: `gear_sonic/camera/xr_video_bridge/h264_pipeline.cpp`
- Create: `gear_sonic/camera/xr_video_bridge/tests/test_h264_pipeline_contract.cpp`
- Create: `gear_sonic/camera/xr_video_bridge/tests/test_h264_pipeline_jetson.cpp`
- Modify: `gear_sonic/camera/xr_video_bridge/Makefile`

- [ ] First implement a pure pipeline-description builder and tests that assert every production constant and bounded property appears. The non-preview production description must contain no ordinary `queue` element and must negotiate exactly:

```text
appsrc name=source is-live=true block=false format=time max-buffers=1 leaky-type=downstream
  caps=video/x-raw,format=BGR,width=2160,height=810,framerate=15/1 !
videoconvert !
nvvidconv ! video/x-raw(memory:NVMM),format=NV12,width=2160,height=810,framerate=15/1 !
nvv4l2h264enc maxperf-enable=1 bitrate=4000000 num-B-Frames=0
  insert-sps-pps=true iframeinterval=15 idrinterval=15 !
h264parse config-interval=-1 !
video/x-h264,stream-format=byte-stream,alignment=au !
appsink name=encoded_sink sync=false emit-signals=true max-buffers=1 drop=true
```

Do not assume a property merely because it is in the string: startup must query the installed elements/properties and fail if the installed JetPack cannot provide equivalent leaky appsrc, dropping appsink, H.264 byte-stream/AU, no-B-frame, SPS/PPS, and periodic-IDR behavior.

- [ ] Implement `H264Pipeline` with injected callbacks and GStreamer ownership isolated behind the header. Set buffer PTS from a monotonic frame identifier and maintain a bounded PTS-to-`FrameMetadata` map of at most the pipeline's in-flight capacity. Reject stale frames again immediately before `gst_app_src_push_buffer`.

- [ ] In the appsink callback, require one complete sample with negotiated `byte-stream, alignment=au`, map it, copy one access unit into `LatestValueSlot<EncodedAccessUnit>`, and return immediately. Never perform TCP I/O in a GStreamer callback.

- [ ] Make each stream start from a fresh encoder and suppress transmission until one access unit contains Annex-B SPS (NAL type 7), PPS (8), and IDR (5). Add a small Annex-B NAL scanner with tests for three- and four-byte start codes, multiple NALs, truncation, and missing startup NALs.

Restart the hardware test pipeline within one process and assert the first transmitted access unit of both sessions independently contains SPS, PPS, and IDR. Continue for at least 31 submitted source frames and verify an IDR interval no greater than 15 frames.

- [ ] Add the optional preview as a tee branch with an explicit one-buffer leaky queue. Verify preview-disabled is the service default and cannot backpressure encoding.

- [ ] Run the pure contract tests on the development PC.

```bash
make -C gear_sonic/camera/xr_video_bridge test-h264-pipeline-contract
make -C gear_sonic/camera/xr_video_bridge native-tests
```

- [ ] On pc2, run the hardware test with a generated color chart. It must write length-prefixed access units plus a stripped `/tmp/gr00t_xr_video_bridge_test.h264`, assert SPS/PPS/IDR in the first sent access unit, and decode the stream with FFmpeg.

```bash
gst-inspect-1.0 nvv4l2h264enc
make -C gear_sonic/camera/xr_video_bridge jetson-tests
ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate -of default=nw=1 /tmp/gr00t_xr_video_bridge_test.h264
ffmpeg -v error -i /tmp/gr00t_xr_video_bridge_test.h264 -frames:v 1 -f null -
```

Expected: H.264, 2160x810, 15/1; decoder commands exit zero. Inspect the decoded color chart to confirm red, green, and blue dominance across the complete JPEG-to-H.264 path.

- [ ] Commit.

```bash
git add gear_sonic/camera/xr_video_bridge
git commit -m "feat: encode bounded XR H264 access units"
```

## Task 7: Orchestrate Control Sessions and Process Lifecycle

**Files:**

- Create: `gear_sonic/camera/xr_video_bridge/control_server.hpp`
- Create: `gear_sonic/camera/xr_video_bridge/control_server.cpp`
- Create: `gear_sonic/camera/xr_video_bridge/bridge_app.hpp`
- Create: `gear_sonic/camera/xr_video_bridge/bridge_app.cpp`
- Create: `gear_sonic/camera/xr_video_bridge/main.cpp`
- Create: `gear_sonic/camera/xr_video_bridge/tests/test_bridge_app.cpp`
- Modify: `gear_sonic/camera/xr_video_bridge/Makefile`

- [ ] Add state-machine tests using fake subscriber, pipeline, and sender interfaces. Cover:

  - listener starts without camera or encoder ownership;
  - valid `OPEN_CAMERA` starts exactly one session;
  - second open while active is rejected;
  - `CLOSE_CAMERA` stops the session;
  - fragmented/coalesced commands traverse the real accumulator;
  - command disconnect, video failure, GStreamer error, SIGINT, and SIGTERM stop the active session but do not crash cleanup;
  - missing ZMQ publisher leaves command listening active;
  - a subsequent valid open works after a failed session.

- [ ] Implement one bounded accumulator per accepted command client. Bind the configured address, accept only an IPv4 peer matching the allowlist, and set finite poll timeouts. An invalid command or protocol error closes that client. Do not allow one idle command connection to block process shutdown indefinitely.

- [ ] Implement the open sequence in this order:

```text
validate request and peer
connect BoundedTcpSender to peer:12345
create fresh H264Pipeline
start Gr00tSubscriber and frame pump
log requested profile and effective 15 FPS / 4,000,000 bps
```

Implement stop in reverse order: stop frame intake, close the decoded slot, stop pipeline, close encoded slot, join sender, close video socket. Make stop idempotent and safe after partial startup.

- [ ] Add a frame-pump worker that calls `take_latest`, rechecks 250 ms age immediately before appsrc, and submits each source sequence at most once. `Gr00tFrameDecoder` performs the earlier age check before decode/resize. A publisher outage must freeze output rather than repeat the cached last frame.

- [ ] Aggregate counters and emit bounded-rate periodic metrics: received, decoded, stale-dropped, decode-dropped, overwritten before encoding, encoded, overwritten before network, network-dropped, transmitted, and source-to-send age percentiles. Avoid per-frame logging.

For the first successfully validated `OPEN_CAMERA` per command connection, log one bounded `control_frame_hex=` field containing the exact raw outer frame. This exists only to compare the deployed v1.1.1 command against the independent canonical test fixture; malformed or oversized frames must never be hex-logged.

- [ ] Implement strict CLI parsing:

```text
--listen ADDRESS             required
--allowed-headset-ip IPV4    required
--gr00t-zmq ENDPOINT         default tcp://127.0.0.1:5555
--mount NAME                 default ego_view
--preview                    optional
--help
```

Reject unknown flags, non-IPv4 allowlist values, malformed listen endpoints, listen ports other than fixed command port 13579, and positional arguments. Do not add FPS, bitrate, source-age, send-deadline, output-size, codec, command-port, or output-port overrides. Tests that need ephemeral ports inject an already-created listener rather than weakening production CLI validation.

- [ ] Install async-signal-safe handlers that only set a flag or write a self-pipe; perform joins and GStreamer cleanup in normal control flow.

- [ ] Run native state-machine and aggregate tests.

```bash
make -C gear_sonic/camera/xr_video_bridge test-bridge-app
make -C gear_sonic/camera/xr_video_bridge native-tests
```

- [ ] Commit.

```bash
git add gear_sonic/camera/xr_video_bridge
git commit -m "feat: serve XR video bridge sessions"
```

## Task 8: Finalize Reproducible Native Builds

**Files:**

- Modify: `gear_sonic/camera/xr_video_bridge/Makefile`

- [ ] Split compilation into reusable object groups and expose these targets:

```text
all                 build/xr_video_bridge/gr00t_xr_video_bridge
native-tests        all hardware-independent unit/integration tests
sanitize-tests      native tests under ASan and UBSan
jetson-tests        encoder/plugin integration tests
clean               only generated files under the bridge build directory
```

Use `pkg-config` for `opencv4`, `libzmq`, `msgpack`, `gstreamer-1.0`, `gstreamer-app-1.0`, and `openssl` as actually needed. Keep include/link output visible in verbose mode. The Makefile must contain no `/usr/local/zed`, `sl/Camera.hpp`, `-lsl_zed`, CUDA include assumption, or dependency on either XR reference clone.

- [ ] Make dependency failures actionable by checking `pkg-config --exists` and naming the missing Ubuntu development package. Use dependency generation (`-MMD -MP`) so header changes rebuild affected objects.

- [ ] Verify no upstream/ZED coupling and run local quality gates.

```bash
rg -n 'Camera\.hpp|sl_zed|/zed|XRoboToolkit-Orin-Video-Sender|XRoboToolkit-Unity-Client' gear_sonic/camera/xr_video_bridge
make -C gear_sonic/camera/xr_video_bridge clean
make -C gear_sonic/camera/xr_video_bridge native-tests
make -C gear_sonic/camera/xr_video_bridge sanitize-tests
```

Expected: `rg` returns no build/runtime coupling; an explanatory comment may name a pinned reference only if necessary.

- [ ] On pc2, build the production binary and verify dynamic linkage.

```bash
make -C gear_sonic/camera/xr_video_bridge clean all jetson-tests
ldd build/xr_video_bridge/gr00t_xr_video_bridge
```

- [ ] Commit.

```bash
git add gear_sonic/camera/xr_video_bridge/Makefile
git commit -m "build: make XR bridge reproducible"
```

## Task 9: Detect Fresh Exporter Frames and Episode Progression

**Files:**

- Create: `gear_sonic/utils/data_collection/camera_freshness.py`
- Create: `gear_sonic/tests/test_composed_camera_client_freshness.py`
- Create: `gear_sonic/tests/test_camera_freshness.py`
- Create: `gear_sonic/tests/test_run_data_exporter_camera_freshness.py`
- Create: `gear_sonic/scripts/validate_episode_camera_progression.py`
- Create: `gear_sonic/tests/test_validate_episode_camera_progression.py`
- Modify: `gear_sonic/scripts/run_data_exporter.py`
- Modify: `gear_sonic/camera/composed_camera.py`

- [ ] Add failing Python tests for a `CameraFreshnessMonitor` that distinguishes a newly received source timestamp from a repeated cached message. Cover strictly increasing values, duplicate cached values, regressions, non-finite/future/stale values, gaps over 500 ms, a ten-second observation window below 12 new messages per second, and a valid 15 FPS window.

Use explicit events rather than inferring freshness from image non-emptiness or timestamp equality:

```python
@dataclass(frozen=True)
class CameraFreshnessEvent:
    source_timestamp: float
    received_monotonic: float
    is_new_message: bool
    source_age_seconds: float
```

- [ ] Add `ComposedCameraClientSensor.read_with_status()` while preserving the existing `read()` return contract for every current caller. Factor both methods through one internal receive operation and return an immutable status containing `is_new_message` plus the exact client monotonic receive time. Add tests with a fake ZMQ receive method proving a network message is marked new, a cached fallback is not, and calling either public API performs at most one receive poll.

- [ ] Implement the monitor with injected wall and monotonic clocks. Consume the explicit client read status. A cached message produces `is_new_message=False`; a network-received message with a duplicate or regressed source timestamp is a validation failure rather than being relabeled cached. Only network-received, strictly advancing timestamps update new-message cadence statistics.

- [ ] Wire `read_with_status()` and the monitor at the actual exporter read boundary in `run_data_exporter.py`. Preserve the returned image message and collection timing. Record each new `timestamps["ego_view"]` with its exact client monotonic receive time, expose a compact end-of-episode summary, and rate-limit warnings for stale, duplicate, regressed, or gapped frames.

Do not change `ComposedCameraClientSensor` caching behavior as part of this feature; expose and correctly identify that existing behavior.

Add a focused exporter integration test with a fake image subscriber returning one message followed by the same cached message. Assert the collector returns both unchanged while the freshness monitor counts only one new source frame. Then return an advancing timestamp and assert the second new-message event is recorded.

- [ ] Persist camera evidence without changing the model-facing LeRobot feature schema. In lockstep immediately after each successful `data_exporter.add_frame(frame_data)`, append one compact JSON object to:

```text
<dataset-root>/meta/camera_freshness/episode_<six-digit-index>.jsonl
```

Each line must contain schema version, episode index, episode frame index, mount, source wall timestamp, client receive monotonic time, exporter observation wall/monotonic times, computed source age, `is_new_message`, and current new-message gap. Flush on episode save/discard and process cleanup; reject an attempt to append a different episode to an open writer. Add temporary-directory tests proving line count and frame indices match exported-frame calls, including cached frames, and that one episode cannot overwrite another.

- [ ] Add a progression validator with a pure library entry point and CLI:

```text
python gear_sonic/scripts/validate_episode_camera_progression.py \
  --dataset-root PATH --episode-index N --mount ego_view \
  --max-source-age-ms 250 --max-new-message-gap-ms 500 \
  --minimum-new-fps 12 --minimum-changing-frame-ratio 0.5
```

The script must read the exact JSONL sidecar for the requested episode, verify its schema and one-to-one frame-index coverage, assert strict source-timestamp advancement for new-message events, and compute temporal progression from downscaled luma-frame mean absolute differences in the corresponding saved video. The threshold itself is a validation-tool argument, not a bridge freshness override. It must reject missing/truncated sidecars and an episode made from one repeated valid image.

- [ ] Add fixture tests for: corrupt/nonempty images, frozen repeated frames, advancing timestamps with frozen imagery, regressed timestamps, excessive gaps, correct RGB temporal color target, and a passing moving target. Keep fixtures small and generate them inside pytest temporary directories.

- [ ] Run focused tests and the existing exporter-related suite.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_composed_camera_client_freshness.py \
  gear_sonic/tests/test_camera_freshness.py \
  gear_sonic/tests/test_run_data_exporter_camera_freshness.py \
  gear_sonic/tests/test_validate_episode_camera_progression.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_run_camera_viewer.py
```

- [ ] Run Python style checks on only the touched Python files.

```bash
ruff check gear_sonic/utils/data_collection/camera_freshness.py \
  gear_sonic/scripts/validate_episode_camera_progression.py \
  gear_sonic/scripts/run_data_exporter.py \
  gear_sonic/camera/composed_camera.py \
  gear_sonic/tests/test_composed_camera_client_freshness.py \
  gear_sonic/tests/test_camera_freshness.py \
  gear_sonic/tests/test_run_data_exporter_camera_freshness.py \
  gear_sonic/tests/test_validate_episode_camera_progression.py
ruff format --check gear_sonic/utils/data_collection/camera_freshness.py \
  gear_sonic/scripts/validate_episode_camera_progression.py \
  gear_sonic/scripts/run_data_exporter.py \
  gear_sonic/camera/composed_camera.py \
  gear_sonic/tests/test_composed_camera_client_freshness.py \
  gear_sonic/tests/test_camera_freshness.py \
  gear_sonic/tests/test_run_data_exporter_camera_freshness.py \
  gear_sonic/tests/test_validate_episode_camera_progression.py
```

- [ ] Commit.

```bash
git add gear_sonic/utils/data_collection/camera_freshness.py \
  gear_sonic/scripts/validate_episode_camera_progression.py \
  gear_sonic/scripts/run_data_exporter.py \
  gear_sonic/camera/composed_camera.py \
  gear_sonic/tests/test_composed_camera_client_freshness.py \
  gear_sonic/tests/test_camera_freshness.py \
  gear_sonic/tests/test_run_data_exporter_camera_freshness.py \
  gear_sonic/tests/test_validate_episode_camera_progression.py
git commit -m "test: detect stale episode camera frames"
```

## Task 10: Add RealSense, Bridge, and systemd Installation

**Files:**

- Create: `install_scripts/install_xr_video_bridge.sh`
- Create: `systemd/gr00t_xr_video_bridge.service.in`
- Modify: `install_scripts/install_camera_server.sh`
- Modify: `docs/source/tutorials/data_collection.md`
- Create: `gear_sonic/tests/test_install_xr_video_bridge.py`

- [ ] Extend `install_camera_server.sh` with an idempotent `--with-realsense` option. After the venv is created, install the binding with uv targeting the venv interpreter explicitly:

```bash
uv pip install \
  --python "$REPO_ROOT/.venv_camera/bin/python" \
  pyrealsense2
"$REPO_ROOT/.venv_camera/bin/python" -c \
  'import pyrealsense2 as rs; print(rs.__version__ if hasattr(rs, "__version__") else "pyrealsense2 import OK")'
```

Do not call `python -m pip` because this uv-created environment may not contain pip. Preserve the existing non-RealSense path when the flag is absent.

- [ ] In `test_install_xr_video_bridge.py`, write failing subprocess tests for `--dry-run`. Verify paths containing spaces are preserved as single arguments, missing/invalid headset IP fails, an invalid listen port fails, and service rendering substitutes every token without leaving an `@...@` placeholder. The tests must point the installer at fake command binaries under pytest's temporary directory so they never call apt, make, sudo, or systemctl.

- [ ] Implement `install_xr_video_bridge.sh` with:

```text
--allowed-headset-ip IPV4    required for service installation
--listen ADDRESS             default 0.0.0.0:13579
--gr00t-zmq ENDPOINT         default tcp://127.0.0.1:5555
--mount NAME                 default ego_view
--install-service            render/install/enable systemd unit
--dry-run                    print mutations without performing them
```

The installer must:

  - derive the repository root from its own location;
  - install exactly `build-essential`, `pkg-config`, `libgstreamer1.0-dev`, `libgstreamer-plugins-base1.0-dev`, `libopencv-dev`, `libzmq3-dev`, `libmsgpack-dev`, and `libssl-dev`;
  - verify `gst-inspect-1.0 nvv4l2h264enc`, `nvvidconv`, `h264parse`, `appsrc`, and `appsink` plus required properties;
  - build from the GR00T checkout with `make -C gear_sonic/camera/xr_video_bridge all`;
  - write `build/xr_video_bridge/build-manifest.txt` containing GR00T SHA, JetPack release, GStreamer version, compiler version, and `dpkg-query` versions for every declared package;
  - optionally render the service using the actual user and repository path;
  - run `systemctl daemon-reload` and enable/start only when explicitly passed `--install-service`.

- [ ] Create a service template with `After=network-online.target composed_camera_server.service` but no hard `Requires=` dependency, because a missing publisher must not kill the listener. Include restart-on-failure, finite stop timeout, the strict runtime flags, and no ZED or direct camera access. Use absolute paths substituted by the installer.

- [ ] Update `docs/source/tutorials/data_collection.md` with a pc2 runbook covering:

  - why `.venv_camera/bin/python -m pip` fails and the exact uv alternative;
  - `pyrealsense2` import and `rs-enumerate-devices` checks;
  - camera viewer verification;
  - composed-camera service installation/status/log commands;
  - bridge installation and service status/log commands;
  - firewall/LAN requirements for TCP 13579 and headset TCP 12345;
  - the PICO4U v1.1.1 selection;
  - manual foreground commands for troubleshooting;
  - simultaneous episode collection and progression validation;
  - explicit warning that only `composed_camera` opens RealSense.

- [ ] Validate scripts and the rendered unit without installing anything on the development PC.

```bash
bash -n install_scripts/install_camera_server.sh
bash -n install_scripts/install_xr_video_bridge.sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_install_xr_video_bridge.py
ruff check gear_sonic/tests/test_install_xr_video_bridge.py
ruff format --check gear_sonic/tests/test_install_xr_video_bridge.py
install_scripts/install_xr_video_bridge.sh --dry-run \
  --allowed-headset-ip 192.0.2.10
rg -n '@[A-Z0-9_]+@' systemd/gr00t_xr_video_bridge.service.in
```

The template should contain tokens; the dry-run rendered output must not. Add an automated assertion for that distinction.

- [ ] Commit.

```bash
git add install_scripts/install_xr_video_bridge.sh \
  systemd/gr00t_xr_video_bridge.service.in \
  install_scripts/install_camera_server.sh \
  gear_sonic/tests/test_install_xr_video_bridge.py \
  docs/source/tutorials/data_collection.md
git commit -m "docs: install GR00T XR camera bridge"
```

## Task 11: Run the Local Pre-Hardware Quality Gate

**Files:** only fixes required by failed checks in Tasks 1-10

- [ ] Run all hardware-independent bridge tests from a clean build.

```bash
make -C gear_sonic/camera/xr_video_bridge clean
make -C gear_sonic/camera/xr_video_bridge native-tests
make -C gear_sonic/camera/xr_video_bridge sanitize-tests
```

- [ ] Run all touched Python tests and static checks.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_composed_camera_client_freshness.py \
  gear_sonic/tests/test_camera_freshness.py \
  gear_sonic/tests/test_run_data_exporter_camera_freshness.py \
  gear_sonic/tests/test_validate_episode_camera_progression.py \
  gear_sonic/tests/test_install_xr_video_bridge.py \
  gear_sonic/tests/test_run_camera_viewer.py
ruff check gear_sonic/utils/data_collection/camera_freshness.py \
  gear_sonic/scripts/validate_episode_camera_progression.py \
  gear_sonic/scripts/run_data_exporter.py \
  gear_sonic/camera/composed_camera.py \
  gear_sonic/tests/test_composed_camera_client_freshness.py \
  gear_sonic/tests/test_camera_freshness.py \
  gear_sonic/tests/test_run_data_exporter_camera_freshness.py \
  gear_sonic/tests/test_validate_episode_camera_progression.py
ruff format --check gear_sonic/utils/data_collection/camera_freshness.py \
  gear_sonic/scripts/validate_episode_camera_progression.py \
  gear_sonic/scripts/run_data_exporter.py \
  gear_sonic/camera/composed_camera.py \
  gear_sonic/tests/test_composed_camera_client_freshness.py \
  gear_sonic/tests/test_camera_freshness.py \
  gear_sonic/tests/test_run_data_exporter_camera_freshness.py \
  gear_sonic/tests/test_validate_episode_camera_progression.py
ruff check gear_sonic/tests/test_install_xr_video_bridge.py
ruff format --check gear_sonic/tests/test_install_xr_video_bridge.py
bash -n install_scripts/install_camera_server.sh
bash -n install_scripts/install_xr_video_bridge.sh
```

- [ ] Audit the invariants mechanically.

```bash
rg -n '250|100|350|4000000|15|2160|810|12345' \
  gear_sonic/camera/xr_video_bridge
rg -n -- '--encoder-fps|--encoder-bitrate|--max-source-age|--send-deadline|--video-port' \
  gear_sonic/camera/xr_video_bridge install_scripts systemd
rg -n 'sl/Camera|sl_zed|/usr/local/zed|pyrealsense2|rs2::|librealsense' \
  gear_sonic/camera/xr_video_bridge
git diff --check
git status --short
```

Expected: invariant values appear in constants/tests; runtime-override and direct-camera searches return no bridge matches; diff check is clean.

- [ ] Fix failures one at a time with a regression test first. If fixes are needed, amend only the task commit responsible for that behavior or make one clearly scoped follow-up commit. Do not bundle unrelated cleanup.

- [ ] Record the exact successful command output in `build/xr_video_bridge/local-verification.txt`; because `build/` is generated, do not commit that log.

## Task 12: Install and Validate on pc2 with the PICO4U v1.1.1 Headset

**Files:** only targeted fixes revealed by Jetson/headset integration

- [ ] Transfer or check out the tested GR00T feature commit on pc2. Record both hashes before installation.

```bash
cd /home/unitree/GR00T-WholeBodyControl
git rev-parse HEAD
git status --short
```

Do not overwrite pc2-local work. If the checkout is dirty, stop and reconcile it before installing.

- [ ] Install the camera environment and confirm pyrealsense2 uses the intended interpreter.

```bash
cd /home/unitree/GR00T-WholeBodyControl
install_scripts/install_camera_server.sh --with-realsense
.venv_camera/bin/python -c 'import sys, pyrealsense2; print(sys.executable); print("pyrealsense2 import OK")'
rs-enumerate-devices
```

Expected interpreter: `/home/unitree/GR00T-WholeBodyControl/.venv_camera/bin/python`. Confirm exactly one connected RealSense device.

- [ ] Start only `composed_camera`, then validate its publisher. If the repository's camera installer did not install the service, follow its documented service-generation step first.

```bash
sudo systemctl restart composed_camera_server.service
sudo systemctl status --no-pager composed_camera_server.service
journalctl -u composed_camera_server.service -n 100 --no-pager
source .venv_camera/bin/activate
python gear_sonic/scripts/run_camera_viewer.py --camera-host 127.0.0.1 --camera-port 5555
```

Close the viewer after confirming a fresh `480x640x3` `ego_view`. Verify with `lsof`/`pgrep` that no bridge process has opened a RealSense device.

- [ ] Read and validate the actual headset IPv4 address, then install the bridge.

```bash
read -r -p "PICO4U IPv4 address: " XR_HEADSET_IP
export XR_HEADSET_IP
python3 -c 'import ipaddress, os; print(ipaddress.IPv4Address(os.environ["XR_HEADSET_IP"]))'
install_scripts/install_xr_video_bridge.sh \
  --allowed-headset-ip "$XR_HEADSET_IP" \
  --listen 0.0.0.0:13579 \
  --gr00t-zmq tcp://127.0.0.1:5555 \
  --mount ego_view \
  --install-service
```

The installer must refuse an empty or invalid address.

- [ ] Verify service and ports before opening Unity.

```bash
sudo systemctl status --no-pager gr00t_xr_video_bridge.service
journalctl -u gr00t_xr_video_bridge.service -n 100 --no-pager
ss -ltnp | rg ':13579'
gst-inspect-1.0 nvv4l2h264enc >/dev/null
```

- [ ] In XRoboToolkit PICO APK v1.1.1, select `PICO4U`. Confirm logs show the exact request (`VR`, 2160x810, 60 FPS, 20,971,520 bps, HEVC 0, render mode 2, port 12345), the validated peer IP, and effective 15 FPS / 4,000,000 bps.

- [ ] Extract the bounded raw request logged by the bridge and compare every byte, including the outer length, with the independently built v1.1.1 fixture.

```bash
XR_CAPTURE_HEX=$(journalctl -u gr00t_xr_video_bridge.service -o cat --no-pager | sed -n 's/.*control_frame_hex=//p' | tail -1)
test -n "$XR_CAPTURE_HEX"
printf '%s' "$XR_CAPTURE_HEX" | xxd -r -p > /tmp/pico4u_v1_1_1_open_camera.bin
build/xr_video_bridge/compare_xr_control_capture \
  /tmp/pico4u_v1_1_1_open_camera.bin "$XR_HEADSET_IP"
```

Expected: byte-for-byte match. Keep the binary capture with the integration record, not in git.

- [ ] Confirm the headset renders a correctly proportioned view in both eyes. Present a controlled red/green/blue target to confirm channel order. Save a bridge-side stripped H.264 sample and verify it independently:

```bash
ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate \
  -of default=nw=1 /tmp/gr00t_xr_video_bridge_capture.h264
ffmpeg -v error -i /tmp/gr00t_xr_video_bridge_capture.h264 -frames:v 30 -f null -
```

- [ ] Run for ten minutes while periodically recording metrics. Acceptance requires bounded RSS, internal slot depths of zero or one, no completed local handoff over 350 ms source age, no progressive video lag, and no second-camera-owner errors.

- [ ] Exercise failures: stop/restart only the bridge; briefly stop/restart the camera publisher; disconnect/reconnect the headset; send `CLOSE_CAMERA`. The command listener must recover without restarting or interrupting the RealSense service.

- [ ] If integration changes are necessary, reproduce with a failing test where possible, commit a narrow fix, rerun Tasks 11 and 12 from the beginning, and update the build manifest SHA.

## Task 13: Prove Concurrent Episode Collection Remains Fresh

**Files:** no planned source changes; fixes require a failing regression test and a focused commit

- [ ] With `composed_camera` and the XR bridge active, start the normal teleop/deployment stack and exporter from the documented data-collection environments. Use an explicit dataset name so validation targets are unambiguous:

```bash
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "XR bridge RealSense progression test" \
  --dataset-name xr_bridge_realsense_progression \
  --camera-host 192.168.123.164 \
  --camera-port 5555
```

Record at least ten seconds with a controlled moving RGB target while the headset is actively rendering.

- [ ] During the episode, restart only the XR bridge and continue recording across the restart.

```bash
sudo systemctl restart gr00t_xr_video_bridge.service
journalctl -u gr00t_xr_video_bridge.service -n 100 --no-pager
```

- [ ] Validate the saved episode, substituting the actual episode index printed by the exporter:

```bash
python gear_sonic/scripts/validate_episode_camera_progression.py \
  --dataset-root outputs/xr_bridge_realsense_progression \
  --episode-index 0 \
  --mount ego_view \
  --max-source-age-ms 250 \
  --max-new-message-gap-ms 500 \
  --minimum-new-fps 12 \
  --minimum-changing-frame-ratio 0.5
```

If the exporter reports an episode index other than zero, pass that exact integer; do not scan for an arbitrary passing episode.

- [ ] Confirm all collection acceptance checks:

  - new-message source timestamps are strictly increasing;
  - source age is below 250 ms;
  - no new-message gap exceeds 500 ms;
  - at least 12 new messages per second are observed;
  - the RGB target has correct channel order and saved frames demonstrate temporal progression;
  - timestamps and imagery keep progressing across the bridge restart;
  - camera, exporter, and bridge logs contain no device-busy, disconnect, or second-owner errors.

- [ ] Stop the test session normally, retain the verification outputs with the experiment record, and run final repository checks.

```bash
git diff --check
git status --short
git log --oneline --decorate -15
```

- [ ] Request code review against the approved design and this plan. Address only concrete findings, rerun the affected focused test plus Tasks 11-13 as appropriate, and do not claim headset compatibility without the physical v1.1.1 gate.

## Definition of Done

- [ ] All Task 11 local checks pass from a clean build.
- [ ] pc2 builds without ZED and the hardware H.264 stream is Annex-B/AU with startup SPS/PPS/IDR.
- [ ] PICO4U v1.1.1 displays correct, fresh SBS video for ten minutes.
- [ ] No access unit completes local kernel handoff beyond 350 ms source age.
- [ ] The concurrent episode satisfies timestamp, cadence, age, color, and temporal-progression checks across a bridge restart.
- [ ] `composed_camera` remains the sole RealSense owner.
- [ ] Installation is reproducible from the GR00T checkout and build manifest, with no external mutable source dependency.
- [ ] Every implementation commit is focused, `git diff --check` is clean, and the unrelated user file remains untouched.
