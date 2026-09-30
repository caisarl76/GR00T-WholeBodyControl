# GR00T to XRoboToolkit Video Bridge Design

## Purpose

Enable XRoboToolkit Unity Remote Vision to display the Unitree G1 RealSense
camera while GR00T data collection records the same camera stream. The bridge
must not open the RealSense device or change the existing episode image path.

Success requires both of the following at the same time:

- the GR00T episode collector receives fresh
  `observation.images.ego_view` frames; and
- XRoboToolkit displays a fresh, correctly colored camera view through its
  `PICO4U` video source.

## Pinned Compatibility Baseline

The deployed headset application is
[`XRoboToolkit-PICO-1.1.1.apk`](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/tag/v1.1.1),
source commit `9f775b535d781618bd2bb7ef8d6c414c0531387c`.

The bridge reimplements the required wire contracts in the GR00T repository.
It does not build against or copy the upstream Orin sender. The behavior of
[`XRoboToolkit-Orin-Video-Sender`](https://github.com/XR-Robotics/XRoboToolkit-Orin-Video-Sender)
at commit `8bedc7bb225628235f97b6e51bfde5a2a3704031` is a protocol reference only.
In particular, the bridge must not reuse that revision's `network_helper.hpp`,
because it passes each individual `recv()` result to the command parser as if
TCP preserved message boundaries.

Supporting a different APK release is outside this design. A mismatched request
is rejected and logged rather than silently mapped to another profile.

## Existing GR00T Camera Contract

The existing GR00T `composed_camera` process is the only RealSense owner. It
publishes MessagePack payloads through ZMQ on `tcp://*:5555`. Each payload has:

- `timestamps["ego_view"]`: a wall-clock source timestamp generated on pc2;
- `images["ego_view"]`: normally a base64 JPEG string for RealSense; and
- optional `ego_view_depth` timestamp and image entries when depth is enabled.

The configured RealSense source is `640x480`, RGB8, at 15 FPS. GR00T's legacy
base64 path passes that RGB array directly to OpenCV JPEG encoding, whose input
contract is BGR. The JPEG therefore carries red and blue in exchanged channel
positions. A bridge that simply calls `cv::imdecode` and labels the result BGR
would preserve the wrong display colors.

Binary JPEG values have a different contract: they are raw on-device MJPEG
bytes, such as those emitted by an OAK device. OpenCV decodes those bytes into
normal BGR and no corrective channel exchange is required. The wire value type
therefore determines the bridge's color conversion:

| Wire value | Origin | After `cv::imdecode` | Before BGR GStreamer input |
|---|---|---|---|
| MessagePack string | Legacy RGB array encoded as BGR | red/blue exchanged | exchange red/blue once |
| MessagePack binary | On-device standard JPEG | normal BGR | no channel exchange |

The bridge ignores optional depth entries.

## PICO4U v1.1.1 Contract

The v1.1.1 `PICO4U` profile sends one `OPEN_CAMERA` request with these exact
values:

| Field | Required value |
|---|---:|
| camera | `VR` |
| width | `2160` |
| height | `810` |
| fps | `60` |
| bitrate | `20971520` |
| enableMvHevc | `0` |
| renderMode | `2` |
| port | `12345` |

The profile describes two 4:3 eye images side by side. The bridge resizes the
RealSense image from `640x480` to `1080x810`, preserving its 4:3 aspect ratio,
then duplicates it horizontally to produce the required `2160x810` frame.

The stock profile targets a 60 FPS PICO camera. The RealSense publisher is 15
FPS, so the bridge uses a fixed 15 FPS encoder cadence and submits each selected
fresh source frame at most once. Replaced or stale frames may be dropped under
load; the bridge never synthesizes four copies of a frame to fill the 60 FPS
request. The encoder bitrate is fixed at 4,000,000 bps because the upscaled 15
FPS source contains no additional detail that benefits from the profile's 20
Mib/s (`20,971,520` bps) capture bitrate. Every accepted `OPEN_CAMERA` log
states both the requested profile and the fixed effective encoder FPS/bitrate.
Width and height remain exactly `2160x810`, which is the dimension passed to
the v1.1.1 native decoder.

## Architecture

```text
RealSense D435
     | pyrealsense2 (single camera owner)
     v
GR00T composed_camera :5555
     | ZMQ PUB
     +--------------------> episode data collector
     |
     +--------------------> gr00t_xr_video_bridge
                               | bounded TCP command server :13579
                               | newest-only JPEG decode and SBS conversion
                               | bounded Jetson H.264 pipeline
                               | deadline-bounded TCP sender
                               v
                         allowed PICO headset :12345
```

The bridge is an independent ZMQ subscriber. It cannot consume, acknowledge,
or remove messages from the episode collector's independent subscription.

## Repository Ownership and Deliverables

All maintained source and deployment artifacts live in
`GR00T-WholeBodyControl`:

```text
gear_sonic/camera/xr_video_bridge/
    main.cpp
    xr_control_protocol.hpp/.cpp
    gr00t_frame_decoder.hpp/.cpp
    h264_pipeline.hpp/.cpp
    bounded_tcp_sender.hpp/.cpp
    latest_value_slot.hpp
    Makefile
    tests/
install_scripts/install_xr_video_bridge.sh
systemd/gr00t_xr_video_bridge.service.in
```

The installer builds
`build/xr_video_bridge/gr00t_xr_video_bridge` from the current GR00T checkout.
It installs build dependencies, verifies `nvv4l2h264enc`, and can generate a
systemd unit with the actual repository path, Unix user, listen address, and
allowed headset IPv4 address. No mutable external clone is used at build or
runtime.

The installer uses these Ubuntu development packages:

```text
build-essential
pkg-config
libgstreamer1.0-dev
libgstreamer-plugins-base1.0-dev
libopencv-dev
libzmq3-dev
libmsgpack-dev
libssl-dev
```

CUDA and the `nvv4l2h264enc` plugin come from the already installed JetPack
image and are verified rather than reinstalled. The Makefile contains no ZED
include path or `libsl_zed` linkage. The reproducibility boundary is the GR00T
commit, the two pinned XR reference commits, the installed JetPack release, and
the package versions reported by `dpkg-query`; the installer writes those
versions to `build/xr_video_bridge/build-manifest.txt`.

## Command TCP Protocol

### Stream framing

The control connection is a TCP byte stream. The bridge maintains a persistent
byte accumulator per accepted client; one `recv()` may contain a partial frame,
one frame, or several frames.

Each outer frame is:

```text
uint32_be body_length
body[body_length]
```

The parser waits for all four length bytes, validates `body_length` before
allocating or accumulating its body, retains partial data, and extracts every
complete frame before reading again. `body_length` must be between 8 and 65,536
bytes. The total accumulator is bounded to 131,080 bytes; exceeding the bound
closes the command connection.

The body is:

```text
int32_le command_length
command[command_length]          # UTF-8; ASCII commands in v1.1.1
int32_le data_length
data[data_length]
```

`command_length` is limited to 64 bytes and `data_length` to 4,096 bytes.
Negative lengths, integer overflow, truncated fields, invalid UTF-8, and body
bytes remaining after the declared data are protocol errors that close the
client. The only supported commands for v1.1.1 are `OPEN_CAMERA` and
`CLOSE_CAMERA`; an unknown command is rejected and logged.

### Camera request payload

`OPEN_CAMERA` data starts with `CA FE`, protocol version 1, then seven signed
32-bit little-endian integers, followed by one-byte-length-prefixed UTF-8
camera and IP strings. Parsing requires exact payload consumption.

The bridge accepts only the complete v1.1.1 profile shown above. The request IP
must equal both the TCP command peer and the required `--allowed-headset-ip`.
The output target is forced to that peer on port `12345`; it is never an
arbitrary request-controlled destination. `CLOSE_CAMERA` must have an empty
payload.

The production service listens on the pc2 LAN interface or `0.0.0.0:13579`, but
the headset allowlist is mandatory. This is still an unencrypted protocol and
is supported only on the trusted robot/headset LAN.

## End-to-End Freshness Invariant

At no point may the bridge maintain an unbounded FIFO of decoded or encoded
frames. Production freshness values are fixed, not runtime parameters:

```text
encoder cadence                 15 FPS
maximum age at encoder input    250 ms
maximum age at TCP send start   250 ms
per-access-unit send deadline   100 ms
maximum age at local handoff    350 ms
```

"Local handoff" means the complete four-byte prefix and access unit have been
accepted by the pc2 kernel socket. Network transit and headset decode latency
are outside this locally enforceable bound.

The implementation enforces this across every stage:

1. The ZMQ SUB socket uses `ZMQ_CONFLATE=1`, `ZMQ_RCVHWM=1`, and zero linger.
2. MessagePack receive/decode publishes into a one-element latest-value slot.
   A producer replaces an unconsumed value instead of waiting.
3. Before resize/duplication and again before `appsrc`, the source wall-clock
   timestamp is checked against the pc2 clock. Missing, non-finite,
   non-monotonic, future-skewed, or older-than-250-ms timestamps are dropped.
4. `appsrc` is live, nonblocking, limited to one queued buffer, and configured
   to leak the oldest buffer downstream when supported by the installed
   GStreamer version. Startup fails if equivalent bounded behavior cannot be
   configured.
5. The non-preview GStreamer path has no ordinary unbounded `queue`. `appsink`
   is nonblocking, drops old samples, and retains at most one sample.
6. GStreamer buffer PTS identifies the originating source frame. The appsink
   callback uses it to preserve the source timestamp through encoding.
7. The appsink callback never writes to TCP. It replaces a one-element latest
   encoded-access-unit slot.
8. Immediately before TCP send, the sender drops an access unit whose source
   age exceeds 250 ms. It uses `TCP_NODELAY`, a bounded socket send buffer, and
   nonblocking writes. Its absolute deadline is the earlier of 100 ms after
   send start and the instant the source frame reaches 350 ms old. A partial
   packet that cannot complete before that deadline closes the video
   connection; the bridge never continues after leaving a truncated
   length-framed packet in the stream.
9. Connect to the headset has a two-second deadline. A connection or send
   failure tears down the current video stream but leaves command port `13579`
   available for the next Unity request.

This policy prefers a visible freeze or explicit reconnect over displaying a
video stream that falls progressively behind real time.

## H.264 Wire Contract

GStreamer receives corrected BGR `2160x810` frames through `appsrc` at the
effective 15 FPS cadence, converts them to NV12 in NVMM memory, and encodes them
with `nvv4l2h264enc`. The encoder uses no B-frames, inserts SPS/PPS with IDR
frames, and emits a periodic IDR at most every 15 source frames.

`h264parse` uses `config-interval=-1`. After the parser, negotiated output caps
are explicit:

```text
video/x-h264,stream-format=byte-stream,alignment=au
```

Thus every appsink sample is one Annex-B H.264 access unit. Each TCP video
packet is:

```text
uint32_be access_unit_length
annex_b_access_unit[access_unit_length]
```

The first transmitted access unit after each video connection must contain
SPS, PPS, and an IDR frame. If the pipeline cannot negotiate byte-stream and
access-unit alignment, or the installed encoder cannot disable incompatible
codec behavior, `OPEN_CAMERA` fails visibly. A request with
`enableMvHevc != 0` is rejected; the bridge never silently substitutes H.264
for a requested codec.

The v1.1.1 Android decoder is implemented in a closed vendor AAR, so headset
integration remains the final compatibility authority. The Annex-B/AU contract
is also validated independently by stripping TCP length prefixes, concatenating
the access units, and decoding the resulting `.h264` stream with FFmpeg.

## Runtime Interface

```text
--listen ADDRESS             required command address, for example 0.0.0.0:13579
--allowed-headset-ip IPV4    required command peer and video destination
--gr00t-zmq ENDPOINT         default tcp://127.0.0.1:5555
--mount NAME                 default ego_view
--preview                    optional local preview with a bounded leaky branch
--help                       print usage
```

Encoder cadence, encoder bitrate, source-age limits, and send deadlines are
fixed production constants in this v1.1.1/RealSense bridge. They have no CLI
overrides. Changing one requires a new reviewed compatibility design rather
than an unchecked service argument.

Example:

```bash
./build/xr_video_bridge/gr00t_xr_video_bridge \
  --listen 0.0.0.0:13579 \
  --allowed-headset-ip 192.168.123.45 \
  --gr00t-zmq tcp://127.0.0.1:5555 \
  --mount ego_view
```

## Lifecycle and Failure Handling

1. The bridge starts its command listener without opening the RealSense or
   starting an encoder.
2. A validated `OPEN_CAMERA` connects to the allowed headset, creates a fresh
   encoder, and begins consuming current GR00T frames.
3. A second `OPEN_CAMERA` while streaming is rejected; the client must close or
   disconnect first.
4. `CLOSE_CAMERA`, command disconnect, video send failure, GStreamer error,
   SIGINT, or SIGTERM releases the subscriber, pipeline, and video socket. The
   RealSense and episode collector are unaffected.
5. A missing ZMQ publisher leaves the command listener alive. During an open
   session it reports the outage periodically without repeating stale frames,
   and resumes from a fresh frame when the publisher returns.

Malformed MessagePack, missing mounts, invalid base64, invalid JPEG, wrong
dimensions, invalid timestamps, and color conversion failures drop one frame
and increment a bounded-rate diagnostic counter. Periodic metrics include
received, decoded, stale-dropped, decode-dropped, overwritten, encoded,
network-dropped, and transmitted counts plus source-to-send age percentiles.
The implementation never logs every frame.

## Testing

### Command framing and protocol tests

- Construct canonical v1.1.1 `OPEN_CAMERA` and `CLOSE_CAMERA` fixtures.
- Feed each outer frame at every possible single split boundary.
- Feed complete frames one byte at a time.
- Feed `OPEN_CAMERA || CLOSE_CAMERA` in one receive chunk.
- Feed a partial frame followed by several concatenated frames.
- Reject zero, negative-equivalent, oversized, overflowing, truncated, invalid
  UTF-8, and trailing-byte cases before large allocation.
- Verify exact v1.1.1 profile acceptance and one-field-at-a-time rejection.
- Verify request IP, command peer, allowlisted IP, and port constraints.
- Capture one real command from the deployed v1.1.1 APK and compare it with the
  canonical fixture, including the outer four-byte length.

### JPEG color and SBS tests

- Generate saturated red, green, and blue RGB patches.
- Reproduce the RealSense legacy path by passing RGB directly to OpenCV JPEG
  encoding, serialize it as base64, and verify the bridge's decoded BGR output
  has the correct displayed channel dominance within JPEG tolerance.
- Encode a standard BGR test pattern as binary MJPEG and verify that the binary
  path does not perform the legacy corrective exchange.
- Verify each corrected `640x480` input becomes `2160x810`, each half is
  `1080x810`, the aspect ratio is preserved, and both halves match.
- Decode a hardware-encoded color-bar stream and verify correct colors after
  the complete JPEG-to-H.264 path.

### Freshness and slow-receiver tests

- Overproduce ZMQ frames and confirm only the latest unconsumed value survives.
- Inject missing, non-monotonic, future, and stale source timestamps and verify
  rejection.
- Stall a fake video receiver and verify all internal queue sizes remain one or
  zero.
- Force a partial TCP write past the deadline and verify the socket is closed
  rather than reused with corrupt framing.
- Verify complete local handoff never occurs after the fixed 350 ms source-age
  ceiling.
- Run a sustained slow-receiver test and verify the process has bounded memory
  and disconnects instead of accumulating latency.

### H.264 tests

- Assert negotiated caps are Annex-B byte-stream and access-unit aligned.
- Inspect the first access unit for SPS, PPS, and IDR NAL units.
- Verify every length prefix equals the complete following access-unit size.
- Strip prefixes, concatenate access units, and decode the stream with FFmpeg.
- Verify periodic IDR and SPS/PPS recovery after restarting a video connection.

### Jetson and headset integration

1. Confirm `nvv4l2h264enc` and all required bounded appsrc/appsink properties on
   pc2 before building the service.
2. Start GR00T `composed_camera` on `5555` and verify fresh `ego_view`
   timestamps and `480x640x3` decoded frames.
3. Start the bridge on `13579` with the actual PICO IPv4 allowlisted.
4. Select `PICO4U` in XRoboToolkit v1.1.1 and confirm the logged request is
   `VR`, `2160x810`, 60 FPS, 20,971,520 bps, render mode 2, and port `12345`.
5. Confirm the bridge logs its effective 15 FPS / 4,000,000 bps adaptation.
6. Confirm the headset renders a correctly proportioned, correctly colored
   image in both eyes.
7. Confirm source-to-send age remains bounded and the process does not build a
   queue during a ten-minute session.

### Collection regression test

The collection check must not pass merely because the exporter reuses its
cached image. While the bridge and episode recording run concurrently:

1. Instrument the actual `ComposedCameraClientSensor` used by the exporter and
   record each newly received `timestamps["ego_view"]` value and client receive
   time for at least ten seconds.
2. Assert source timestamps are strictly increasing on new-message events,
   source age is below 250 ms, no new-message gap exceeds 500 ms, and at least
   12 new messages per second are observed for the configured 15 FPS source.
3. Record a controlled moving RGB test target. Decode the saved episode and
   verify the expected color ordering and temporal progression; a single
   nonempty or endlessly repeated frame is a failure.
4. Restart only the XR bridge while collection continues. Assert the exporter's
   source timestamps continue advancing across the restart and the saved
   episode retains temporal progression.
5. Confirm logs contain no RealSense device-busy, disconnect, or second-owner
   errors.

## Acceptance Criteria

- Only `composed_camera` opens the RealSense.
- The v1.1.1 PICO4U request is parsed correctly under arbitrary TCP
  fragmentation/coalescing, and other profiles are rejected.
- XR output is `2160x810` SBS H.264 with correct color and decodes on the
  deployed v1.1.1 headset.
- The bridge has no unbounded frame queue and never completes local socket
  handoff for an access unit older than 350 ms.
- The data collector receives advancing, fresh camera timestamps while XR is
  active, and the recorded episode demonstrates temporal progression.
- Restarting or failing the XR bridge does not interrupt GR00T collection.
- Installation and service generation require no unpinned external checkout.

## Non-Goals

- Capturing the RealSense directly from the XR bridge.
- Streaming depth to Unity.
- Producing true stereo depth views from the monocular color stream.
- Modifying or rebuilding the v1.1.1 Unity APK.
- Supporting other XRoboToolkit APK profiles or releases.
- Authenticating or encrypting the trusted-LAN XR protocol.
