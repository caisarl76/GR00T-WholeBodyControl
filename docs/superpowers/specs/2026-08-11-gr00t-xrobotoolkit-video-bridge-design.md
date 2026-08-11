# GR00T to XRoboToolkit Video Bridge Design

## Purpose

Enable XRoboToolkit Unity Remote Vision to display the Unitree G1 RealSense
camera while GR00T data collection records the same camera stream. The bridge
must not open the RealSense device or change the existing episode image path.

Success requires both of the following at the same time:

- the GR00T episode collector receives valid `observation.images.ego_view`
  frames; and
- XRoboToolkit displays the camera through its `PICO4U` video source.

## Existing Contracts

The existing GR00T `composed_camera` process is the only RealSense owner. It
publishes MessagePack payloads through ZMQ on `tcp://*:5555`. Each payload has
an `images` map containing:

- `ego_view`: a `640x480` JPEG represented as either a base64 string or binary
  bytes; and
- `ego_view_depth`: depth data that the XR bridge ignores.

XRoboToolkit Unity connects to a command server on TCP port `13579`. An
`OPEN_CAMERA` request for the `PICO4U` profile supplies camera type `VR`, output
size `1280x480`, frame rate `15 FPS`, bitrate `1,000,000 bps`, and the headset
IP and streaming port. Encoded video is returned to the headset on TCP port
`12345`. Each H.264 buffer is preceded by a four-byte big-endian payload
length.

## Architecture

```text
RealSense D435
     | pyrealsense2 (single camera owner)
     v
GR00T composed_camera :5555
     | ZMQ PUB
     +--------------------> episode data collector
     |
     +--------------------> OrinVideoSenderGR00T
                               | TCP command server :13579
                               | MessagePack/JPEG decode
                               | mono-to-SBS conversion
                               | Jetson H.264 encoding
                               v
                         PICO headset :12345
```

The bridge is an independent ZMQ subscriber. Its receive queue retains only
the newest frame so XR backpressure cannot produce an increasingly stale
display. This does not change the independent subscriber queue used by the
episode collector.

## Components

### GR00T XR sender

Add `main_gr00t_zmq_tcp.cpp` to the XRoboToolkit Orin sender checkout. It reuses
the upstream XR command deserialization, TCP server, TCP client, and encoded
buffer framing. It does not include ZED headers, link `libsl_zed`, or access a
`/dev/video*` device.

The executable is named `OrinVideoSenderGR00T` and accepts:

```text
--listen ADDRESS       XR command address; production value 0.0.0.0:13579
--gr00t-zmq ENDPOINT   GR00T publisher; default tcp://127.0.0.1:5555
--mount NAME           color-image mount; default ego_view
--preview              optional local preview
--help                 print usage
```

### Independent build

Add `Makefile.gr00t` instead of changing the upstream ZED-selected Makefile.
The GR00T build links GStreamer, GStreamer app, GLib, OpenCV, ZeroMQ,
MessagePack, OpenSSL, CUDA, and pthread dependencies. It excludes all ZED SDK
include paths and libraries.

Build with:

```bash
make -f Makefile.gr00t
```

### Service

After manual validation, add `gr00t_xr_video_bridge.service`. It starts after
network availability and the composed-camera service, launches the bridge with
the production arguments, and restarts on failure. It is a separate service so
stopping or restarting XR streaming cannot stop camera capture or episode
collection.

## Runtime Data Flow

1. The bridge listens on `0.0.0.0:13579` without starting an encoder.
2. Unity connects and sends `OPEN_CAMERA` using the `PICO4U` profile.
3. The bridge validates the `VR` request and records its dimensions, FPS,
   bitrate, target IP, and target port.
4. The bridge connects a TCP video client to the headset target, normally port
   `12345`.
5. A ZMQ subscriber reads the newest MessagePack payload from port `5555`.
6. The bridge selects `images["ego_view"]`, base64-decodes it when necessary,
   and uses OpenCV to decode the JPEG into BGR pixels.
7. The `640x480` image is duplicated horizontally to produce a `1280x480`
   side-by-side image. Both eyes therefore see the same monocular camera view
   without cropping or aspect distortion.
8. GStreamer receives BGR frames through `appsrc`, converts them to NV12 NVMM
   memory, and encodes them with `nvv4l2h264enc`. Encoder FPS and bitrate come
   from the validated Unity request. SPS/PPS insertion and periodic IDR frames
   allow the Unity decoder to recover after reconnects.
9. Each encoded appsink buffer is prefixed with its four-byte big-endian size
   and sent over TCP to the headset.
10. `CLOSE_CAMERA` or a Unity disconnect stops the subscriber, encoder, and
    video TCP client but leaves the command listener alive.

## Validation and Bounds

The bridge accepts the `PICO4U` contract: camera type `VR`, width `1280`, height
`480`, and positive FPS and bitrate. It rejects invalid dimensions, ports,
bitrates, or unsupported camera types without affecting the command server.
Encoded output uses H.264 even if a malformed request asks for an unsupported
codec.

The implementation bounds command and frame lengths before allocating memory.
It handles one active Unity command client and one active output stream. A
second open request replaces no existing stream; Unity must close or disconnect
the active stream first.

## Failure Handling

- A missing mount, malformed MessagePack payload, invalid base64 value, or JPEG
  decode failure drops only that frame and increments a diagnostic counter.
- A GR00T publisher outage leaves the command server alive. The ZMQ subscriber
  continues waiting and resumes when frames return.
- A headset connection or send failure stops the current XR stream and returns
  to the command-listening state.
- A GStreamer construction, state, or encode failure tears down the current
  pipeline without affecting `composed_camera`.
- `CLOSE_CAMERA`, command-client disconnect, SIGINT, and SIGTERM release all
  GStreamer, ZMQ, and TCP resources deterministically.
- The bridge prints periodic counters for received, decoded, dropped, encoded,
  and transmitted frames without logging every frame.

## Testing

### Automated tests

- Parse valid `OPEN_CAMERA` and `CLOSE_CAMERA` packets and reject malformed
  lengths, magic bytes, versions, dimensions, and ports.
- Verify H.264 payload length framing is four-byte big-endian.
- Decode MessagePack payloads containing base64 JPEG and binary JPEG values.
- Ignore `ego_view_depth` and reject missing or malformed `ego_view` values.
- Convert a known `640x480` BGR image into `1280x480` and verify that both
  output halves exactly match the input.
- Verify bounded latest-frame behavior by submitting frames faster than the
  consumer and confirming stale frames are discarded.

Protocol and frame conversion logic must be isolated from sockets and
GStreamer so these tests run without a RealSense, headset, or Jetson encoder.

### Jetson integration test

1. Start GR00T `composed_camera` on port `5555` and confirm `ego_view` is
   `480x640x3` after JPEG decoding.
2. Start `OrinVideoSenderGR00T` on port `13579`.
3. Select `PICO4U` in XRoboToolkit and open Remote Vision.
4. Confirm the bridge logs `VR`, `1280x480`, `15 FPS`, the headset IP, and port
   `12345`.
5. Confirm the headset renders a correctly proportioned image in both eyes.
6. Confirm the sender approaches 15 FPS without unbounded queue growth.

### Collection regression test

1. Start XR display and record a short GR00T episode concurrently.
2. Confirm the episode contains decodable, nonempty
   `observation.images.ego_view` frames.
3. Confirm logs contain no RealSense device-busy, disconnect, or second-owner
   errors.
4. Restart the XR bridge while collection continues and verify new episode
   frames remain available.

## Non-Goals

- Capturing the RealSense directly from the XR sender.
- Streaming `ego_view_depth` to Unity.
- Producing true stereo depth views from the monocular color stream.
- Modifying the XRoboToolkit Unity client or its `PICO4U` profile.
- Replacing the GR00T camera or episode data-collection protocols.
