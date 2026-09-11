# PICO Optical Hand Tracking for Full-Body Teleoperation

## Status

Implementation authorized on 2026-09-07 and carried out on
`fix/xr-hand-tracking-watchdog`. Software verification and remaining
qualification work are tracked in
`../plans/2026-09-07-pico-optical-hand-tracking.md`; operating instructions are
in `../../source/pico_optical_hand_tracking.md`. D0 recorded headset replay,
D1 simulation and attached-hand stages remain unverified.

Original-APK compatibility was authorized on 2026-09-07 after the original
timestamp prerequisite blocked finger control. This amendment supersedes the
timestamp-only admission requirement below: complete original hand payloads
use a per-side host content-change clock; identical cached packets never
refresh it. Stationary data may conservatively hold after 100 ms and movement
must satisfy five-sample readmission. Snapshot/capture `timestamp_source`
distinguishes this from device timestamps; it does not establish sensor
freshness. A side that has supplied device timestamps cannot silently fall
back to host timing. Limits, rate bounds and feedback admission are unchanged.
The optional `external_dependencies/pico_optical_hand_source.patch` provides
stronger evidence by timestamping successful side-specific SDK reads, but the
native API still does not expose sensor acquisition timestamps.
Raw hand positions are native OpenXR coordinates; they must not
receive the Unity rendering Z reflection. See the operating guide for pinned
source evidence and the corresponding body-coordinate check.

## Goal

Use the PICO 4 Ultra full-body setup for simultaneous:

- full-body G1 teleoperation through
  `gear_sonic/scripts/pico_manager_thread_server.py`;
- wrist and arm 6-DoF targets from the fused PICO body skeleton produced from
  the waist, ankle, and wrist trackers;
- optical finger articulation from the PICO hand skeleton;
- Dex3 control with human thumb, index, and middle fingers mapped separately;
- Inspire FTP control with all five human fingers mapped separately; and
- restart-safe episode recording controlled before and after the operator puts
  down the PICO controllers.

If optical tracking for one hand is lost, only that robot hand holds its last
safe command. The other hand and full-body motion continue.

## Existing System and Reusable Work

The current manager already provides most of the body path:

- `PicoReader` reads the fused 24-joint PICO body skeleton.
- `ThreePointPose` extracts the left and right arm end-effectors and neck.
- pose and planner messages already carry Dex3
  `left_hand_joints[7]` and `right_hand_joints[7]` fields.
- PICO face buttons and grip buttons already enter the manager loop.
- `XRClient.get_hand_tracking_state()` exposes 26 position/quaternion joints,
  and separate getters expose `isActive`.

The current hand values are synthetic trigger/grip gestures. They do not use
optical landmarks. The existing `G1GripperInverseKinematicsSolver` also
assumes synthetic unit-scale landmarks, so raw PICO positions in metres must
not be passed to it.

The branch contains a certified XR upper-body safety bridge, but that path is
stationary-lower-body and treats the arms and both hands as one safety unit.
It is not the runtime for this feature, which requires live full-body motion
and per-hand isolation.

The previously developed Inspire FTP PC2 bridge supplies reusable DDS
ownership, feedback validation, watchdog, bounded-envelope, and fail-open
behavior. Its old q7 gesture projection is not the optical command contract;
the optical path sends the six independently retargeted Inspire motors.

## Selected Architecture

Hand capture, validation, watchdogs, and retargeting run in the PICO manager
process:

```text
PICO body skeleton / five trackers
        |
        +----> body pose + arm 6-DoF ----> existing full-body pipeline
        |
PICO optical left/right hand skeletons
        |
        v
atomic per-side snapshot + schema/geometry validation
        |
        v
independent per-side 100 ms watchdog
        |
        +---- valid ------> robot retargeter ----> one manager filter/limiter
        |
        +---- invalid ----> last accepted command (hold that hand only)

PICO controller buttons --------+
keyboard T/C/S -----------------+----> manager + acknowledged recorder FSM
```

Direct manager integration keeps body and hand commands in one 50 Hz loop.
It does not make optical hand health part of the body watchdog.

The implementation is split into dependency-light pieces:

1. Extend the XRoboToolkit binding with one atomic per-side hand snapshot.
2. A hand-sample module owns joint ordering, validation, per-side source
   epochs, watchdog state, and immutable status.
3. A retargeting module owns the audited Unitree-config translation, named
   output permutations, limits, and the single manager-side filter/limiter.
4. A nonblocking keyboard module owns terminal setup and T/C/S events.
5. A recording-control module owns the restart-safe command queue and ACK
   state; the exporter remains the sole recording-state owner.
6. `pico_manager_thread_server.py` composes those parts for `PoseStreamer` and
   `PlannerStreamer` without duplicating safety state.

## Atomic XR Hand Snapshot

### Binding change

The current C++ binding loses each hand's `timeStampNs` and tracking flags and
returns pose and `isActive` through separate getter calls. That cannot prove
independent freshness and can tear one logical sample.

For each side, the binding will parse the side object into temporaries and,
only after the complete side object passes structural parsing, atomically swap
this snapshot under the existing side mutex:

```text
HandSnapshotV1
  pose:                  float64[26,7]
  location_flags:        uint64[26]
  radius:                float64[26]
  scale:                 float64
  is_active:             int32
  source_timestamp_ns:   int64
  binding_generation:    uint64
```

`location_flags` comes from each `HandJointLocations[i].s`, `radius` from
`.r`, and `source_timestamp_ns` from that side's `timeStampNs`. The binding
increments `binding_generation` only when it commits a completely parsed side
object. An absent or malformed side does not mutate its previous snapshot.

The Python API exposes `get_left_hand_snapshot()` and
`get_right_hand_snapshot()`. Each call copies every field above under one
mutex acquisition. Optical mode does not call the legacy separate hand getters.
Older XR service data that omits a per-hand timestamp or location flags is
incompatible with optical mode and fails startup rather than using the shared
body/global timestamp.

### Independent freshness

The clocks are not assumed to share an epoch. For each side, Python stores the
last strictly increasing `source_timestamp_ns` and the local
`time.monotonic_ns()` at which that advancement was first observed.

- A greater source timestamp is a new side sample and refreshes local age.
- An equal source timestamp is cached data and never refreshes local age, even
  if body/global time or `binding_generation` advances.
- A lower timestamp starts a new side source epoch, immediately puts only that
  side into `HOLDING`, clears its valid streak, and requires five strictly
  increasing samples in the new epoch before recovery.
- A side is stale when its local age is greater than or equal to 100,000,000 ns.

`binding_generation` proves that the fields came from one committed binding
snapshot; it is not a substitute for source-time advancement.

## Joint Layout and Coordinate Use

XRoboToolkit follows the OpenXR joint enumeration and copies the array without
reordering:

```text
raw[0] = PALM
raw[1] = WRIST
raw[2:6] = THUMB metacarpal, proximal, distal, tip
raw[6:11] = INDEX metacarpal, proximal, intermediate, distal, tip
raw[11:16] = MIDDLE ... tip
raw[16:21] = RING ... tip
raw[21:26] = LITTLE ... tip
```

The bundled `test_hand_isactive.py` comment that labels raw joint 0 as wrist
is wrong and will be corrected.

The Unitree retargeting convention expects 25 points with wrist at index 0 and
tips at 4, 9, 14, 19, and 24. The only adapter is therefore:

```text
canonical[25,3] = raw[1:26, 0:3]
```

This drops only PALM. It must not concatenate raw joint 0 with raw joints
2-25.

Retargeting uses wrist-relative positions from this canonical array. Optical
quaternions do not drive finger articulation. They must be finite because they
are part of the atomic source record, but quaternion norm or orientation-
tracked flags do not reject an otherwise tracked position sample.

Arm pose continues to use body joints 22/23, the existing Hand/end-effector
points used by `_process_3pt_pose()`. The optional 0.30 m optical alignment
check instead compares optical raw joint 1 against body joints 20/21, the
anatomical left/right Wrist points, in the original common XRT tracking frame
before root-relative or robot-coordinate transforms.

## Per-Hand Validation

Left and right are validated independently. A side is valid only when all of
the following hold:

- `is_active == 1`;
- pose has exact shape `[26,7]`, `float64` at the binding boundary, and finite
  values;
- location flags have exact shape `[26]` and dtype `uint64`;
- for every retained raw joint 1-25, both OpenXR `POSITION_VALID` (`0x2`) and
  `POSITION_TRACKED` (`0x8`) are set;
- `source_timestamp_ns` is positive and belongs to the current side epoch;
- every retained landmark is within 0.5 m of raw joint 1;
- each consecutive anatomical digit segment is in `[0.002,0.15]` m and total
  retained-landmark spread is at most 0.5 m;
- raw optical wrist to same-side body joint 20/21 is available and its distance
  is at most 0.30 m; and
- the independent side age is less than 100 ms.

An unavailable body wrist produces `BODY_WRIST_UNAVAILABLE` and invalidates
that optical side for the tick. It does not invalidate the opposite side or
the body stream. Five fully comparable frames are required before first
hardware activation and every recovery.

Validation produces a stable enum reason code and never clips malformed source
geometry into a valid sample. Hand invalidity never feeds the existing body/XR
watchdog and cannot transition the manager to `OFF`.

## Per-Hand Watchdog, Recovery, and Filtering

Each side owns this state machine:

```text
WAITING or HOLDING -- 5 valid frames --> RECOVERING
RECOVERING -- converged 5 valid ticks --> TRACKING
RECOVERING or TRACKING -- invalid/stale/source restart --> HOLDING
```

The behavior is exact:

- `WAITING` uses the latest measured robot-hand position. Without fresh
  measured feedback, optical arming is refused.
- One invalid, inactive, malformed, regressing, or stale sample immediately
  changes `RECOVERING` or `TRACKING` to `HOLDING`. The last emitted bounded
  command remains unchanged for that side.
- Five consecutive valid, source-advancing 50 Hz samples move `WAITING` or
  `HOLDING` to `RECOVERING`. A duplicate source timestamp does not count.
- On entry to `RECOVERING`, the post-retargeting filter state is reset to the
  held command and its time origin is reset to the current monotonic tick.
- `RECOVERING` follows the valid target through the same filter and limiter as
  `TRACKING`; it is not a jump. It reaches `TRACKING` after five consecutive
  valid ticks whose maximum output-to-target error is at most 0.02 rad for
  Dex3 or 0.01 normalized units for Inspire.
- Any invalid tick resets both the validity and convergence streaks and returns
  to `HOLDING`.

Official `dex-retargeting` 0.4.6 filtering is disabled by setting
`low_pass_alpha=-1.0` in the normalized config; its builder creates an
`LPFilter` only for alpha in `[0,1]`. The PICO manager is the one
and only smoothing owner. Its per-side post-retargeting filter is:

```text
dt = clamp(now - previous_tick, 0, 0.040 seconds)
alpha = 1 - exp(-dt / 0.060 seconds)
filtered = previous + alpha * (target - previous)
output = rate_limit(filtered, previous, max_rate * dt)
```

No delayed-frame credit above 40 ms is accumulated. The output is then checked
for exact shape, finiteness, and backend limits. Production Dex3 manager rate
is 2.0 rad/s per joint. The existing downstream C++ 2.5 rad/s clamp remains a
hard independent safety ceiling, not a second smoothing filter, and must not
activate during nominal manager output. Inspire manager rate is 1.0 normalized
unit/s before the stricter PC2 envelope/slew gate.

No optical failure automatically switches to controller trigger/grip finger
control. Examples:

- left hand lost: left holds; right hand and full body continue;
- both hands lost while controllers are picked up: both hold; body continues;
- controller data lost on the table: latched tracking and recording states do
  not change.

Optical-source loss is different from manager, transport, DDS, or physical
driver loss. Backends retain their stricter stop/open behavior for those
system faults.

## Retargeting Dependency and Audited Config Loader

The teleoperation environment pins official `dex-retargeting==0.4.6`. This is
the pre-0.5 line compatible with this repository's NumPy 1.26 and installed
Torch 2.x. The Unitree fork is not installed implicitly.

Unitree assets are vendored from `unitreerobotics/xr_teleoperate` commit:

```text
934361718c5886356e19f9068f4536f04d2b9feb
```

The exact files and SHA-256 digests are:

| File | SHA-256 |
| --- | --- |
| `assets/unitree_hand/unitree_dex3.yml` | `2c39bc8c123bdc00f299d02d995d9bc4a12179adf4942eabc27349252c1c7f15` |
| `assets/unitree_hand/unitree_dex3_left.urdf` | `569e4d8ca55b1b76e51cd41df3cd6f655c5475a3e02e908f673f999759c51bd0` |
| `assets/unitree_hand/unitree_dex3_right.urdf` | `4e36e939466d386524d38079f8e2ff84d9dd4d3a7f8db6319374067970ad7075` |
| `assets/inspire_hand/inspire_hand.yml` | `94feea4fc387a52bdda57538715f508b51a7704225e102e59db24941de5c05bf` |
| `assets/inspire_hand/inspire_hand_left.urdf` | `bfd26847a64b3794aabdeaeda6c790866b5d702c919d765fecf56fb584212367` |
| `assets/inspire_hand/inspire_hand_right.urdf` | `3dc82ee57e8918ce6316b4d0a0572450e719474b437f91dca4fea1253547b8c1` |

The vendored directory includes a provenance manifest and upstream license.
Startup verifies the manifest; a mismatch refuses optical arming.

Official 0.4.6 accepts `target_link_human_indices`, while Unitree YAML uses
type-suffixed keys. A small audited loader performs this deterministic
translation for each selected `left`/`right` section:

1. Parse with `yaml.safe_load`; require an exact mapping and a declared side.
2. Normalize `type` to lower case and accept only `dexpilot` or `vector`.
3. Select exactly `target_link_human_indices_<normalized type>`.
4. Copy that value to `target_link_human_indices` in a new dictionary.
5. Remove all suffixed human-index keys and set `low_pass_alpha=-1.0`.
6. Reject unknown keys against the 0.4.6 `RetargetingConfig` field allow-list.
7. Resolve the URDF path under the vendored asset root; reject path escape.
8. Import `RetargetingConfig` from
   `dex_retargeting.retargeting_config` (0.4.6 does not re-export it at the
   package root) and call `RetargetingConfig.from_dict()` with the translated
   copy.

The source YAML is never mutated. Missing selected keys, both selected and
canonical keys, an unsupported type, an unknown option, a checksum mismatch,
or an out-of-root URDF path is fatal.

CI includes a non-mocked smoke test that constructs left and right Dex3 and
Inspire retargeters from the real vendored YAML/URDF files and performs one
finite retarget call for each. Unit tests alone with fake configs are not an
acceptance substitute.

## Dex3 Output Contract

Dex3 is implemented and qualified first. Only the three human digits that
exist on the robot participate:

| Human digit | Dex3 digit | Robot joints |
| --- | --- | --- |
| Thumb | Thumb | `thumb0`, `thumb1`, `thumb2` |
| Index | Index | `index0`, `index1` |
| Middle | Middle | `middle0`, `middle1` |

Ring and little landmarks are diagnostic only for Dex3.

The retargeter output is reordered by exact joint name into the established
hardware/ZMQ API order:

```text
thumb0, thumb1, thumb2, middle0, middle1, index0, index1
```

Left and right use side-specific URDF signs and limits. The manager emits
contiguous little-endian `float32[7]` only after exact-name coverage, shape,
finiteness, limit, filter, and rate checks. A missing, duplicate, or unexpected
joint name prevents profile startup. The existing C++ Dex3 writer remains the
sole motor writer.

## Inspire FTP q6 Command Protocol

### Semantic output

Inspire consumes the same canonical optical input directly; it never projects
from Dex3 q7. Named retargeter joints are permuted into this authoritative
normalized order:

| Index | Human input | FTP motor | Convention |
| ---: | --- | --- | --- |
| 0 | Little curl | Pinky | `1=open`, `0=closed` |
| 1 | Ring curl | Ring | `1=open`, `0=closed` |
| 2 | Middle curl | Middle | `1=open`, `0=closed` |
| 3 | Index curl | Index | `1=open`, `0=closed` |
| 4 | Thumb flexion | Thumb bend | `1=open`, `0=closed` |
| 5 | Thumb opposition/abduction | Thumb rotation | `1=open`, `0=closed` |

Each side is exact contiguous little-endian `float32[6]`, finite and in
`[0,1]`. FTP counts are created only on PC2 with
`round(normalized * 1000)`, after the PC2 envelope and slew limiter.

The retargeter's named Inspire joint output is normalized deterministically.
For `normalize(q, lower, upper)`, first require
`q in [lower - 1e-4, upper + 1e-4]`, clamp only that numerical tolerance to the
endpoint, then calculate `1 - (q - lower) / (upper - lower)`. The exact mapping
for both sides is:

| q6 index | Retargeter joint suffix | Lower/upper radians |
| ---: | --- | --- |
| 0 | `pinky_proximal_joint` | `[0.0, 1.7]` |
| 1 | `ring_proximal_joint` | `[0.0, 1.7]` |
| 2 | `middle_proximal_joint` | `[0.0, 1.7]` |
| 3 | `index_proximal_joint` | `[0.0, 1.7]` |
| 4 | `thumb_proximal_pitch_joint` | `[0.0, 0.5]` |
| 5 | `thumb_proximal_yaw_joint` | `[-0.1, 1.3]` |

Left names have `L_` and right names have `R_`. Exact named coverage is
required; list position from the YAML is never treated as hardware order.
Values farther outside the vendored URDF limits invalidate that side rather
than being silently saturated.

### Manager-to-PC2 wire contract

All manager ZMQ messages use the existing 1,280-byte self-described header.
The generic serializer and every strict decoder add native `uint8 <-> u8`
support and reject, rather than coerce, unsupported dtypes.

At manager start, a random nonzero 128-bit `manager_session_id` is generated. Every
accepted stream-mode transition increments a nonnegative signed-int64
`mode_epoch`; overflow at `2**63 - 1` is terminal. The compact field `pv` is a
`uint8[28]` byte array:

```text
bytes 0..15   manager_session_id
bytes 16..23  mode_epoch, little-endian int64
bytes 24..27  effective stream_mode, little-endian int32
```

`inspire_hand` protocol/header version 1 is published on the manager's existing
port 5556 at 50 Hz while SONIC is active. Required fields, in order, are:

| Field | Dtype/shape | Meaning |
| --- | --- | --- |
| `pv` | `uint8[28]` | session, mode epoch, effective mode |
| `message_seq` | `int64[1]` | starts at 0 for each new `pv`; strictly increments per packet |
| `sample_generation` | `int64[1]` | manager body/hand loop generation |
| `left_command` | `float32[6]` | live or held normalized left command |
| `right_command` | `float32[6]` | live or held normalized right command |
| `left_tracking_state` | `int32[1]` | `0 WAITING, 1 HOLDING, 2 RECOVERING, 3 TRACKING` |
| `right_tracking_state` | `int32[1]` | same enum |
| `left_source_epoch` | `int64[1]` | increments on left source-time regression |
| `right_source_epoch` | `int64[1]` | increments on right regression |
| `left_source_timestamp_ns` | `int64[1]` | last accepted left source timestamp |
| `right_source_timestamp_ns` | `int64[1]` | last accepted right source timestamp |

The pair is sent from the same loop generation as the body packet. In POSE or
PLANNER_VR_3PT, valid sides can advance. In other modes, the latest bounded
commands remain in diagnostic packets with non-`TRACKING` states, but those
packets are not eligible for PC2 publication. A transition out of an
authorized mode invokes the PC2 bridge's bounded opening behavior. The q6
packet never enters the C++ Dex3 decoder.

The PC2 bridge uses one SUB socket on port 5556 so `manager_state` and
`inspire_hand` ordering is preserved. A q6 packet is eligible only when:

- protocol version, required field order, dtype, shape, byte length, and range
  are exact;
- its `pv` equals the latest earlier valid `manager_state.pv` on that socket;
- its mode is POSE (`1`) or PLANNER_VR_3PT (`5`); and
- `message_seq` is greater than the last accepted sequence for that `pv`.

Duplicates are ignored. Because every q6 packet is an absolute complete pair,
forward gaps are accepted and counted. A sequence regression within one `pv`,
mode epoch rollback, or a retired session replay latches a provenance fault in
publish mode. No hand packet itself adopts a session.

A new valid `manager_state` session retires the previous session, clears the
cached pair, and changes an active bridge to bounded `OPENING`. The new session
is usable only after ten healthy open-feedback ticks and then a later matching
q6 packet. Retired session IDs are never evicted; more than 64 distinct manager
sessions is a terminal bridge fault requiring restart.

Manager-state and q6 transport staleness remain the PC2 bridge's independent
system watchdogs: each becomes stale at `>=250 ms` from local receipt. Optical
side staleness is already handled in the manager at 100 ms and continues to
produce a fresh held q6 packet, so it does not cause the other hand or body to
stop. A manager/transport timeout is a different fault and slews both Inspire
hands open under the PC2 safety FSM.

### Feedback ownership and PC2-to-host status

`Headless_driver_double.py` remains the sole Modbus owner. The PC2 bridge is
the sole DDS command writer on `rt/inspire_hand/ctrl/{l,r}` and the sole
component in this feature that validates DDS feedback from
`rt/inspire_hand/state/{l,r}`. It retains the reviewed DDS writer-discovery
proof and competing-writer fault latch.

The PC2 bridge publishes `inspire_hand_status` protocol/header version 1 at
10 Hz on PC2 port 5563. It uses a random `pc2_session_id` per process and has:

| Field | Dtype/shape | Meaning |
| --- | --- | --- |
| `pc2_session_id` | `uint8[16]` | PC2 bridge process identity |
| `status_seq` | `int64[1]` | strictly increasing status sequence |
| `accepted_pv` | `uint8[28]` | current manager provenance or all zero when none |
| `bridge_state` | `int32[1]` | `0 MONITORING, 1 READY, 2 ACTIVE, 3 OPENING, 4 FAULT_LATCHED, 5 SHUTDOWN_OPENING` |
| `last_applied_message_seq` | `int64[1]` | q6 sequence used for the applied target, `-1` if none |
| `left_angle_act` | `int32[6]` | validated measured FTP counts |
| `right_angle_act` | `int32[6]` | validated measured FTP counts |
| `left_err` | `uint8[6]` | raw validated FTP error bytes |
| `right_err` | `uint8[6]` | raw validated FTP error bytes |
| `left_applied` | `float32[6]` | last successfully DDS-written normalized command |
| `right_applied` | `float32[6]` | same |
| `feedback_healthy` | `bool[1]` | both sides fresh, shaped, ranged, and zero-error |
| `fault_code` | `int32[1]` | stable bridge fault enum, zero when healthy |

The manager and exporter each subscribe directly. They use local receipt age,
not the PC2 monotonic clock; status is stale at `>=500 ms`. A changed
`pc2_session_id` clears measured hand readiness and requires ten new healthy
statuses. No second process republishes or fabricates physical feedback.

## Runtime Profiles and CLI

The manager adds explicit startup choices:

```text
--hand-input optical|controller|off
--hand-profile dex3|inspire_ftp
```

The requested production configuration is `--hand-input optical`. Existing
trigger/grip articulation remains only in `controller` mode. A running process
never switches articulation sources automatically.

The exporter receives the same required `--hand-profile`; a manager/exporter
profile mismatch prevents recording. The default remains backward compatible
until the optical qualification ladder passes, while production launchers pass
both options explicitly.

## Controller and Keyboard Arbitration

### Exact chord resolver

Button samples arrive in the 50 Hz manager loop. Each digital input must be
stable for two consecutive samples (40 ms) before its debounced value changes.
After the first debounced press among A/B/X/Y or left grip, the resolver opens
a 200 ms chord window. It emits exactly one highest-priority matching action at
the deadline, or immediately when A+B+X+Y is complete. It then enters
`WAIT_RELEASE` until A/B/X/Y and left grip are all released for 40 ms.

Priority is:

```text
A+B+X+Y  >  left grip+B  >  left grip+A  >  A+X
```

This gives a staggered full chord up to 200 ms to suppress its A+X subset. If
A+X remains the best chord at the deadline, tracking toggles; adding B+Y after
that deadline does nothing until full release and a new attempt. Bounce inside
the window changes only the candidate after the 40 ms debounce; it cannot emit
twice.

Controller packets absent for 100 ms cancel a pending chord and disarm the
resolver without changing mode or recording. After reconnection, all inputs
must be neutral for 40 ms before rearming. A controller loss never synthesizes
a release action or changes latched tracking/recording.

### Transition table

| Manager/recorder state | A+B+X+Y | A+X or T | Grip+A | C | S | Grip+B |
| --- | --- | --- | --- | --- | --- | --- |
| `OFF` | start SONIC, enter planner | ignore | ignore | ignore | ignore | ignore |
| planner + recorder `IDLE` | stop SONIC and exit | enter full-body tracking | ignore | ignore | ignore | ignore |
| tracking + recorder `IDLE` | stop SONIC and exit | return to planner | enqueue TOGGLE | enqueue START | no-op | no-op |
| tracking + `RECORDING` | enqueue ABORT best-effort, then stop SONIC | refuse until recorder stops | enqueue TOGGLE | no-op | enqueue STOP_AND_SAVE | enqueue ABORT |
| tracking + `SAVING` | stop SONIC and exit; exporter finishes independently | refuse | no-op | no-op | no-op | no-op |
| tracking + recorder `ERROR`, capture inactive | stop SONIC and exit | return to planner | refuse | refuse | no-op | no-op |
| any state + recorder status unknown/stale | global stop remains allowed | refuse tracking exit if it could hide an active recording | refuse new recording command | refuse | refuse | refuse |

Controller A+B+X+Y remains the only manager-side SONIC start/stop input.

The manager keyboard is case-insensitive and edge-triggered:

| Key | Action |
| --- | --- |
| T | Toggle full-body tracking under the table above |
| C | Explicit START recording |
| S | Explicit STOP_AND_SAVE recording |

There is no manager keyboard SONIC key. In particular, manager O/o is ignored.
O/o in `gear_sonic_deploy/deploy.sh` remains the independent deploy-side SONIC
emergency stop. Existing deploy gamepad controls remain unchanged.

Ctrl+C restores terminal settings and follows the shutdown behavior specified
in the recording section; it does not publish the normal SONIC-stop command.

## Restart-Safe Recording Protocol

### Wire types

The generic packed-message serializer gains native `uint8 -> u8`; both the
shared decoder and the exporter decoder gain exact `u8 -> numpy.uint8` and
reject unknown dtype strings. Tests lock byte offsets and prevent silent
float32 conversion.

`manager_state` is upgraded to protocol/header version 4 while retaining
legacy fields for old consumers. It carries these recording fields:

| Field | Dtype/shape | Meaning |
| --- | --- | --- |
| `pv` | `uint8[28]` | manager session/mode provenance |
| `recording_protocol_version` | `int32[1]` | constant `1` |
| `recording_command` | `int32[1]` | enum below |
| `recording_command_seq` | `int64[1]` | per-session sequence, starts at 0 |

Commands are `0 NONE, 1 START, 2 STOP_AND_SAVE, 3 TOGGLE, 4 ABORT`.
Deduplication identity is `(manager_session_id, recording_command_seq)`, where
the session is the first 16 bytes of `pv`. Sequence 0 is reserved for NONE and
session adoption. Accepted commands begin at 1 and increment without wrap;
`2**63 - 1` is terminal.

For migration, the legacy `toggle_data_collection` pulse is true only on the
first packet of an accepted TOGGLE. An updated exporter ignores that legacy
pulse whenever valid protocol-v1 recording fields are present, so the same
operator action cannot be applied twice. Old consumers still see one pulse.

The exporter is the sole recorder state owner. In managed mode it disables its
independent `ZMQKeyboardSubscriber`; all controller and manager-keyboard
actions enter this protocol. Legacy exporter keyboard behavior remains only
behind an explicit `--legacy-recording-controls` mode that cannot be combined
with manager recording fields.

### ACK/status channel

The exporter binds a PUB socket on configurable local port 5562. The manager
connects a SUB socket to topic `recording_status`. Protocol/header version 1
is published immediately after any state/command decision and at 20 Hz:

| Field | Dtype/shape | Meaning |
| --- | --- | --- |
| `recorder_session_id` | `uint8[16]` | random exporter process identity |
| `status_seq` | `int64[1]` | strictly increasing status sequence |
| `adopted_manager_session_id` | `uint8[16]` | adopted manager or all zero |
| `last_applied_command_seq` | `int64[1]` | `-1` before adoption |
| `recording_state` | `int32[1]` | `0 IDLE, 1 RECORDING, 2 SAVING, 3 ERROR` |
| `capture_active` | `bool[1]` | exporter is still appending frames |
| `episode_index` | `int64[1]` | active/next exporter episode index |
| `error_code` | `int32[1]` | stable error enum, zero when none |

The manager treats ACK as stale at `>=500 ms`. A changed
`recorder_session_id` clears all manager recorder assumptions until the new
exporter adopts the current session and reports IDLE.

### Delivery and ordering

The manager owns a FIFO of at most 16 unacknowledged commands. It emits the
oldest command in every 50 Hz `manager_state` until
`last_applied_command_seq >= that sequence`, then advances to the next. Thus
resend cadence is exactly 20 ms while the manager loop is healthy. Commands are
accepted from the operator only while the FIFO has space; overflow is prevented
by refusing the new command, latching `RECORDER_QUEUE_FULL`, and leaving body
motion unchanged.

Exporter rules are:

- It adopts a new manager session only from a valid NONE/sequence-0 heartbeat
  while exporter state is IDLE and `capture_active == false`.
- It never adopts another session while RECORDING or SAVING.
- It keeps every retired session ID for the process lifetime. More than 64
  retired sessions changes the exporter to ERROR and requires restart.
- The expected first command is sequence 1. A command equal to the last applied
  sequence and value is a duplicate: do not reapply it, but republish ACK.
- The same identity with a different command is corruption and enters ERROR.
- A sequence below the last applied value or from a retired session is rejected.
- A forward gap is not applied. ACK repeats the last applied sequence, causing
  the manager FIFO to resend the missing command.
- Commands are state-setting/idempotent: START only moves IDLE to RECORDING;
  STOP_AND_SAVE only moves RECORDING to SAVING; ABORT only acts on RECORDING;
  TOGGLE is deduplicated before it resolves against actual exporter state.

An unadopted manager session does not refresh the adopted session's heartbeat.
If the manager restarts during RECORDING, the old adopted session therefore
times out after 500 ms and the exporter discards that interrupted episode;
once IDLE, the next new-session NONE/sequence-0 heartbeat is adopted. If the
manager restarts during SAVING, the exporter completes or fails that owned
save before adopting the new session. The new manager may start SONIC and
tracking, but reports recording unavailable and refuses all recording controls
until its session is ACKed as adopted and IDLE. A manager-side
`recording_may_be_active` latch is set when START or TOGGLE is enqueued and is
cleared only by a fresh ACK proving IDLE, or ERROR with capture inactive; only
that latch can make a normal tracking-exit request wait for recorder state.

A 500 ms manager-state timeout while actively recording and without a
previously accepted STOP_AND_SAVE aborts the in-memory episode as discarded.
It never promotes an interrupted episode to a successful one.

### Stop, save, and failure outcomes

| Event | Required outcome |
| --- | --- |
| Grip+A/S while recording | Stop appending immediately, ACK `SAVING`, atomically save; ACK `IDLE` only after save and metadata flush succeed |
| Grip+B | Stop appending, save as discarded using existing discard path, then ACK `IDLE` |
| Manager Ctrl+C while recording | Enqueue STOP_AND_SAVE; wait up to 2.0 s for matching `IDLE`; then exit manager without SONIC-stop. Timeout is reported loudly; exporter continues its owned save if alive |
| Manager Ctrl+C while idle | Exit manager immediately without SONIC-stop |
| PICO A+B+X+Y while recording | Enqueue ABORT and allow at most one 20 ms send; SONIC stop/manager exit is never blocked |
| Deploy-terminal O / C++ emergency stop | A `g1_debug` receive age of `>=500 ms`, even if manager heartbeat continues, discards active capture; manager heartbeat age `>=500 ms` has the same result |
| Exporter Ctrl+C | Active capture is saved as discarded; idle exits cleanly |
| Save exception | Preserve the episode buffer and temporary artifacts, publish `ERROR` with capture inactive, block new recordings, and allow tracking to be disabled |

Normal tracking disable is accepted only after a fresh ACK proves IDLE, or
ERROR with `capture_active == false`. Global stop paths are never blocked.

## Episode Data Contract

The hand profile is fixed for an exporter process and recorded in
`script_config`. A frame is never padded with seven zeros when a required hand
field is absent. Missing, stale, wrong-profile, or wrong-provenance hand data
makes that recording frame ineligible and increments a drop counter; it does
not fabricate an observation/action.

### Dex3 episodes

Dex3 retains the existing 43-DoF body-plus-14-hand RobotModel and fields:

```text
teleop.left_hand_joints   float32[7]
teleop.right_hand_joints  float32[7]
```

Both use the Dex3 API order declared above. `observation.state` and
`action.wbc` use RobotModel joint order after the existing explicit API-to-model
permutation. Optical metadata is added for each side: tracking state `int32[1]`,
source epoch `int64[1]`, source timestamp `int64[1]`, and valid/held `bool[1]`.

### Inspire episodes

Inspire uses a profile-specific 41-DoF RobotModel: the same 29 body joints plus
six active joints per side. Per-side model order is:

```text
little_1, ring_1, middle_1, index_1, thumb_2, thumb_1
```

Normalized q6 is converted to model radians only at the RobotModel boundary:

```text
q_model = (1 - q6) * [1.4381, 1.4381, 1.4381, 1.4381, 0.5864, 1.1641]
```

`observation.state float64[41]` uses body measured q plus measured normalized
q6 from fresh `inspire_hand_status`, converted by that equation.
`action.wbc float64[41]` uses body `last_action` plus PC2
`left_applied/right_applied`, also converted. Manager intent remains separate:

| Dataset feature | Shape | Meaning/order |
| --- | ---: | --- |
| `teleop.left_inspire_hand_command` | `float32[6]` | manager live/held q6 in FTP order |
| `teleop.right_inspire_hand_command` | `float32[6]` | same |
| `observation.left_inspire_hand_state` | `float32[6]` | measured counts / 1000 in FTP order |
| `observation.right_inspire_hand_state` | `float32[6]` | same |
| `action.left_inspire_hand_applied` | `float32[6]` | last successful PC2 DDS command |
| `action.right_inspire_hand_applied` | `float32[6]` | same |
| `teleop.left_hand_tracking_state` | `int32[1]` | optical watchdog enum |
| `teleop.right_hand_tracking_state` | `int32[1]` | optical watchdog enum |
| `teleop.hand_sample_generation` | `int64[1]` | manager generation shared with body packet |
| `teleop.inspire_applied_message_seq` | `int64[1]` | PC2 applied q6 sequence |

Inspire episodes do not declare Dex3 q7 teleop fields. The exporter requires
the latest `inspire_hand_status` to match manager session/mode provenance, have
`feedback_healthy == true`, and be younger than 500 ms. An optical `HOLDING`
state is still recordable because it is explicit and the q6 command is real;
stale physical feedback is not.

## Intended Operator Procedure

1. Start `pico_manager_thread_server.py` in optical Dex3 or Inspire mode and
   start the exporter with the identical hand profile.
2. Hold PICO A+B+X+Y to start SONIC.
3. Hold PICO A+X, or press manager T, to start full-body tracking.
4. Hold left grip+A, or press C, to start recording; wait for the RECORDING ACK.
5. Put the PICO controllers on the table. Modes remain latched.
6. Confirm both optical hands report TRACKING, then perform the episode.
7. Pick up the controllers. A lost optical side holds while body motion and the
   other side continue.
8. Hold left grip+A, or press S, to stop and save; wait for IDLE ACK.
9. Repeat steps 4-8 for more episodes.
10. Hold A+X, or press T, to stop full-body tracking.
11. Hang the robot on the crane.
12. Hold A+B+X+Y to stop SONIC. Deploy-terminal O remains an independent stop.

## Diagnostics and Artifacts

Dex3 body packets carry the exact live or held q7 with the matching body
generation. Inspire uses the q6 protocol above. A separate `hand_tracking`
topic records manager diagnostics without expanding the fixed body header:

- `pv`, sample generation, profile, and input mode;
- per-side source epoch/timestamp, binding generation, age, state, validity,
  and reason code;
- exact retargeted and emitted commands;
- optional raw 26-by-7 poses and 26 location flags; and
- recorder command/ACK state.

Console output is transition-based and rate-limited. It says, for example,
`LEFT HAND HOLD: SOURCE_STALE`, `LEFT HAND RECOVERING`, and
`LEFT HAND TRACKING`.

Every qualification run produces:

- a JSON result report with thresholds and pass/fail reasons;
- JSONL transition/fault logs;
- NPZ raw snapshots plus pre/post-filter commands;
- per-frame body/hand/recording provenance;
- simulator video for simulation gates; and
- commanded/measured CSV plus DDS ownership evidence for hardware gates.

## Verification and Qualification Gates

### Automated tests

Unit and integration tests must cover:

- exact `raw[1:26]` canonical ordering and tips 4/9/14/19/24;
- atomic pose/active/timestamp/flags snapshots with no torn fields;
- a frozen left hand becoming stale within 100 ms while body/global/right
  timestamps continue advancing;
- position tracking flags, shape, dtype, finite, geometry, wrist-distance, and
  timestamp-regression rejection;
- independent side hold/recovery and continued body/opposite-hand output;
- five-frame admission, filter reset, 40 ms dt cap, convergence tolerances,
  limits, and no double low-pass filter;
- manifest verification, audited config translation, unknown-key rejection,
  and real four-retargeter URDF/config initialization plus finite calls;
- named Dex3 and Inspire permutations and independent motion of every intended
  human digit;
- 40 ms debounce, 200 ms chord window, stagger, bounce, subset suppression,
  controller loss/rearm, and every transition-table row;
- native u8 serializer/decoder interoperability and unknown-dtype rejection;
- recording session adoption, duplicate, conflict, gap resend, retired
  session, exporter restart, manager restart, ACK staleness, queue-full, save
  failure, Ctrl+C, and emergency-stop outcomes;
- exact Inspire q6/status field order, shape, provenance, session restart,
  watchdogs, and feedback gating; and
- profile-specific dataset shapes with no missing-field zero substitution.

The production-shaped packed-message test requires every unpadded header to be
at most 1,200 bytes, leaving 80 bytes reserve in the fixed 1,280-byte header,
and verifies Python/C++ byte-level interoperability for affected messages.

### Gate D0: recorded replay

- Dataset: at least 10 minutes and at least 1,000 frames each of open, fist,
  pinch, every individual digit, crossing hands, occlusion, controller
  put-down, and pickup.
- Pass: zero non-finite/out-of-limit commands; every injected single-side
  freeze enters HOLDING no later than 120 ms after its last advancing source
  timestamp; body output gap p99 <= 25 ms and maximum <= 60 ms; opposite-hand
  output continues at >=45 Hz; retarget/filter processing p99 <= 5 ms on the
  deployment host.
- Abort: any wrong finger/order, body mode change caused by hand invalidity,
  command discontinuity beyond configured rate, or provenance acceptance from
  a retired source.

### Gate D1: simulation and recording

- Run at least 20 minutes and 50 full cycles per independently moved digit,
  including unilateral/bilateral loss and recovery and 20 recorded episodes.
- Pass: no collision/limit violation; commanded rate remains within limits;
  all 20 episodes reach durable IDLE ACK and pass schema validation; zero
  fabricated zero-hand frames; body and opposite hand satisfy D0 continuity.
- Dex3 must pass before any attached-hand test. Inspire repeats D0/D1 only
  after Dex3 passes.

### Gate D2: Dex3 on crane

Robot is crane-supported, workspace clear, deploy O operator stationed, and
only one hand is powered first.

1. Stage A: manager max 0.5 rad/s and 20% of each joint's open-to-close range;
   50 open/close cycles per thumb/index/middle on each side.
2. Stage B: after Stage A artifacts pass, max 1.0 rad/s and 50% range; repeat 50
   cycles per digit and 50 unilateral loss/recovery injections.
3. Stage C: after inspection, production manager max 2.0 rad/s and full
   configured joint limits; 100 mixed two-hand cycles and the complete
   recording procedure for 20 saved episodes.

Each stage passes only with zero unexpected body stops, zero joint-limit or
2.5 rad/s downstream-clamp hits, tracking-state latency <=120 ms, measured
joint error <=0.20 rad for 99% of settled samples, and no visible collision or
oscillation. Any clamp hit, driver fault, wrong digit, tracking/body coupling,
collision, abnormal sound/heat, or operator stop aborts the stage and requires
inspection; it is not retried by widening limits.

### Gate I0/I1: Inspire monitor and crane

- Monitor-only PC2: 20 minutes, zero malformed/provenance/ownership faults,
  100% correct q6 ordering, and all injected 250 ms transport timeouts enter
  OPENING on the next 10 Hz control tick.
- Crane Stage A: fixed `[800,1000]` counts, at most 5 counts per 100 ms tick;
  50 cycles per motor one side at a time.
- Crane Stage B: after Stage A inspection, `[500,1000]`, at most 5 counts/tick;
  50 cycles per motor and 50 optical single-side holds.
- Crane Stage C: after Stage B inspection, `[0,1000]`, at most 10 counts/tick;
  100 mixed two-hand cycles and 20 saved episodes.

Every attached Inspire stage retains the DDS sole-writer proof, fresh zero-error
feedback, and deploy O operator. Pass requires zero ownership/fault latches,
zero envelope/slew violations, measured settled error <=75 counts for 99% of
samples, and worst-case safe-open completion, measured from the first OPENING
decision, within 4.1 s in Stage A, 10.1 s in Stage B, and 10.1 s in Stage C.
It also requires no wrong motor, collision, heat, or abnormal sound. Any foreign writer,
feedback error/staleness, loop interval >110 ms, envelope/slew violation,
wrong motor, operator stop, collision, or save/provenance fault aborts.

## Acceptance Criteria

- Optical joint 1 is treated as wrist; canonical input is exactly raw 1-25.
- A frozen side becomes stale at exactly 100 ms and is held on the first 50 Hz
  manager tick thereafter, with a measured last-sample-to-hold bound of 120 ms,
  without stopping body motion or the other hand.
- Dex3 thumb/index/middle and all five Inspire fingers respond independently
  in their named output orders.
- Recovery has exact admission, filter reset, dt cap, and convergence rules,
  with one smoothing owner.
- Official retargeting 0.4.6 loads verified real Unitree assets through the
  audited translation and passes real initialization smoke tests.
- Controller chords, keyboard T/C/S, PICO A+B+X+Y, and deploy-terminal O behave
  exactly as specified.
- Recording commands survive drops and manager/exporter restarts without
  duplicate state changes, and manager state is governed by fresh exporter ACK.
- Inspire q6, physical feedback, and episode action/observation data are exact,
  versioned, provenance-bound, and never replaced by seven zeros.
- D0/D1 pass before Dex3 hardware, Dex3 passes before Inspire, and every
  hardware stage meets its measurable thresholds and artifacts.

## Non-Goals

- Optical wrist pose does not replace the fused body tracker arm target.
- No automatic optical-to-controller articulation fallback.
- No simultaneous second writer for G1 body LowCmd, Dex3, Inspire DDS, or
  Inspire Modbus.
- No weakening of global emergency-stop or backend watchdog authority.
- No Inspire DFX backend in this delivery.
- No claim that manager hold protects against power, process, DDS, driver, or
  hostile-writer failure; backend and physical safety layers remain required.

## Normative External References

- [XRoboToolkit hand schema](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/blob/main/README-CN.md#3-%E6%89%8B%E5%8A%BF)
- [OpenXR `XrHandJointEXT` enumeration](https://registry.khronos.org/OpenXR/specs/1.0/man/html/XrHandJointEXT.html)
- [`dex-retargeting` v0.4.6 configuration loader](https://github.com/dexsuite/dex-retargeting/blob/v0.4.6/dex_retargeting/retargeting_config.py)
- [Pinned Unitree Dex3 configuration](https://github.com/unitreerobotics/xr_teleoperate/blob/934361718c5886356e19f9068f4536f04d2b9feb/assets/unitree_hand/unitree_dex3.yml)
- [Pinned Unitree Inspire configuration](https://github.com/unitreerobotics/xr_teleoperate/blob/934361718c5886356e19f9068f4536f04d2b9feb/assets/inspire_hand/inspire_hand.yml)
- [Unitree `dex-retargeting` fork declaration](https://github.com/unitreerobotics/xr_teleoperate/blob/934361718c5886356e19f9068f4536f04d2b9feb/.gitmodules)
