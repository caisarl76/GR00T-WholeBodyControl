# Inspire FTP MuJoCo Teleoperation Design

## Purpose

Integrate the two Inspire FTP dexterous hands mounted on the Unitree G1 into
the GEAR-SONIC teleoperation stack, beginning with a hardware-free MuJoCo
validation path. The same six-motor command contract must later drive the FTP
DDS service on PC2 and accept six-motor VLA actions without changing SONIC's
29-DoF body policy.

This phase succeeds when PICO 4 Ultra controller inputs can drive both Inspire
hands in MuJoCo while the existing SONIC body loop remains independent and
stable. Real-hand commands are explicitly outside this phase.

## Evidence and Pinned Contracts

The design is based on four independent sources:

1. Unitree's official, pinned
   [`g1_29dof_rev_1_0_with_inspire_hand_FTP`](https://github.com/unitreerobotics/unitree_ros/blob/4ddbf6df0aa5bf8c8789d3edfa83e5e3ca45fe48/robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf)
   description declares twelve kinematic hand joints per side. The official
   G1 description table identifies this as the FTP model. The pinned Unitree
   revision is `4ddbf6df0aa5bf8c8789d3edfa83e5e3ca45fe48` from 2026-08-24.
2. Unitree's
   [`robot_hand_inspire.py`](https://github.com/unitreerobotics/xr_teleoperate/blob/main/teleop/robot_control/robot_hand_inspire.py)
   and the archived checkout at
   `/mnt/data/jihun/backup/physical_ai_IL/repos/xr_teleoperate` establish the
   six-motor command order, normalized range, DDS topics, and retargeting
   behavior.
3. The verified PC2 installation uses `Headless_driver_double.py` with one FTP
   service per hand. It accepts `angle_set` values from 0 through 1000 on
   `rt/inspire_hand/ctrl/{l,r}` and publishes measured values on
   `rt/inspire_hand/state/{l,r}`.
4. NVIDIA's response in
   [GR00T-WholeBodyControl issue 159](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/159)
   states that dexterous hands bypass SONIC and are controlled directly by the
   teleoperation or VLA interface. The released SONIC policy does not control
   hand joints.

The archived `dfx_inspire_service` and Isaac Lab bridge use a different,
combined `rt/inspire/{cmd,state}` protocol. They are useful kinematic
references but are not wire-compatible with the installed FTP driver and will
not be reused.

## End-to-End Architecture

```text
PICO 4 Ultra / XRoboToolkit
  controller poses + trigger/grip
                |
                v
pico_manager_thread_server.py
  existing body/ankle mapping
  Inspire controller profile -> normalized left[6], right[6]
                |
                | ZMQ pose/planner message, latest value
                +------------------------------+
                |                              |
                v                              v
      SONIC body processes             Inspire hand adapter
      (29-DoF, unchanged)               validate + watchdog
                |                              |
                v                              v
       G1 body in MuJoCo            6 motors -> 12 joints/hand
                                               |
                                               v
                                      Inspire hands in MuJoCo
```

The C++ deploy process is launched with Dex3 hand output disabled. It may still
receive the hand fields because they are part of the existing ZMQ message, but
it does not own or reinterpret them. The Python Inspire adapter subscribes to
the same message and owns the simulated hand actuators.

The later real-robot backend will replace only the final MuJoCo plant with an
FTP DDS publisher on PC2. It must not open the hand TCP/Modbus connection;
`Headless_driver_double.py` remains the sole Modbus owner.

## Canonical Six-Motor Contract

Every boundary uses a `float64[6]` vector in this order:

| Index | Hardware motor | Active MuJoCo joint |
|---:|---|---|
| 0 | pinky | `{left,right}_little_1_joint` |
| 1 | ring | `{left,right}_ring_1_joint` |
| 2 | middle | `{left,right}_middle_1_joint` |
| 3 | index | `{left,right}_index_1_joint` |
| 4 | thumb bend | `{left,right}_thumb_2_joint` |
| 5 | thumb rotation | `{left,right}_thumb_1_joint` |

The normalized convention is hardware-compatible:

- `0.0` means closed;
- `1.0` means open; and
- the PC2 FTP command is `round(normalized * 1000)`.

The MuJoCo active-joint conversion follows the pinned official FTP URDF:

```text
finger radians         = (1 - u) * 1.4381
thumb-bend radians     = (1 - u) * 0.5864
thumb-rotation radians = (1 - u) * 1.1641
```

The XR retargeter first maps hand tracking into older generic ranges of 1.7,
0.5, and `[-0.1, 1.3]`, then normalizes those values before sending them to the
FTP driver. Those intermediary retargeting radians are not the G1 FTP model's
physical joint angles. Simulation therefore consumes the normalized command
and maps it into the pinned FTP URDF limits above. Normalized replay remains
identical across simulation and hardware without conflating the two angle
spaces.

## Twelve-Joint Kinematics

Each hand has six active joints and six dependent joints. MuJoCo equality
constraints express the URDF mimic relations:

```text
index_2  = 1.0843 * index_1
middle_2 = 1.0843 * middle_1
ring_2   = 1.0843 * ring_1
little_2 = 1.0843 * little_1
thumb_3  = 0.8024 * thumb_2
thumb_4  = 0.9487 * thumb_3
```

There are twelve hand actuators total, one for each active joint. Dependent
joints have no independent actuator. All runtime lookup is by explicit joint
and actuator name; no joint-number arithmetic or substring ordering is valid
for this model.

The maintained model is derived from the pinned official FTP description and
the matching `left_*` and `right_*` Inspire meshes already tracked under
`gear_sonic/data/robots/g1/meshes`. The untracked model found in
`/home/jihun/work/SIMPLE` uses the older `L_*`/`R_*` kinematics, has its
equality block commented out, and independently actuates dependent joints. It
is not used by this integration.

## PICO Controller Profile

The current saved GEAR-SONIC and archived XR controller paths implement a
single binary trigger command and output seven Dex3 joints. They do not contain
a completed six-motor Inspire mapping.

The new profile consumes calibrated close signals in `[0, 1]`:

```text
u[pinky:ring:middle:index:thumb_bend] = 1 - trigger_close
u[thumb_rotation]                    = 1 - grip_close
```

This produces:

| Trigger | Grip | Normalized hand command |
|---:|---:|---|
| 0 | 0 | `[1, 1, 1, 1, 1, 1]` open |
| 1 | 0 | `[0, 0, 0, 0, 0, 1]` fist without thumb rotation |
| 0 | 1 | `[1, 1, 1, 1, 1, 0]` thumb rotation only |
| 1 | 1 | `[0, 0, 0, 0, 0, 0]` fully closed command |

The input adapter applies configurable lower/upper calibration and deadzones.
It rejects non-finite values. No assumption about raw XRoboToolkit ranges is
promoted to a hardware guarantee until a live PICO capture confirms them.

`compute_hand_joints_from_inputs` remains the single shared mapping point for
the pose and planner loops. A named `hand_profile` selects existing Dex3
behavior or the new Inspire behavior; the default stays Dex3 for backwards
compatibility.

## Command Receiver and Safety State Machine

The MuJoCo adapter subscribes to `pose` and `planner` ZMQ messages using a
one-value latest-message buffer. It accepts a hand pair only when both vectors:

- contain exactly six values;
- contain only finite numbers; and
- lie exactly within `[0, 1]` without silent clipping.

The safety states are:

```text
STARTUP -> OPEN
OPEN/ACTIVE + valid command -> ACTIVE
ACTIVE + stale for 250 ms -> OPENING
OPENING -> rate-limited open command
OPENING + valid command -> ACTIVE
invalid command -> reject and retain current state
```

Startup never reuses an uninitialized or zero vector because zero is fully
closed. A stale source does not cause a discontinuous jump: the adapter ramps
to open using per-motor normalized speed limits derived from the official
1-rad/s active-joint velocity:

```text
finger:         1.0 / 1.4381 normalized units/s
thumb bend:     1.0 / 0.5864 normalized units/s
thumb rotation: 1.0 / 1.1641 normalized units/s
```

For the simulation phase these values are defaults with test coverage. Any
later hardware tuning must be explicit configuration and start slower.

## Simulator Boundary

The existing `DefaultEnv` incorrectly assumes:

- hand motor count equals hand joint count;
- joint IDs imply qpos, qvel, and actuator indexes;
- hand joints contain `left_hand` or `right_hand`; and
- every hand joint is independently actuated.

The integration replaces those assumptions with a named hand layout object.
It resolves:

- all 12 left and 12 right kinematic joint IDs;
- the six active qpos/qvel addresses per hand;
- the twelve named actuator IDs; and
- conversions between normalized commands and measured active joint angles.

Body lookup and torque application remain unchanged in behavior. Hand commands
are position targets applied through the model's named hand position
actuators, while the Unitree SDK bridge continues to carry only the 29 body
motors. Dex3 DDS hand publishers/subscribers are not initialized in Inspire
mode.

MuJoCo joint limits are soft constraints, so measured active-joint state is
projected onto the physical range before it is reported in the normalized
hardware convention. Command validation remains strict, and the plant records
the largest projection error for diagnostics. The generated model also
excludes the bilateral palm/`thumb_2` mesh pairs that overlap at the official
zero-angle open pose, and uses stiff equality parameters for the mechanical
mimic couplings. All other contact pairs remain enabled.

The Inspire configuration uses:

```text
NUM_HAND_MOTORS: 6
NUM_HAND_JOINTS: 12
HAND_TYPE: inspire_ftp
```

and points to a dedicated Inspire scene. The existing Dex3 configuration and
scene remain untouched.

## Launcher Integration

Simulation launch adds `--hand-profile inspire_ftp`. That option is propagated
to:

1. `pico_manager_thread_server.py`, selecting the six-value PICO mapper;
2. `run_sim_loop.py`, selecting the Inspire scene and Python hand adapter; and
3. `gear_sonic_deploy/deploy.sh`, adding `--disable-dex3-hands` to the C++
   process.

The launcher rejects an Inspire profile unless all three processes receive a
consistent configuration. This prevents the C++ Dex3 controller and Python
Inspire adapter from competing for hand ownership.

## Test and Evaluation Ladder

Verification proceeds from the smallest boundary outward:

1. Pure mapping tests confirm order, inversion, calibration, clipping policy,
   and invalid-input rejection.
2. Pure contract tests confirm normalized/radian round trips at open, closed,
   and mid-range commands.
3. Model-load tests require 29 body joints, 24 hand joints, 41 actuators, 12
   hand equality constraints, and the two verified open-pose contact
   exclusions.
4. A headless per-motor sweep confirms only the intended finger chain moves,
   the dependent-joint ratios hold, values remain within limits, and no state
   becomes non-finite.
5. Receiver tests confirm latest-value behavior, six-value validation, startup
   open state, stale-command ramp-to-open, and command recovery.
6. Existing MuJoCo smoke tests run against the unchanged Dex3 profile.
7. A contact-enabled hand regression records finite state, joint-limit error,
   contact count, and maximum mimic error through an open/close/open cycle. A
   separate bounded full-stack run verifies the TensorRT SONIC body loop,
   loopback DDS, C++ Dex3 disable flag, and the same scripted ZMQ hand cycle.
8. A live PICO-to-MuJoCo session is run only with user approval because it
   opens a viewer and depends on external headset state.

No real DDS hand publisher is started during this ladder.

## Hardware Promotion Gate

Promotion to PC2 is a separate change and requires all of the following:

- the complete simulation ladder passes;
- a live PICO capture establishes trigger/grip raw ranges and neutral noise;
- normalized commands and measured simulated state are recorded in the same
  six-motor order used by the FTP SDK;
- the real bridge defaults open and demonstrates watchdog behavior without a
  hand attached; and
- the first attached-hand test is hand-only, low-rate, and conducted with the
  G1 arms stationary.

Only after that gate may the MuJoCo backend be exchanged for the PC2 FTP DDS
backend. SONIC body inference remains 29-DoF throughout. Future VLA inference
publishes the same six normalized hand actions and therefore does not require a
second hand-control path.
