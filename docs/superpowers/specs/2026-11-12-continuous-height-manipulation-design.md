# Continuous joystick-controlled manipulation and walking

Date: 2026-11-12
Status: Approved by the user on 2026-11-12 as the basis for revising the PRD and implementation plan. Manual height/waist control is user-selected. Runtime implementation and the replacement recovery experiment remain subject to the gate and review requirements below. No new validation is claimed.

Related: [current PRD](2026-11-09-pico-manipulation-mode-prd.md), [current plan](../plans/2026-11-09-pico-manipulation-mode.md).

## Intent and user decisions

Complete one recorded episode: walk to a table or cabinet; stop; lower and bend with the joysticks; reach and grasp using controller wrist poses and hand primitives; raise or straighten while retaining the grasp; turn and walk to the destination; adjust height, reach and place. Also support transferring a free object from a low surface to a high surface while the feet remain in place.

Height and waist remain joystick-controlled throughout stationary manipulation, including while holding an object. The user selected manual recovery over an automatic stand-up action. Arms may remain raised or be manually positioned beside the body before walking. Body adjustment is normal manipulation, not an exceptional recovery operation.

The first validation target is a free object the G1 can grasp. Record actual mass, dimensions, grasp and carry pose in each validation; no universal payload capability is assumed. Drawer/door interaction is also desired and requires separate contact-control validation.

## Approaches

1. **Manual body adjustment with temporary walking readiness — selected.** The operator controls height, waist and arms. A supervisor refuses locomotion from an unsuitable configuration and can permit it after correction. Low-to-high placement works without requesting walking.
2. **Automatic prepare-to-walk motion — not selected.** An action would raise/straighten the body and possibly arrange the arms. It needs additional object-clearance planning and can interfere with a desired placement height.
3. **Replace the balance controller — contingent.** Initially test the existing planner and SONIC within a measured posture/load range. If they cannot maintain balance and tracking, controller adaptation or replacement is required. An input gate cannot create missing control capability.

## Stability and settling

Controller wrist motion changes arm references, not the desired height, waist or locomotion. SONIC retains balancing authority. The goal is bounded unintended motion and successful manipulation, rather than physically immobile legs and waist. Hard waist locking remains experimental and off in the first loaded validation.

Settling is the physical response after a reference stops changing: the robot catches up and residual motion decreases. A motionless crouched or displaced robot is not necessarily ready to walk; a balanced robot can make small continuing corrections.

Do not require walking readiness to accept height, waist or wrist commands during stationary manipulation. Those commands intentionally move the robot. Rate/workspace limits, feedback health and balance protections still apply. Evaluate a dwell only when considering locomotion, after relevant requested and applied references stop changing.

The old 4.5 s rest comes from a specific unloaded height experiment. It is not a universal settling time for loaded reaching or rising. The failed endpoint-displacement/mean-gyro detector and the earlier failed peak detector remain disabled.

## Architecture

Retain the PICO backend adapters, planner-message transport and planner/SONIC backend. Extend the planned manipulation streamer with persistent state and transitions; no additional control service is needed.

Controller poses -> wrist targets -> arm-only IK -> upper-body reference.
Joysticks -> requested height/waist or locomotion -> bounded references.
Hand primitive events -> persistent grasp state -> hand commands.
Deploy combines references, runs the planner/SONIC and handles input loss.
Measured feedback informs transition readiness; one recorder spans every phase.

The supervisor owns requested height/waist, controller anchors, arm targets, hand primitive/last-sent command, mode and walking transition state. The deploy owns applied control state and communication-loss handling. Record requested intent, transmitted reference and measured response separately. An acknowledgment proves command processing, not physical completion.

## Operator behavior

| State | Body and hand control | Locomotion |
|---|---|---|
| Stationary manipulation | Joystick height/waist; controller wrists; persistent grasp | Zero movement and fixed heading reference |
| Carry selection / pending walk | Hold selected arm/hand references; return to manipulation for adjustment | Check readiness; never reposition the body automatically |
| Walking | Hold selected carry references and grasp | Joystick movement/turn within the validated range |
| Controlled stop | Preserve grasp and appropriate body references | Complete the tested stop before manipulation resumes |
| Fault hold | Preserve last applied posture/grasp as control permits; report fault | Refuse new walking; gait faults use the tested stop |

Proposed controls retain the manipulation mapping: left Y changes height; right Y changes pitch and right X waist yaw; controller motion controls wrists. B+Y selects manipulation/carry while preserving references and recording. A toggle is not a walk command. A refused walk shows its reason; the operator can return to manipulation to correct the posture.

Simultaneous slow wrist and body adjustment remains permitted during stationary manipulation within the validated range. No stand-up or carry pose is required between grasp and higher placement if the robot is not walking. During actual walking, hold the selected body/arm references. Height or arm repositioning first requires a controlled stop and manipulation mode.

## Wrist and grasp behavior

Retain torso-relative wrist targets for the initial free-object implementation. With controllers still, rising carries the wrists upward; waist rotation also moves and rotates them through space. Controller movement adds wrist motion relative to the torso. This supports lifting with body and arms together. It does not guarantee world-fixed wrist position or object orientation.

Before rising with an object, the operator withdraws it clear of the shelf/table/cabinet. The operator owns that clearance decision in the first implementation; do not present it as automatically sensed. Validate in a known clear workspace with the held object's geometry represented. Arms may stay raised within a tested carry range or be manually moved down; no automatic tuck is inserted.

Each hand has an explicit primitive and held command. Preserve the actual last-sent command through posture changes, mode changes, recentering and communication recovery. A mode change never means OPEN. A new deliberate primitive event controls release. Define input edges and re-arm so an already released trigger on re-entry cannot release the grasp. The current 5% input tolerance is insufficient for discrete primitives; 0.51 and 0.49 cross the current trigger boundary. Validate primitive transitions and physical object retention.

Arm IK remains arm-only. Report solver success, Cartesian residual and measured tracking error separately. A successful QP or unsuccessful alternate seed proves neither reachability nor collision clearance. Bound applied joint steps and clearance in the supported workspace; preserve continuity through re-anchoring and reseeding.

Drawer/door use requires separate contact behavior, a continuous target-frame handoff and validated compliance/force response. World/contact-relative control requires a verified torso/base pose estimate. Do not treat a handle attached to the environment as a free carried object. Sensors and controller choice for that extension remain a later design task.

## Recoverable walking eligibility

The intended product replaces the session-long height-history latch with readiness that can return after manual correction. Sending -1 or restarting a process is not proof of recovery.

The new, separately validated readiness protocol must establish:

- A measured configuration in the tested carry range: height/posture, waist/tilt, arm extension, and the selected validated object/load setup. A high stationary placement pose need not be walkable.
- Completed relevant height, waist and arm transitions, bounded tracking errors and residual dynamics over a dwell.
- Fresh complete feedback and valid fresh controllers for every active movement/turning input.
- Completed controlled stop, input re-arm and watchdog conditions. Optional waist-lock release needs fresh acknowledgment; reassess readiness after release.
- A fresh deliberate walk request. A held request must not fire later when a refusal clears.

Invalidate readiness on relevant reference changes, disturbances outside the tested range, tracking loss, stale feedback or watchdog events. Before gait starts, return to pending/hold; during gait, use gait-specific protections and the tested stop. Do not evaluate ordinary gait motion with a stationary predicate.

Use explicit-height static control during manual adjustment, and -1 for standing-height locomotion only after a validated handoff. Preserve requested height and separately log temporary transmitted overrides. Do not restore an old lowered posture automatically after walking. Initial locomotion remains at standing height; crouched walking is not implied.

### Feedback feasibility and release gate

Joint/IMU feedback does not establish pelvis translation or foot support/slip in every condition. The selected first scope assumes a flat non-slip floor, declared object, and feet planted after a completed controlled stop. C1 checks joint/tilt posture, FK conditional on that assumption, tracking and non-cancelling dynamics. Sim truth audits support/foot displacement offline; it is never a gate input. Hardware slip remains an explicit residual risk; this scope does not require adding slip sensing. In-scope slips/falls are failures, not cases to remove after observing their outcome.

The planning phase must first propose a separate experiment with sensing assumptions, causal signal definitions, candidate algorithms, posture/load range, detection latency, disturbances, usefulness criteria and held-out validation. Approve that protocol before tuning or collecting validation data. Sim ground truth scores outcomes; it must not feed a deployable decision. If relevant failures cannot be distinguished using available feedback, additional sensing or control capability is needed. This design supplies no invented readiness thresholds.

Evaluate actual loaded stop/adjust/rise/walk transitions as well as detector decisions. Misclassifying a 3 mm band is not itself a demonstrated unsafe gait. Set task-relevant outcome limits before validation, allowing normal balance corrections. Unknown feedback cannot permit walking. A failed recovery protocol prevents release of the complete workflow; a permanently disabled carry walk does not meet the product goal.

## Fault behavior and recording

While stationary at explicit height, input loss must retain the height-effective static mode and applied upper-body/grasp references. Forcing IDLE with the same height scalar can cause a rise because G1 found IDLE ignores height. Walking faults use the separately tested controlled stop. Recovery preserves continuity, clears stale movement requests and requires deliberate re-arm. Holding a reference is not a guarantee of physical immobility under arbitrary load.

Review amendment (2026-11-12): the user requires solo headset recording. Reserve a held two-thumbstick-click gesture in manipulation sessions for START or STOP_AND_SAVE, with centered-stick entry, immediate motion suppression from the first click, and release/re-arm before further motion. Startup HOLD/OFF can record before the first walk; after gait require a completed controlled stop. Face-button motion chords retain priority. The PRD defines timing and cancellation. Keep a working keyboard publisher as fallback; keyboard-only operation is no longer the selected design. Suppress A/B+grip recording shortcuts from their initial button edge and retain default-session controls.

Recording uses explicit START/STOP_AND_SAVE actions and exporter-published state; duplicates in the current state do not toggle. No transaction leases, revisions or restart reconstruction are required. A confirmation wait releases after 1 s with an unconfirmed-boundary warning; keyboard save/discard works directly if the manager is down. Give the headset operator an audible confirmation through a verified audio route. Retain grasp throughout a recording gesture. A valid-marked one-sample trigger dropout must also not release an object: OPEN requires sustained fresh low-trigger input, and unverified backend freshness disables hand release.

Keep one episode ID across all phases. Record controller timestamps/validity, mode/transition, requested and sent references, hand primitive, measured state, readiness/refusal/fault state and command validity. Correct the exporter's mode-5-only planner filtering so FROZEN and manipulation commands retain their labels. Unavailable data must not silently become valid zero/default commands.

## Validation and delivery

Current implementation is gate tooling and an isolated arm IK prototype. Phase B is not integrated. Preserve G1's failure, G1b's limited handoff evidence and the withdrawn still checks. G0/G3 operator evidence and teleop-PC G2 remain outstanding under Task 6.

After design approval, revise the PRD and plan together and execute prerequisite work within the agreed gate order. Pre-register the replacement recovery experiment before validation. Distinguish a prototype from a released complete workflow.

Acceptance includes both one-recording sequences through the real streamer/deploy:

1. Walk, stop, manually lower/bend, reach/grasp, withdraw, manually raise/straighten, choose raised or lowered carry arms, turn/walk, stop, adjust and place.
2. With stationary feet, lower/reach/grasp and transfer a free object to a higher surface using body and wrist inputs, then deliberately release. No walk request is needed.

Use measured object mass/geometry, defined contact/friction, multiple working heights and arm poses, repeated transfers and separate development/validation data. Check retention, placement, clearance, falls/slip, bounded unintended motion, tracking, reference continuity, readiness delay/usefulness and uninterrupted recording labels. Inject controller/network/feedback faults during low grasp, rising, handoff and walking. Test primitive threshold crossings and all recording/control overlaps.

Builder-owned planning choices: code organization in existing components, primitive input-event/re-arm mechanics, deterministic fixtures, specific object/range candidates, and the proposed sensing/validation protocol. These need verification rather than assumed hardware capability. New hardware or a different balance controller requires a further design decision.
