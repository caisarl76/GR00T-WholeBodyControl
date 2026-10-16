# Native handover lift-first return

The production ready-return path now accepts a named raised right-arm waypoint
before the final handshake-ready target. It holds the entry fingers during the
lift, requires distinct measured upper-body/heading settling for 0.5 seconds,
then opens the hand and approaches final ready. Final all-joint settling and
the total 15-second deadline remain unchanged.

Handover profiles require `raised_right_arm_joints` and
`ready_right_arm_joints`, each with seven named native joints. Both poses are
validated against native bounds. `ready_pose_reviewed` covers the whole path.
Older handover profiles without a raised target are rejected. The separate
placement profile and ordinary standing-reset targets are preserved.

Verification: 664 `gear_sonic/tests` passed; the paired VoLoAgent suite passed
745 with 19 skips, and explicit cross-process software tests passed 13. A fresh
review found an early-credit dwell defect; its regression failed before the fix
and passed with the final suite afterwards.

Motion acceptance remains failing. The corrected production SONIC/MuJoCo run
timed out in the lift: right shoulder yaw was about 0.20 rad from its target.
The simulated hand stayed closed, with zero sampled hand/forearm desk contacts
and at least 37.01 mm conservative sampled gap. This does not qualify actual
handover entry poses, swept/person clearance, balance or the full VLA cycle.
Both hardware flags remain false. Exact native feedback/reference tracing and
repeatable settling are the next diagnostic work; limits were not relaxed.

The video, raw runs, source hashes, review record and remaining jobs live in the
VoLoAgent feature branch at
`docs/artifacts/g1_handover_20261014_native_lift/README.md`.
The native environment preflight reports low disk space and an unmaterialized
decoupled-WBC balance LFS model. The existing SONIC assets were usable and CUDA
passed outside the sandbox. No hardware commands, deletions or downloads ran.
