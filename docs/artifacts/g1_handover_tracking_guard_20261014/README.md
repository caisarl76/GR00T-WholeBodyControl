# Reset reference tracking guard

Exact paired simulation telemetry reproduced a reference lead of 0.20855 rad
when a measured arm joint fell back. The old reset update stopped moving toward
its goal but held a reference beyond the intended 0.15 rad bound.

The reset now intersects measured lead and right-arm joint-limit intervals and
approaches them at the unchanged 0.5 rad/s rate. If both bounds cannot be met,
it rejects the update without publishing a new reset command. The native loop
invalidates old policy actions, interrupts and requests its existing measured
hold. Out-of-bound right-arm measurements cannot construct a hold/reset, so
recovery cannot publish that invalid angle. Operator resets pause and hold too.
The existing nominal targets, 0.05 rad tolerance, 0.5 s dwells and 15 s total
deadline are retained.

Verification: 674 native tests, 745 VoLoAgent tests and 13 explicit memory-only
cross-process tests passed. A fresh review's outward joint-limit counterexample
was observed failing and fixed with both-sign regression tests and measured-hold
validation. The final two real SONIC/MuJoCo traces respected rate/lead bounds on
all 31 joints, but still timed out without measured settling. No hardware
commands ran or hardware review flags changed.

The required broad [environment preflight](environment-preflight.log) ran outside
the sandbox. CUDA/TensorRT passed, but the all-purpose check failed on the
training-only Python/dependency requirements, the unused decoupled-WBC LFS
balance asset and its 10 GiB disk recommendation (9.6 GiB then). The component
trials used the existing inference/simulation environments, with no training.

The last decoder trace found about 0.181 rad between the decoded right-shoulder
motor target and measured angle. The recorded poses' gravity estimate is
consistent with investigating arm dynamics; gains/masses are not yet established
as wrong. Source-15's alternate lift also failed to preserve the other arm and
was not adopted. Effective encoder targets and model/actuator calibration are
the next diagnosis, followed by post-handover clearance and a connected VLA cycle.

Paired streams, source hashes, comparison plot, corrected-run video and the full
report are on the VoLoAgent branch in
`docs/artifacts/g1_handover_20261014_tracking/README.md`.
