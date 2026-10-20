# Right Dex3 posture evidence for handover

The native harness now attaches `right_hand_state`, `right_hand_sample_at`
and `right_hand_sample_index` to camera observations. The four pose values are
open, closed, partial and unknown. Classification uses measured DDS joints,
not action targets, in thumb-first native order.

The C++ output interface publishes the original right-hand DDS receipt and a
Linux boot/time-namespace clock identity. Python accepts that receipt only
when its clock identity matches, its age is at most 0.5 s, and its difference
from camera receipt is at most 0.1 s. Repeated logger packets preserve the
same hand sample ID. Missing metadata or a remote/unmapped clock produces
unknown. Update/rebuild the C++ controller with the Python harness; older
binaries cannot supply this evidence.

Open keeps the existing ±0.05 rad tolerance. Provisional closed rules require
four index/middle angles ≥0.35 rad and thumb1/thumb2 ≤−0.25 rad. XML joint
bounds plus 0.05 rad measurement margin reject implausible readings. These
rules describe finger posture, not object contact, stability or an empty hand.

The paired VoLoAgent integration requires two affirmative offer judgments
with distinct aligned open-hand samples and retains visual failure checks.
It supplies pose only to the offer prompt. Hardware profiles remain disabled
and no robot commands were sent in this work.

The complete recorded Qwen comparison and verification are in the paired
VoLoAgent branch at `docs/artifacts/g1_handover_20261013_hand_state/README.md`.
