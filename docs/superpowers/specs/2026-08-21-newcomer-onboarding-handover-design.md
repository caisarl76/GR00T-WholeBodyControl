# Newcomer Onboarding Teammate Handover Design

Date: 2026-08-21

## Purpose

Create a concise repository-local handover for the teammate who will review,
adopt, and maintain the new PC2/workstation newcomer-onboarding workflow. The
handover complements the user-facing runbook; it does not duplicate the
runbook's runnable commands.

## Destination and Audience

Write the handover at:

```text
docs/superpowers/progress/2026-08-21-newcomer-onboarding-handover.md
```

The audience is a teammate familiar with the repository who needs enough
context to review the change, fill in lab-owned configuration, execute the
documented verification, and help a newcomer use the runbook safely.

## Sources of Truth

The handover links to, rather than restates:

- `docs/source/getting_started/newcomer_onboarding.md` for the operational
  procedure;
- `docs/source/getting_started/gear_sonic_deployment_9c0ff22.sha256` for pinned
  deployment-artifact integrity;
- `docs/superpowers/specs/2026-08-19-newcomer-onboarding-design.md` for the
  approved design rationale;
- `docs/superpowers/plans/2026-08-20-newcomer-onboarding.md` for the
  implementation plan.

## Required Content

The handover contains these compact sections:

1. **Outcome and scope** — what the branch adds and which machine runs each
   component.
2. **Canonical documents** — direct repository-relative links to the sources
   of truth.
3. **Safety-critical decisions** — camera and manager readiness before
   `ACTUATE`, the independent hardware-stop prerequisite, confirmed deployment
   shutdown before remediation, and episode finalization/discard behavior.
4. **Lab-owned inputs** — the placeholders, robot-owner stop procedure,
   physical no-hands confirmation, network access, and camera identity that
   cannot be supplied by the repository.
5. **Verification evidence** — the focused tests, command-fence parsing,
   checksum validation, privacy scan, and forced Sphinx build used for this
   branch.
6. **Teammate onboarding checklist** — review the runbook, supply lab values,
   rehearse simulation and stop paths, validate both machines at the same
   revision, and perform a supervised first run.
7. **Branch and publishing context** — feature branch, base branch, final
   implementation commit, and draft-PR intent.

## Constraints

- Do not include private IP addresses, usernames, credentials, secrets, or
  machine-specific repository paths.
- Do not copy the full operational command sequence into the handover; avoid a
  second source of truth that can drift.
- Do not claim that hardware execution occurred. The implementation and review
  used source inspection, static checks, local tests, artifact verification,
  and documentation builds only.
- Keep the PICO setup section in the user-facing runbook blank as requested;
  the handover may point to the existing VR setup prerequisite but must not
  introduce a competing PICO procedure.
- Distinguish repository-verified facts from lab-owner decisions and external
  prerequisites.

## Verification and Acceptance

The handover is accepted when:

- every referenced repository path exists;
- the stated branch, base, and final commit match Git;
- no unfinished-marker tokens remain;
- the privacy scan finds no concrete home path or private IPv4 address;
- `git diff --check` passes;
- the existing onboarding tests and forced Sphinx build remain green; and
- a teammate can identify the canonical runbook, required lab inputs, safety
  gates, and first-run checklist without reading the implementation history.

## Publishing

Commit the handover on `docs/newcomer-onboarding`, push that branch to the
configured fork, and open one draft pull request against the branch's actual
upstream base. The PR summary links the handover and user-facing runbook and
reports verification without claiming real-hardware execution.
