<!-- GENERATED FILE. DO NOT EDIT DIRECTLY.
Sources: workspace AGENTS.md and docs/agent-overlays/api.md
source-sha256: 4822fbdcaabb8de3195b8ce23b1f4d0760b9e6f4d95ccf190c20c1dbce499114
Regenerate from the workspace root with: make sync-context
-->

# Spine Screening Workspace Instructions

## Mission and scope

This workspace contains a research-only spine screening system. The active products are
`Spine-Screening-API` and `Spine-Screening-Frontend`. Treat
`Spine-Opportunistic-Screening`, `SOS-Colab`, and `From Phu` as read-only references unless the
user explicitly includes research or training work in scope.

The system produces screening-support geometry and measurements. Never describe model output as
a diagnosis or claim that it proves scoliosis, fracture, osteoporosis, or another condition.
Uploaded images must be anonymized; never put patient-identifiable data, radiographs, secrets, or
checkpoint tensors in prompts, task records, logs, fixtures, or documentation.

## Context loading

Use progressive disclosure to conserve context:

1. Read this file and the active repository's generated `AGENTS.md`.
2. Read only the relevant root document in `docs/`.
3. Inspect target files and their closest tests with `rg` or `rg --files`.
4. Expand into adjacent modules only when a discovered dependency requires it.

Do not broadly read notebooks, checkpoints, datasets, generated builds, dependency directories,
coverage output, caches, or whole-repository listings. The root documents are canonical; README
files are entry points, not duplicate architecture specifications.

## Agent operating model

One lead agent owns each task. Use no specialist for small isolated work. Delegate only bounded,
independent work where parallelism or domain expertise materially helps, with at most two
specialists running concurrently:

- Backend/model specialist: FastAPI, Pydantic, inference, checkpoints, landmarks, geometry, and
  readiness.
- Frontend specialist: Next.js, React, UI state, overlays, accessibility, and API consumption.
- Contract/verifier specialist: OpenAPI compatibility, cross-stack behavior, tests, and safety
  language.

Concurrent agents must have mutually exclusive file ownership. Serialize tasks that touch the
same files or public contract. For cross-stack changes, stabilize and export the backend OpenAPI
contract before editing frontend consumers. Public API, landmark order, coordinate transforms,
model loading, clinical wording, and cross-repository changes require independent verification.

Every delegated task must state its goal, acceptance criteria, allowed repositories and files,
required documents, interface impact, validation commands, prohibited actions, and risks. A
specialist handoff contains only changed behavior, changed files, tests run, unresolved risks, and
durable decisions. Do not retain raw reasoning or full command logs.

## Engineering boundaries

- Preserve user changes in dirty worktrees and avoid unrelated rewrites.
- Agents may inspect, edit, and test locally. Do not commit, push, merge, deploy, publish, or
  transmit sensitive data without explicit user authorization.
- Keep frontend and backend as independent Git repositories. The workspace root coordinates them
  but does not absorb their histories.
- FastAPI OpenAPI is the authoritative HTTP contract. Regenerate the frontend schema after route
  or response changes; never hand-edit generated schema files.
- Preserve floating-point coordinates without premature rounding or clamping.
- AP/PA landmark order is TL, TR, BL, BR. LA order is top-left, top-middle, top-right,
  bottom-left, bottom-middle, bottom-right.
- LA Cobb output is four-outer-landmark geometry agreement, not clinical AP/PA Cobb measurement.
- Never add large datasets, checkpoints, generated outputs, environments, or secrets to the root
  coordination repository.

## Validation and handoff

Run the narrowest relevant checks during development, then use root automation before handoff:

- `make check-context` validates generated repository instructions.
- `make check-contract` validates OpenAPI and generated frontend schema drift.
- `make verify-fast` runs normal backend and frontend quality gates.
- `make verify-all` adds production build and container checks; real-checkpoint tests remain
  opt-in through `SPINE_RUN_MODEL_TESTS=1`.

Record only durable architectural decisions in `docs/decisions/`. Final handoffs should lead with
the outcome, name validation performed, identify remaining risk, and avoid pasting generated files
or verbose logs.

---

# Backend Repository Rules

This repository owns FastAPI routes, Pydantic contracts, model runtime lifecycle, inference,
post-processing, and checkpoint validation. When this repository is opened inside the coordinated
workspace, read root `docs/api-contracts.md`, `docs/model-contracts.md`, and
`docs/medical-safety.md` when those areas are relevant.

Keep dependency direction `api -> services -> inference/postprocessing`. HTTP schemas belong in
`app/schemas`; inference value objects must remain independent of FastAPI. AP/PA and LA readiness
must remain isolated while sharing the configured process-wide inference limit.

Never load or serialize checkpoint tensor contents merely for inspection. Use strict safe loading,
metadata validation, and focused smoke tests. Preserve candidate IDs, floating-point coordinate
restoration, landmark order, and projection-specific measurement wording.

Run targeted pytest files during development. Before handoff run Ruff and the normal pytest suite;
run tests marked `model` only when requested or when checkpoint/runtime compatibility changes.
After route or response-schema changes, return to the workspace root and run
`make export-contract` followed by `make check-contract`.
