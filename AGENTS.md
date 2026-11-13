# AGENTS.md — Worldline

Worldline is an independent framework for evaluating, distilling and training stateful agents.

- Cross-domain platform code belongs in `worldline/`; tests belong in `tests/`.
- Keep public contracts independent of Harbor and domain-specific action/observation schemas.
- Connect external systems through `worldline/adapters/` and environment entry points.
- Keep environment/evaluation, trace transformation and model training separate. Tasks and sandboxes never depend on trainers.
- Optional backends must remain lazy: importing core Worldline must not import simulators, browsers, Harbor or trainers.
- Read `README.md`, `docs/worldline/architecture.md` and `CONTRIBUTING.md` before changing the platform.
- Run `python3 -m unittest discover -s tests -v` with Python 3.12+, then `ruff check worldline tests` and `ruff format --check worldline tests`.
- Record task/environment/harness versions, seeds, effective budgets, data splits and trusted verdicts. Never retry an agent failure as infrastructure or train on held-out traces.
- Use the repository's configured human commit identity. Do not add AI/bot authors or AI `Co-authored-by` trailers.

EFE is a pinned external Git submodule at `third_party/embodied-first-exam/`, not Worldline source.
Preserve its internal layout and task DIGESTs. For work inside that checkout, read its own
`AGENTS.md`, `README.md`, `CONTRIBUTING.md`, `docs/protocol/README.md` and `docs/protocol/running.md`.
The eight invariants in EFE protocol Appendix H bind all EFE tasks and official runs.
Task-only upstream contribution rules apply inside EFE, not to Worldline platform changes.

Do not edit a submodule to compensate for an adapter defect. Prefer an explicit adapter; a deliberate
upstream update must use a reviewed commit and update the pin and provenance documentation together.
