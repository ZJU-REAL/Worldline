# Contributing to Worldline

Worldline platform code lives in `worldline/`, regression tests in `tests/`, small runnable tasks in
`examples/`, and platform documentation in `docs/worldline/`. External projects live in `third_party/`
or in user-selected checkouts and are connected by adapters.

Read [the architecture](docs/worldline/architecture.md) before changing contracts. Environment
plugins own action semantics and trusted verification. Runtime code owns lifecycle and scheduling.
Dataset transformations consume versioned traces; training integrations consume data without being
imported by a task or simulator. Preserve lazy optional dependencies and record meaningful versions.

Use Python 3.12 or 3.13 and install only the extras your change needs:

```bash
uv venv --python 3.12
uv pip install -e '.[rua,cua,handoff,dev]'
source .venv/bin/activate
python -m playwright install chromium
python3 -m unittest discover -s tests -v
ruff check worldline tests
ruff format --check worldline tests
```

For runtime/data changes, test failure recovery, trusted verdicts, multimodal fidelity, and split
isolation where relevant. Describe verified behavior and remaining limitations; fixture scores are
not benchmark or model-training results. Do not commit runs, datasets, model weights or credentials.

EFE is an optional, pinned submodule. Its own contribution and protocol rules apply only inside
that checkout. See [upstream integration](docs/worldline/upstreams.md) for initialization, external
checkout selection, attribution and updating the pin. Worldline does not modify upstream task files
or install EFE's global Harbor import hook.
