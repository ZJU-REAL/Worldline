# External projects

Worldline owns the framework in `worldline/`. External benchmark code is not part of its package.

| Project | Integration | Pinned revision |
| --- | --- | --- |
| [Embodied First Exam](https://github.com/mll-lab-nu/embodied-first-exam) | Optional Git submodule, `embodied-first-exam/`; explicit Worldline adapter | `e695c327deacb346fb468564a5cf40b72569cd34` |

Initialize only when you need EFE tasks:

```bash
git submodule update --init third_party/embodied-first-exam
```

The upstream's RoboTwin submodule belongs **inside EFE**. Initialize it only if required by your
selected benchmark; it retains its upstream pin and dependencies:

```bash
git -C third_party/embodied-first-exam submodule update --init --recursive third_party/RoboTwin
```

You may instead set `WORLDLINE_EFE_ROOT` to a separate checkout. No adapter automatically clones,
installs or patches upstream projects. Preserve their licenses, internal layouts and protocol rules.
See [integration and update guidance](../docs/worldline/upstreams.md).
