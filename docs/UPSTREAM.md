# Upstream

Worldline currently vendors Embodied First Exam as its initial RUA task and evaluation layer.

| Component | Repository | Commit |
|---|---|---|
| Embodied First Exam | `https://github.com/mll-lab-nu/embodied-first-exam.git` | `e695c327deacb346fb468564a5cf40b72569cd34` |
| RoboTwin | `https://github.com/RoboTwin-Platform/RoboTwin.git` | `6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755` |
| XPolicyLab | RoboTwin submodule | `fa431ecd893ee706883e64fe5fe1464ec8cd928d` |

The EFE task layout, protocol documents, CI checks and compatibility commands are kept close to upstream. Worldline
platform code lives separately under `worldline/`. When updating upstream, import the new pinned tree first, review its
protocol changes, then run both the EFE checks and the Worldline tests before updating the commits in this file.

The inherited GitHub Actions definitions are stored under `ci_templates/github-workflows/` in the initial Worldline
repository snapshot. Move them to `.github/workflows/` when the repository credentials used to publish the change have
GitHub's `workflow` scope.

EFE is MIT-licensed. Third-party simulators, assets, models and task data retain their original licenses; see the
upstream task documentation and `LICENSE`/`NOTICE` files before redistributing images or datasets.
