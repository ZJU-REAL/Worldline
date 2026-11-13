# Worldline 平台文档

Worldline 面向需要有状态环境的 Agent：任务和环境只接入一次，评测、rollout、数据蒸馏和模型训练共享版本与证据。RUA 是首个 Domain；通用契约不规定机器人动作、浏览器事件或任何训练算法。

- [架构与模块分工](architecture.md)：依赖方向、插件接入与协作边界。
- [RUA 使用与集成](rua.md)：真实 MuJoCo 示例、模型接入、RUA/EFE 任务包与训练闭环。
- [上游与版本](upstreams.md)：EFE 的可选 submodule、外部 checkout 与更新边界。
- [0.4 工程迭代](iteration-04.md)：恢复、视觉数据、训练会话、状态回溯与验收。
- [Live training](live-training.md)：外部训练循环、新动作评分、deadline 与 checkpoint。
- [基础设施复审](infra-review.md)：缺陷、修复、效率证据与剩余限制。
- [训练数据交付](training-handoff.md)：LLaMA-Factory 与 verl 的数据包和接入边界。
- [CUA / BrowserGym](cua.md)：真实浏览器环境与 benchmark 接入。

EFE 原有任务、镜像、服务和正式协议位于可选 `third_party/embodied-first-exam/` checkout 内；平台的原生任务使用 `worldline.toml`，已有 Harbor 任务继续使用 `task.toml`。
