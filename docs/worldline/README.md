# Worldline 平台文档

Worldline 面向需要有状态环境的 Agent：任务和环境只接入一次，评测、rollout、数据蒸馏和模型训练共享版本与证据。RUA 是首个 Domain；通用契约不规定机器人动作、浏览器事件或任何训练算法。

- [架构与模块分工](architecture.md)：依赖方向、插件接入与协作边界。
- [RUA 使用与集成](rua.md)：真实 MuJoCo 示例、模型接入、RUA/EFE 任务包与训练闭环。
- [RUA 论文调研](research/rua-related-work.md)：截至 2026-11-13 的一手资料核对及研究方向。
- [基础设施复审](infra-review.md)：缺陷、修复、效率证据与剩余限制。
- [训练数据交付](training-handoff.md)：LLaMA-Factory 与 verl 的数据包和接入边界。
- [CUA / BrowserGym](cua.md)：真实浏览器环境与 benchmark 接入。

EFE 原有任务、镜像、服务和正式评测协议仍在其原目录中；平台的原生任务使用 `worldline.toml`，已有 Harbor 任务继续使用 `task.toml`。
