# Worldline Architecture

Worldline 在 EFE 之上增加平台层，而不是直接重写已有任务。`tasks/`、`images/`、`embodied_first_exam/` 和原有脚本继续承担 RUA 任务及可信评测；`worldline/` 提供跨领域的稳定接口。未来的 CUA 环境通过 adapter 接入，不需要复制 EFE 的机器人专用实现。

## 边界

Task 描述任务身份、指令、环境、版本、数据划分和运行约束。Worldline 可以读取 Harbor 任务，但公共接口不依赖 Harbor 的目录结构。

Sandbox 管理一次有状态环境的生命周期。所有 backend 都需要支持启动、观察、动作、重置和关闭；快照、分支、实时执行及特权状态通过 capability 显式声明，不作为最低要求。

Agent 负责完成任务。框架只规定 Agent 与任务、环境之间的边界，不限制其内部是 coding agent、GUI agent、开源模型还是自定义控制器。

Verifier 在 Agent 无法修改的信任边界内计算结果。EFE 的隔离验证、不泄漏答案和可信 verdict 等规则继续作为默认安全基线。

Trace 是评测和训练之间的数据接口。一次运行中的消息、观察、动作、工具调用、产物、奖励和成本都以追加式事件保存，并带有任务、模型、环境和父事件来源。

Learning Pipeline 消费通过验证的 Trace，生成 SFT、偏好学习或 RL 数据，再调用外部训练系统。Worldline 负责数据与环境闭环，不重新实现模型训练引擎。

## 兼容策略

EFE 作为首个内置 RUA task provider，并尽量以接近上游的形式保留。Worldline 自身功能只依赖公共 contract，通过 adapter 与 Harbor/EFE 交互。以后新增 CUA provider 时，应实现相同 contract，并保留各自原生的 observation、action 和 verifier 语义。
