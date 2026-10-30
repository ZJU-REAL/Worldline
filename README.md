# Worldline

Worldline 是一个面向 RUA（Robot-Use Agent）和 CUA（Computer-Use Agent）的开源训、蒸、评一体化框架。开发者只需接入一次任务环境和验证器，就可以在同一套沙盒与任务定义上完成 Agent 评测、并行 rollout、教师轨迹采集、数据蒸馏、模型训练和训练后复评。

项目初期以 RUA 为主要场景，先打通机器人虚拟环境中的完整闭环，再逐步扩展到桌面、浏览器和移动端等 CUA 环境。

项目以 [Embodied First Exam](https://github.com/mll-lab-nu/embodied-first-exam) 的任务、沙盒和可信评测协议为基础，并保留其 Harbor 兼容能力。Worldline 在此之上增加通用运行时、统一轨迹、数据蒸馏和训练闭环；上游任务仍保持原有结构，便于持续同步和复用。

## 核心目标

- 一次接入，同时支持评测、数据生成、蒸馏和训练。
- 隔离 Agent、环境和验证器，保证奖励可信、实验可复现。
- 统一任务生命周期和轨迹格式，不强行统一不同领域的动作空间。
- 兼容 Harbor、Embodied First Exam、Robo Use 等现有任务和环境。
- 对接 verl 等训练后端，不重复实现底层训练算法。

## 核心架构

Worldline 围绕一条完整链路设计：任务定义运行边界，沙盒承载真实交互，轨迹保存过程，验证器给出可信结果，训练管线再把这些结果转化为模型能力。

**Task Package** 定义一项可复现的任务，包括任务指令、运行环境、初始状态、Agent 权限、资源预算、提交产物、验证方式、数据划分和版本。它是评测与训练共用的任务入口。

**Sandbox Runtime** 承载 Agent 与环境之间的有状态交互，统一环境的创建、重置、调用和销毁，并适配 Docker、本地进程、远程实例及不同的 RUA/CUA 环境。后续将在这一层支持状态快照、恢复和分支探索。

**Agent Runtime** 负责启动 Agent、传递任务和环境反馈，并记录模型调用与工具使用。框架支持 Codex、Claude Code、OpenAI-compatible 模型和自定义 Agent，同时尽量保留各自原生的工作方式。

**Verifier** 与 Agent 隔离运行，根据提交产物和环境保存的可信状态进行评分，统一输出最终奖励、过程指标、安全约束和基础设施错误。

**Trace Store** 保存每次运行中的对话、观察、动作、工具调用、代码产物、奖励、成本和版本信息。轨迹可以被检索、筛选、切片、配对和重放，也可以转换为 SFT、偏好学习或 RL 数据。

**Learning Pipeline** 根据验证结果清洗和转换轨迹，再调用 verl 等训练后端完成 SFT、DPO 或在线 RL。训练后的模型回到同一套任务、沙盒和验证器中复评，形成完整闭环。

## 基本工作流

一项任务从沙盒中启动，由 Agent 完成交互并产生轨迹，随后交给独立验证器评分。通过验证的轨迹经过筛选和转换，用于 SFT、偏好学习或在线 RL；训练后的模型再回到未见任务中复评。

## 第一阶段

第一版优先完成：

- 本地 Docker 和进程沙盒；
- Harbor/EFE 任务适配；
- 一个原生 RUA 环境适配；
- Codex、Claude Code 和 OpenAI-compatible Agent；
- 并行 rollout 与统一轨迹记录；
- 独立验证器和错误分类；
- 轨迹到 SFT、偏好和 RL 数据的转换；
- 接通至少一个开源训练后端；
- 使用 held-out 任务完成训练后复评。

当前仓库已完整采用 EFE 的目录结构，并新增独立的 `worldline/` 平台层。现阶段已建立 Task、Sandbox、Agent、Verifier 与 Trace 的基础接口，以及 Harbor/EFE 任务读取适配器；原有 `efe` 和 `harbor` 运行方式继续保留。

可以先检查一个现有任务：

```bash
python -m worldline.cli inspect tasks/<benchmark>/<task>
```

也可以先规划一次评测，检查将要运行的命令并生成标准 Run 记录，而不启动沙盒：

```bash
worldline eval tasks/<benchmark>/<task> -a oracle --dry-run
```

去掉 `--dry-run` 后，Worldline 会调用 Harbor 执行任务，并在 `runs/<run-id>/` 下保存 `run.json`、
`trace.jsonl` 和 `backend.log`。Harbor trial 会保存在该 Run 的 `harbor/` 子目录，执行结束后自动生成
`result.normalized.json`；多次 attempt 还会汇总到 `results.normalized.json`。运行数据默认不提交到 Git。

已有的 Harbor trial 可以直接归一化，不需要重新执行：

```bash
worldline ingest /path/to/jobs/<job>/<trial>
```

该命令会生成 `result.normalized.json`，统一记录 success、failure 或 infra_error、奖励、过程指标、
token 使用量以及轨迹和视频的来源引用。大型产物不会被重复复制。

## 后续方向

- 状态快照、恢复和多分支 rollout；
- 反事实轨迹与失败驱动的数据蒸馏；
- RUA 的视觉、控制和程序多层轨迹；
- 分布式调度与云端沙盒；
- 浏览器、桌面和移动端 CUA 适配；
- 仿真到真实机器人的迁移与评测。

## 项目原则

- 环境只接入一次，训、蒸、评共用同一套协议。
- 轨迹是核心数据对象，必须可追踪、可转换、可重放。
- 最终奖励来自独立验证器，而不是 Agent 自报结果。
- 训练集、开发集和隐藏测试集严格分离。
- 先做好 RUA 的完整闭环，再扩展环境数量和任务领域。
