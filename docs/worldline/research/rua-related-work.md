# RUA Harness、评测与训练：Worldline 设计调研

核对日期：2026-11-13（UTC+8）。起点是本地 RUA benchmark related-work、空间/机器人智能 roadmap、RUA 任务工作区和 EFE 参考仓库；本次重新查询 arXiv 的近期 `robot-use`、`robot-use agents` 与机器人 harness 条目，并读取下列论文的一手摘要和版本信息。社区 [Awesome-Robot-Use-Agent](https://github.com/kairunwen/Awesome-Robot-Use-Agent) 用于发现线索，论文与项目原文用于确认论点。这里是针对框架建设的选择性综述，不声称穷尽论文，也不把未复现的作者结果当作 Worldline 成绩。

30 篇重新核对的一手论文及提交日期、当前版本记录在 [sources.json](sources.json)。表中的方法分析以作者摘要为基础，具体复现实验仍需对照全文、代码和任务配置。

## 近期工作的实际变化

RUA 已从“给 VLM 一个 robot API”发展为完整系统设计：可执行代码、感知与控制工具、持久状态、恢复逻辑、资源预算、验证器和训练数据共同决定性能。Worldline 的机会在于把这些不同方法放入同一套环境生命周期和证据链，支持固定模型比较 Harness，也支持固定 Harness 比较训练后的模型。

本地十月资料之外，本次核对推进到了 **11 月 10 日** 的 LEASH，并确认了 RoboACE、SimplEvo、AMP1、HITCH、uni-robot-agent、DiMuRe 等工作。训练方面，十月底的 **CaPtain** 和 **TinyAstral** 尤其贴近“共享环境与 Harness，采集验证轨迹，再训练小模型”的 Worldline 目标。

| 论文 / 当前核对版本 | 方向 | 核心机制 | 对 Worldline 的直接启示 |
| --- | --- | --- | --- |
| [LEASH: Stay Safe in Agentic Robotics](https://arxiv.org/abs/2611.14115v1)，2026-11-10 | 安全评测 / Harness | RoboRisk 闭环任务评测，task/action/commit gates | 可作为可选策略插件与安全事件评测；研究设置应单独记录 |
| [RoboACE](https://arxiv.org/abs/2611.13034v1)，2026-11-09 | 课程 / Harness 共演化 | 固定模型，交替更新课程与执行 Harness，固定挑战快照比较 revision | 任务生成与执行方法分离，保存准入快照和比较配置 |
| [Review the Episodes, Refine the Lessons / SimplEvo](https://arxiv.org/abs/2611.11509v1)，2026-11-08 | 记忆演化 | train episodes 更新有界文字记忆，test 时冻结，无权重更新 | 明确经验更新 split 与记忆版本，复评时冻结学习资产 |
| [REINS](https://arxiv.org/abs/2611.10748v1)，2026-11-07 | 实时规划 / Harness | 动力学 grounding、谓词与 forward rollout 验证、索引规则记忆 | 快执行与慢模型决策需要不同事件和时间预算，不能把实时性归给同步 LLM |
| [On Agentic Manipulation Policies / AMP1](https://arxiv.org/abs/2611.09892v1)，2026-11-06 | 通用 Harness | 同一 reference harness 覆盖单臂、双臂、移动操作；可增加持久记忆 | 平台不固化本体或 skill-only 执行方式，版本化可选记忆 |
| [Flex-RSI](https://arxiv.org/abs/2611.08946v1)，2026-11-06 | 自改进评测 | 固定预算下 executor/improver 修改解与 Harness，用 validation 准入、hidden test 判分 | 系统演化、模型训练、验证集准入和测试判分应是独立阶段 |
| [HITCH](https://arxiv.org/abs/2611.08039v1)，2026-11-05 | 全身 Harness / SFT | 视觉点定位与全身执行 tracker，教师五个基础任务的轨迹训练 VLM | 观测/动作由本体定义，教师/学生共用接口并保留轨迹来源 |
| [VABAR](https://arxiv.org/abs/2611.06881v1)，2026-11-04 | Benchmark | Newton 物理环境、六类操作、任务变体与难度、参考程序 | 作为后续独立 Domain adapter，保留物理可实现性与变化维度 |
| [FEIJOA](https://arxiv.org/abs/2611.04080v1)，2026-11-02 | 移动操作 / grounding | 指令驱动 RGB-D 查询、技能前置条件与前后状态验证 | 工具错误、物理技能错误和 grounding 错误分别记录 |
| [Serving Agentic Robot Fleets](https://arxiv.org/abs/2611.01404v1)，2026-11-01 | Serving / 调度 | 长上下文 KV residency 与 VLA 动作截止共同限制并发容量 | 平台并发上限要与模型 serving、renderer、控制 deadline 联合规划 |
| [Verifying Test-State Identity in Paired Evaluation of VLA Models](https://arxiv.org/abs/2611.00896v1)，2026-10-31 | 评测完整性 | vector reset/autoreset 下相同 seed/episode ID 未必等于同物理初态 | 后续增加可信 scene fingerprint；当前配对元数据只能提供必要条件 |
| [TinyAstral](https://arxiv.org/abs/2610.41216v1)，2026-10-30 | Harness 演化 / 蒸馏 | 成对统计验收 Harness 修订，再固定 Harness 蒸馏小型学生 | 同时版本化系统和模型，比较每个系统 revision 对应的教师/学生 |
| [CaPtain](https://arxiv.org/abs/2610.40799v1)，2026-10-29 | Code-as-Policy / SFT+RL | 教师 synthetic SFT 后 multi-turn RL，physical critic 用隐藏仿真状态分配信用 | 保存训练期监督真值与运行时可见观测的区别，在线 RL 独立于 task/verifier |
| [Democratizing Robot-Use Agents: A Unified Harness with Open Models](https://arxiv.org/abs/2611.01560v2)，2026-11-03 更新 | Harness / 开放模型 | uni-robot-agent 支持开放模型；分析通用能力与 RUA 执行的关系 | 模型适配与环境解耦，留下训练及复评接口 |
| [DiMuRe](https://arxiv.org/abs/2611.04226v1)，2026-11-03 | 恢复 / 评测 | 基于阶段证据和前置条件选择代码复用、恢复技能和终端重规划；提出 LIBERO-Disturb | 保存扰动时点、局部检查和恢复结果；比较时固定物理步预算 |
| [Multimodal Embodied Agents: A Unified Capability-centric Perspective from Computer-Use to Robot-Use](https://arxiv.org/abs/2610.30108v1)，2026-10-21 | RUA/CUA 统一视角 | PAPAV：Perceive、Anticipate、Plan、Act、Verify；强调物理环境的不可逆性 | 跨 Domain 共用生命周期、轨迹、验收，领域动作保持独立 |
| [Beyond Success Rate: Interventional Evaluation of Memory in Vision-Language-Action Systems](https://arxiv.org/abs/2610.23372v1)，2026-10-16 | 过程诊断 | 用干预评价记忆作用，而不只观察最终成功率 | 未来在相同状态、任务和预算下比较 memory/harness 干预 |
| [Towards Agentic Robots: Model, Data, Environment, and Harness](https://arxiv.org/abs/2610.20442v1)，2026-10-14 | 综述 | 按模型、数据、环境、Harness 四域组织 agentic robotics | 平台目录与训练职责按这类边界划分，避免单体系统 |
| [Embodied Skill Bench](https://arxiv.org/abs/2610.16238v1)，2026-10-12 | 技能复用评测 | 比较 native、skill-only、Agent-Directed Reuse；记录动作、tokens、耗时 | 原生 API 和增强工具链应分实验配置，不能默认技能越多越好 |
| [Embodied Turing Machines / COAP](https://arxiv.org/abs/2610.12369v1)，2026-10-08 | 状态代码 / RSI | 代码显式测量与维护状态，共享程序库在测试时不调用 VLM/VLA | `Agent` 不应绑定逐步模型调用；程序策略也可以进入统一评测 |
| [EmbodiedRSI](https://arxiv.org/abs/2610.10498v1)，2026-10-07 | 持续学习 / Harness 演化 | 假设图、信息价值实验选择、代码与技能共同演化、分层记忆 | 未来记录候选修改、物理验证和跨任务晋升，而非只保存最终脚本 |
| [RobotWorld](https://arxiv.org/abs/2610.10409v1)，2026-10-07 | 跨本体评测 | 84 项仿真任务覆盖操作、移动、平衡、驾驶和飞行；明确预算与可执行判据 | 用统一 session/run 格式接入异构环境，不统一低层动作 |
| [RobotUse](https://arxiv.org/abs/2610.04929v2)，2026-10-06 更新 | Harness / 上下文 | 视觉目标与姿态指定、后端几何/规划/控制、子目标上下文和 playbook | 感知/控制工具与高层 Agent 分离，保存工具版本和状态反馈 |
| [OpenRUA](https://arxiv.org/abs/2610.02459v1)，2026-10-01 | 原生接口 / 最小 Harness | 编程 Agent 在 ROS 2 workspace 中通过原生接口感知和控制 | 保留原生编程路线；平台不强行加入任务专用技能或规划流 |
| [PyRUA-Lean](https://arxiv.org/abs/2610.01939v2)，2026-10-13 更新 | 代码执行 / GRPO | Python cells 组合原语、本地检查与重试、按需观测；用执行奖励训练 Coder | 未来代码执行 Harness 应与逐工具调用并列，成本与 reward 同时保存 |
| [DynaHarness](https://arxiv.org/abs/2609.40306v1)，2026-09-30 | 动态执行 / 演化 | 慢语义规划与快物理执行、执行约定、失败归因和成对回归 | 局部执行反馈与后续 revision 要有独立事件，系统更新必须回归验收 |
| [Recursive Harness Distillation](https://arxiv.org/abs/2609.33378v1)，2026-09-27 | Harness 蒸馏 | 强 Agent 将干预经验压成操作手册，并据弱 Agent 反馈递归修订 | 蒸馏产物可为手册或权重；数据转换和 trainer 应分开 |
| [RoboFoundry](https://arxiv.org/abs/2609.32862v1)，2026-09-26 | System-as-Policy | 演化上下文、文件记忆、分层技能，从重复验证的任务修订提升通用能力 | Harness 配置与权重都要版本化，区分 task patch 与可迁移能力 |
| [PhyAgentOS](https://arxiv.org/abs/2607.16636v1)，2026-07-18 | Runtime / 验收 | Session 为调度单元，文件状态连接认知与执行，语义 verifier 区分结束与完成 | Session、公开工具通道、可信 verifier 与经验存储应成为独立模块 |
| [Guava](https://arxiv.org/abs/2606.18363v3)，2026-09-29 更新 | 工具蒸馏 / 小模型 | 感知-推理-行动闭环、语义动作、多模态观测；仿真验证轨迹训练 4B 模型 | 保留模型实际可见的观测、决策与工具反馈，不把纯动作轨迹冒充 RUA 决策数据 |

对部分一手来源的进一步阅读也很有价值：[Robo-Harness K1](https://arxiv.org/abs/2609.29389) 的三维感知工具与教师工具轨迹、[Code-as-Monitor](https://arxiv.org/abs/2412.04455) 的约束检查、[Recova](https://arxiv.org/abs/2610.01178) 的恢复策略训练。它们在本地资料中已有整理，本次未对所有全文及最新版本逐项复核，不和上表的重新核对条目混记。

## 结果怎么比较

PyRUA-Lean v2 的作者摘要报告：同 planner 与 primitives，在 700 个实例上 SR 63.1%→71.7%；49% 更少调用、65% 更少输入 tokens 的统计限于共同成功实例。这个实验更适合定位 Harness 增益；不能解释为模型权重单独改善。其 45 个训练实例的 GRPO 是另一项训练实验，实例数也不是 rollout 总数。

DiMuRe 的 Local 比较保持相同物理步 allowance；Full 版本同时增加 terminal replanning、额外 allowance 和 early handoff，因此两者不能作为单一模块的同预算增益。OpenRUA 原生 ROS 2、RobotUse 的几何后端以及 Guava 的语义动作也具有不同信息和执行权限，不做跨论文裸 SR 排名。

Worldline 应始终固定并保存：任务版本、初态与种子、split/group、权限、Harness/skill/controller 版本、模型版本、预算、重置次数和 verifier。基础设施错误独立报告；动作接口返回成功、执行程序退出和任务最终成功是三个不同事实。Test-State Identity 的实验使用 emulated controllers，未完成真实 VLA 的共享向量实验，不能据此声称已证明已有榜单失效；它提示后续接入向量环境时应核对 resolved state 与 fingerprint。

## Benchmark 接入顺序

| 载体 | 主要用途 | 接入与数据边界 |
| --- | --- | --- |
| EFE / 本地 RUA Harbor 包 | 复用已有可信任务与多模拟器 | 保留 task.toml、DIGEST、容器/验证器协议，先目录索引和原生 Harbor |
| [LIBERO](https://arxiv.org/abs/2306.03310)、[LIBERO-PRO](https://arxiv.org/abs/2510.03827)、[LIBERO-Plus](https://arxiv.org/abs/2510.13626) | 单臂操作及扰动泛化 | 三者与子集分别报告；训练任务、官方初态及测试变体按来源分组 |
| [RoboCasa365](https://robocasa.ai/) | 家务长流程、移动操作和组合任务 | 保留厨房/任务组合划分与控制器信息 |
| [RoboTwin 2.0](https://arxiv.org/abs/2506.18088) | 双臂、工具和接触协调 | 不把单臂动作 Schema 强加给双臂接口 |
| [EmbodiedBench](https://embodiedbench.github.io/) | 视觉与多步骤工具使用 | 子 benchmark 和动作权限独立配置 |
| [RoboDojo](https://robodojo-benchmark.com/) / RobotWorld / EmbodiedSkillBench | 异构难任务、过程诊断与技能比较 | 官方 harness、subset 与开发轨分开；simulation 不证明真机能力 |

RUA 工作区已经有任务来源、初态和 handoff 资产。数量、准备状态与训练轨迹是不同维度：ready-for-collection 不代表全量模型成功，也不代表 SFT loader 验收。本次 Worldline 用 adapter 接入这些包，不把参考解法、公司请求或未核定许可的资产复制进开源框架。

## 可在 Worldline 上做的 method 论文

1. **Evidence-conditioned Harness。** 固定模型、环境和预算，比较逐动作工具、持久代码、按需观测、状态化检查。记录 tokens/调用、无效重复、首次达标、最终提交和失败恢复；增加组件前后做同状态配对实验。
2. **Recovery-aware Distillation。** 从经过验证的成功/失败分支学习故障定位、恢复与停止决策。扩展 snapshot/fork 后在同一可信状态比较候选修复，按物体族/场景族/原演示分组留出。当前版先保留轨迹、错误和 split_group，尚未实现快照与分支执行。
3. **Harness/Model Co-training。** 固定 Harness 比较 SFT/GRPO 后权重，固定权重比较技能与记忆演化；分别追踪数据、模型和 Harness revision。当前文本 SFT 已有独立入口，在线 RL 需后续环境桥和训练 backend。
4. **Cross-domain Stateful Agents。** 用 RUA 与 CUA 共用的 session、trace、tool feedback 和 verifier 探索反馈利用、状态保持及错误恢复迁移。统一的是实验与生命周期；接触控制、坐标系、DOM 或桌面事件保持各自 Domain 定义。

第一版实现的范围是平台与可运行集成：真实仿真、模型工具接口、trusted verdict、任务适配、数据与训练入口。接入更多正式模拟器、代码执行容器、snapshot/fork、在线 RL、多模态 trainer 和 GPU 调度是后续独立模块，不能用当前 fixture 成功代替这些验收。
