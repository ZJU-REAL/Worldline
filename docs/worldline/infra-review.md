# Worldline 0.2 基础设施复审

上一版是可运行的接入骨架，44 个测试主要证明了小任务的功能闭环；它没有证明批次恢复、规模化数据处理、真实训练框架格式和跨领域运行可靠。本轮按“会影响评分正确性、数据质量或批量使用”的 A 级优先项修复，仍将系统定位为单机研发基础设施。

## 发现、修复与证据

| 优先项 | 上一版问题 | 本轮实现 | 验证方式 |
| --- | --- | --- | --- |
| A：完整结果发现 | Harbor 多 trial 仅写聚合 JSON，数据发现器可能全部漏掉 | 每个 trial 有可发现的独立 normalized result | 多 trial 回归测试 |
| A：受控故障恢复 | 暂时 API 错误、鉴权失败、动作失败未分清；批次无持久化状态 | 类型化 retryable 错误、模型退避重试、至多一次新环境 infra retry、鉴权/模型端点失败阻断未开始任务 | 本地 HTTP 503/401 测试、同上下文重试、失败不重试 |
| A：一次评分一次计数 | 重跑/导入同一个 trial 可能重复计数；旧 attempt 无显式归属 | logical trial / attempt / counted / superseded 字段，报告与导出排除被替代 attempt | 重试后只计一个 trial；旧目录和成本保留 |
| A：恢复批次 | 中断后无法区分已完成与待执行工作 | SQLite 队列、进程所有权锁、逐 trial 提交、相同任务版本与 Agent 配置恢复 | 已完成任务不执行；模拟 verdict 已落盘但 queue 未提交的恢复 |
| A：数据规模 | distill 三次读取轨迹；split 全部 materialize；preference 全部驻留内存 | 单次流式读取生成多个视图、两遍流式 split、SQLite preference spool | 每个 trajectory 只读一次；约 13 MiB 对话数据的 split 峰值低于 2 MiB |
| A：视觉产物 | 同一帧在环境事件、工具消息和轨迹中反复 base64 保存 | 内容寻址媒体目录，轨迹仅保存带 hash 的引用；导出时校验与还原 | 重复图片只写一次，篡改图片被拒绝 |
| A：训练交付 | 通用 messages JSONL 不等于训练框架能直接消费的格式 | LLaMA-Factory ShareGPT 包、verl SFT/RL Parquet 和薄数据适配器 | 工具顺序、图片占位符、Parquet roundtrip 与 RL prompt 检查 |
| A：跨领域检验 | CUA 插件仅停留在接口设想 | BrowserGym/Chromium 真环境、BID/坐标动作、截图、原生验证 | 真实提交、空操作失败、CUA 轨迹进入训练包 |
| A：资源与生命周期 | 每项任务预建 future；CPU/GPU 无 admission；launcher 死亡可能留 worker | 固定 worker 数领取队列、CPU admission、独占 GPU lease、Linux parent-death 回收 | 超容量拒绝、worker 超时回收、正常退出 |

评分还有一个容易忽略的边界：已经取得可信成功状态后，最后一次模型请求故障不能抹掉成功；源环境已经 terminal failure 后，模型错误也不能将它变成免费重试。本轮继续向独立 verifier 请求最终结果，并区分“无法继续的 provider 故障”和“已经确定的任务终态”。

## 高效不只看并发数

`worldline profile-trace --events 100 --unique-images 10 --output-dir reports/trace-profile` 提供可复现的 I/O 检查。本机一次有效 PNG fixture 测量为：inline 6,605,090 bytes，内容寻址 538,920 bytes，存储减少约 91.8%；同步次数从 101 降为 4。这个 fixture 有重复帧，不能外推为任意机器人/模型的吞吐增益。图片几乎不重复时，收益主要来自二进制保存和减少同步。

非交互 BrowserGym 通常不需要第二个聊天 UI 浏览器；本地 fixture 用内存聊天通道保留任务所需消息。外部基准默认保持 BrowserGym 原通道，允许显式 `headless_chat=true`，并将通道及实际 Chromium 版本纳入有效环境配置。任务初始化、Agent、验证和整个 run 的耗时独立记录，用来找真实瓶颈。

浏览器在本机默认启动组合中会阻塞于 `new_page()`；CPU headless 配置通过了真实运行与并发检查。
该模式、启动预检查与缓存准备是基础设施配置，不能归因为模型能力增长。批次缓存可选 plugin preflight，
缺依赖/浏览器时快速失败；永久模型鉴权/端点错误阻断尚未开始的任务，避免重复创建环境和发起无效请求。

数据库只适用于单机队列。CPU/GPU admission 是调度限制，不是 cgroup 资源强制；容器、显存、renderer 的硬约束仍由部署层或 EFE/Harbor 处理。`max_workers` 不是 GPU 容量，所有 worker 共享一张卡并不能自动提高吞吐。

Harbor 批次也使用有界 future 数，避免为整套任务池预建 future。持久化 resume、plugin preflight 和资源 lease
目前属于 native rollout 路径；EFE 正式运行仍使用其 runner 的资源、网络和一次 infra retry 规则，不叠加第二套自动重试。

## 与已有工作的对应

- [PyRUA-Lean](https://arxiv.org/abs/2610.01939v2) 提示应该同时测模型调用、视觉成本和实际执行组织；本轮先修复轨迹 I/O 与统计，代码 cell harness 仍是后续独立 Agent 模块。
- [TinyAstral](https://arxiv.org/abs/2610.41216)、[CaPtain](https://arxiv.org/abs/2610.40799) 提示教师/学生应使用明确的 Harness 与物理验证数据。本轮保留环境版本、工具结果和数据归属，不把日志格式转换冒充在线 RL。
- [BrowserGym](https://github.com/ServiceNow/BrowserGym) 与 [AgentLab](https://github.com/ServiceNow/AgentLab) 提供真实浏览器任务/动作/验证边界。接入复用其执行器与判据，外围 session、trace、数据交付复用 Worldline。
- [LLaMA-Factory 数据协议](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md) 要求角色交替、工具 function_call/observation、图片路径和占位符对应。本轮显式转换并拒绝无法对齐的数据；函数调用旁的 prose 不默默假称已监督，会在 manifest 计数。
- [verl 数据准备](https://github.com/volcengine/verl/blob/main/docs/preparation/prepare_data.rst) 与 [MultiTurnSFTDataset](https://github.com/volcengine/verl/blob/main/verl/utils/dataset/multiturn_sft_dataset.py) 支持独立数据接口。本轮 SFT 保留完整工具对话，RL 文件只放初始 prompt 和任务身份；新响应必须重新执行判分。
- [Agent Lightning](https://github.com/microsoft/agent-lightning) 强调 Harness 与训练系统解耦。本轮不将任何 trainer 变成环境依赖，也不引入其代理到 EFE 正式运行。

以上接口来源在本轮重新读取；本轮验证了格式和环境，不声称复现论文模型分数。

## 面向方法论文的研究点

工程接入、格式转换和 I/O 提速提供实验底座；论文贡献应由新的学习/决策机制和真实留出实验支持。比较值得推进的三个可验证假设是：

| 研究点 | 假设与机制 | 必须做的对照 | 本轮底座与缺口 |
| --- | --- | --- | --- |
| 同状态的恢复分支蒸馏 | 同一可信失败状态中的成功/失败修复，比仅成功的长轨迹更能训练诊断、恢复与停止 | 相同 rollout/训练预算的 success-only、自然失败、同状态配对；按物体/布局族留出 | 结果/轨迹/group/attempt 已有；完整状态快照、分支执行和因果配对尚需实现 |
| RUA→CUA 的反馈能力迁移 | 从实际状态变化学习的进度保持、错误定位、终态判断，可以跨不同动作 Schema 迁移 | RUA-only、CUA-only、联合训练、只学 tool format 的对照；固定模型和 Harness | 两个真环境共用工具/结果/训练接口；正式任务池、权限对齐和训练复评尚缺 |
| 预算感知的 Harness/模型联合改进 | 把按需观察、局部执行与记忆作为受验证的策略资产，能改善成功率—成本曲线 | 固定权重比 Harness；固定 Harness 比训练权重；跨任务迁移而非逐题 patch | 全 attempt 成本、分阶段耗时、配置隔离已有；持续代码环境与 Harness revision 搜索尚缺 |

PyRUA、TinyAstral、Guava、RoboFoundry 等已覆盖相邻方向，不能仅靠重命名主张新颖性。要面向 A 类论文，需要明确差异机制、可复现的强基线、跨任务/跨域留出、误差与成本分析；当前两个 fixture 和格式测试不提供这样的证据。

## 仍需继续做的工作

1. 在真实大 RUA/CUA 任务池上测 cold start、GPU/renderer 容量、模型 serving 吞吐及成本，而不是仅比较小型 fixture。
2. 引入分布式 queue/lease、节点心跳和可观察的取消流程；本轮不是多机容错系统。
3. 快照、状态 fingerprint 与反事实 branch rollout；可见观测一致不等于完整物理状态一致。
4. 训练侧的 verl AgentLoop、工具环境桥和 trusted reward callback，及多轮信用分配。Parquet 准备不能替代它们。
5. 在安装的 LLaMA-Factory/verl 与所选模型/processor 上运行实际训练、留出复评；格式测试不等于训练 loader/收敛验收。
   历史 Harbor/ATIF 图像引用还需要专门的附件/gateway hydration，原生媒体路径的验收不能覆盖所有历史格式。
6. 桌面 OSWorld/虚拟机 CUA、持久代码执行容器及更强隔离。当前 CUA 接入是浏览器方向。

完成本轮后，Worldline 的进步在于有了可检验的恢复、数据与 CUA 边界；是否达到生产规模和论文实验要求，还应由这些后续真实实验决定。
