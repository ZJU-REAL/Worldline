# Worldline 0.3 工程迭代

本轮优先补框架边界、运行恢复和数据交付，不继续扩展论文调研。0.2 已有原生 RUA/CUA 路径，但第三方源码混在根目录、历史轨迹元数据传播和产物发布仍有缺口。

## 完成的修改

| 问题 | 0.3 行为 | 所有权 |
| --- | --- | --- |
| EFE 整份快照占据根目录，wheel 携带上游 CLI/.pth | 可选固定版本 Git submodule；核心 wheel 不含 EFE/全局 hook；显式 checkout 子进程启动 | `adapters/efe.py`、`third_party/` |
| 核心 readiness 被 tasks、uv、Harbor 的存在绑住；Docker 二进制被误当 daemon 可用 | 核心、原生依赖、Harbor daemon/Compose 各自检查；wheel 可离开 checkout 使用 | `doctor.py` |
| Harbor 多 trial 结果缺少 split/seed/group；catalog 的 split 只停留在索引 | 每个 normalized result 带任务元数据；catalog 传播到 run/task snapshot，禁止覆盖已声明 split/seed | `eval.py`、`results.py`、`adapters/rua/catalog.py` |
| 切分后样本仍为 unspecified，误传 test 文件可能漏过训练拒收 | 实际 split 写回样本；evaluation/held_out 固定归 test；训练入口执行拒收 | `datasets.py`、`learning/` |
| ATIF 图片只剩文字引用 | 支持 ATIF 本地 image source，在可信 trial 内定位并保留真实图片；越界/缺图/远程/歧义或不支持模态拒收 | `adapters/atif.py` |
| 记录临时名在同进程线程间冲突；媒体可能暴露半文件；交付中断留下半包 | 唯一临时文件、fsync 后原子发布；媒体完整后独占链接；完整蒸馏/训练包 staging 后发布，失败单独留诊断目录 | `storage.py`、`media.py` |
| 输入与 Parquet 哈希整体 read_bytes，内存随数据文件大小上涨 | 文件哈希流式读取；格式转换拆分公共消息、LF、verl 模块 | `learning/formats/` |
| 批次缺少实时查询和停止领取；完成结果丢失/改变后恢复仍跳过 | SQLite 实时状态和 owner 检查、停止领取并 drain、恢复校验结果身份与已保存 hash；不提供免费重跑 | `runtime/queue.py`、`runtime/batch.py` |
| 设备列表重复可发出两个相同 GPU 的独占 lease | 拒绝重复 device；资源 admission 独立模块 | `runtime/resources.py` |
| Harbor 进程启动/结果导入异常使 Run 长期显示 running | 异常记录终态和时间，关闭 trace，保留日志和异常 | `eval.py` |

新规则仍保留上游语义：Agent 失败不转为基础设施重试；最多一次新环境 infra retry；不对不支持重建的外部 socket 环境自动重建。控制命令针对 native 批次，Harbor 正式运行使用 EFE runner。

## 验收

- 平台测试：81 项；覆盖真实 MuJoCo、真实 BrowserGym、模型暂时/永久错误、deadline、worker 回收、批次恢复、媒体、两种训练格式和本轮回归。
- Ruff 格式/静态检查通过。
- EFE 固定在 `e695c327deacb346fb468564a5cf40b72569cd34`；核对原根目录中的 40,705 个对应上游文件，其 blob 完全一致。上游包含 2,128 个任务包，内部布局没有改写。
- 在独立 EFE checkout 对 RoboTwin stack_blocks_two privileged/standard 参考任务对运行 `ci_checks/run_invariants.sh`：10 个命令全部通过，包含 task DIGEST、隔离、twins、retry-policy、runner-results；不是完整 2,128 任务的实跑验收。
- 实际 CLI 链路：RUA train + CUA train/test 三项成功；相同批次 resume 复用三个 Run ID；蒸馏两条 SFT 训练样本，排除一条显式 test；LF/verl 交付各两条样本，图片和 Parquet 校验通过。
- wheel/sdist 构建及无依赖 wheel 安装验证；核心可在仓库外运行 version/plugins/doctor/inspect，安装不包含 EFE 或 `.pth`。

本地验收产物在忽略目录 `reports/v03-verification/`、`runs/v03-accept/`、`datasets/v03-accept/`、`datasets/v03-lf/` 和 `datasets/v03-verl/`。这些是平台 fixture 的证据，不是模型 benchmark 或训练结果。

## 接下来仍需补的工程能力

1. **正式多模拟器验收**：按任务声明准备镜像、Compose、GPU renderer 和网络限制，在实际模型上覆盖更多 RUA benchmark。本机当前缺少 Harbor/Compose 的完整运行条件，不能以静态检查冒充容器实跑。
2. **在线 RL 桥**：训练侧 AgentLoop、多轮工具消息和独立 verifier reward。当前交付 SFT 数据及 RL 初始任务 prompt，尚无可直接启用的 RUA/CUA 在线 PPO/GRPO 后端。
3. **更长视觉 episode 的内存上限**：语料级流式处理已具备，但单条 episode 和一个 Parquet batch 的图片仍在内存中；应加入逐 turn 流、按字节控制 batch 和媒体 manifest。
4. **完整状态快照/分支**：先声明各环境可恢复状态、时间和随机源，再实现 snapshot/fork，不能把 reset/replay 当完整状态分支。
5. **桌面 CUA**：独立接入 OSWorld/桌面 VM 的会话与评测边界；BrowserGym 的浏览器能力不能代替桌面 VM。
6. **分布式与硬隔离**：当前 SQLite/owner 锁适用于单机；跨节点租约、远程 worker、cgroup/GPU 配额和不可信插件隔离需独立后端。
7. **正式 CI 激活**：平台 workflow 模板已保留；GitHub 发布凭据缺少 workflow scope，本轮没有注册自动 workflow。

跨版本恢复使用保守边界：0.3 把 Worldline 版本与资源配置写进 batch signature，旧版本批次不静默继续。已有结果仍可 report/distill；后续 rollout 使用新 batch ID。非空数据包输出也不覆盖，便于追溯不同迭代。
