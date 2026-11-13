# 外部训练循环与有状态环境

`NativeTrainingSession` 让训练器逐轮驱动真实环境，再从独立 verifier 获取新动作的结果。环境不导入训练器，模型调用、token/logprob 和权重更新由外部框架负责。SFT 数据交付仍可独立使用；这个接口提供在线交互基础，不等于已经完成某个版本的 verl PPO/GRPO 训练。

```python
from pathlib import Path
from worldline.learning.rollout import NativeTrainingSession
from worldline.runtime.resources import ResourcePool

resources = ResourcePool(cpus=4, gpu_devices=())
with NativeTrainingSession(
    Path("examples/rua/reach-train"),
    runs_dir=Path("runs/training"),
    resources=resources,
    model="your-model",
    agent_config={"method_version": "your-method/v1"},
) as session:
    while not session.done:
        # Implement this callback using your trainer's sampling/inference API.
        message = policy_generate(
            messages=session.model_context(max_images=8),
            tools=session.environment.spec["tools"],
            deadline=session.environment.deadline,
        )
        session.advance(message)  # OpenAI-style assistant message / tool calls
    outcome = session.training_outcome()

if outcome.valid:
    consume_reward(outcome.reward, outcome.run_id)
else:
    discard_unscorable_rollout(outcome.error)
```

`policy_generate`、`consume_reward` 和 `discard_unscorable_rollout` 是训练侧回调，不是框架内置训练算法。失败的 Agent rollout 是有效的 0；基础设施错误是 `reward=None, valid=False`，必须在计算 advantage 或样本筛选前丢弃，不能填成失败分数。不能把旧轨迹的 reward 用在新动作上。

任务必须明确 `split="train"`。每个 session 创建新 worker，使用同一套原生任务、公开工具、预算和 verifier。会话方法串行化，跨会话可并行；共享资源池管理进程内 CPU admission 和 GPU lease，跨节点调度仍由外部训练系统负责。GPU session 需要显式提供设备池。

`finish()` 幂等并保存结果、消息、媒体和 verifier evidence。`abort()` 标记不可计数结果。使用 context manager 可保证异常路径清理；deadline watchdog 会在控制器空闲/卡住时对最终状态判分并释放环境资源，超时返回的模型动作不执行。模型推理服务本身的取消和 GPU 配额由 serving/trainer 后端负责。

## Checkpoint / restore

MuJoCo reach fixture 实现了版本化状态 checkpoint：积分状态、控制输入、warm start、展示缓存和 episode 计数。CPU 和离屏渲染的观测一致性、相同行动后的状态一致性都有回归测试。

```toml
[evaluation]
restore_budget = 2
```

在可信 controller 中使用 `session.checkpoint()` 和 `session.restore(checkpoint)`。默认恢复预算为 0；只允许该 session 签发且未修改的 checkpoint，校验 Task/环境版本。动作数、累计仿真步数和墙钟不会回退。恢复在轨迹中留下事件和明确的观察提示；模型公开工具不包含这些管理接口。

restore 是单 worker 的受控回溯；0.5 的独立 fork 另建 worker，并要求环境实现 fork_restore。其他环境必须实现自己的 `CheckpointEnvironment`，不能把网页 reload、reset 或动作 replay 当完整状态恢复。BrowserGym 和现有 RUA socket 没有声明快照能力。

## 与 verl 对接

离线 SFT 使用 [训练交付指南](training-handoff.md) 的 Parquet 和薄数据 adapter。在线训练可在你所用版本的 AgentLoop/sampler 内调用上述 session 接口，保留模型侧 response mask/logprob，并在 rollout 完成后请求最终 reward。

需按实际版本接入生命周期。本轮核对的 verl `ToolAgentLoop` 对 `BaseTool` 的 create/execute/release 可发生在单次工具调用内；直接在每次 create 启动新环境、每次 release 关闭环境会丢失多轮状态。因此本版不提供这种错误的“一次工具即一个 episode” shim，也不声称已经验证完整 GPU PPO/GRPO 作业。

## 0.5 episode service 与独立 fork

`EpisodeRegistry`、`learning.service` 和可选 `WorldlineEnvironmentTool` 补齐训练侧的生命周期入口：可信 sampler 先 begin，per-tool create 绑定稳定 episode_id，release 只清理短 handle，最终奖励由可信 finish 取得。重复 request_id 读取持久响应；未知 intent 不再执行。响应缓存保留媒体引用并有界，完成会话从活动内存移出。

`session.fork(checkpoint)` 创建独立 worker，继承原墙钟、动作/turn/reset、调用 ID 和累计仿真计数。任务显式声明 `fork_budget`，quota 在整棵树共享；原 source checkpoint 不被修改。每条子轨迹记录 parent run/checkpoint 与恢复后的真实观察，split/group 跟随原任务。该能力当前由 MuJoCo fixture 实现；其他环境需自行实现 fork_restore。

执行和最终 GPU/trainer 验收见 [验收入口](acceptance.md)。这里的 BaseTool 适配器记录结构化 action view，外部框架保留完整 policy token/logprob/mask；需要完整教师消息时直接通过 sampler 的 advance 传原 assistant message。
