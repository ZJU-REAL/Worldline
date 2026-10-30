# RUA 评测与蒸馏

Worldline 目前把 RUA 的评测、结果归一化和训练数据制作串成同一条链路。环境仍使用 EFE/Harbor 的
任务目录与隔离沙盒，Worldline 负责批量运行、记录、比较和蒸馏；训练器作为下游消费这些数据，避免
让 simulator 或 task 依赖某个训练框架。

## 安装与检查

```bash
uv tool install -e .
worldline doctor
```

执行 RUA 任务需要 Harbor、Docker Compose v2（或兼容的 Podman 环境），GPU 任务还需要 NVIDIA
容器运行时。没有容器运行时的节点可以运行 `ingest`、`report`、`export`、`distill` 等离线流程。

## 评测

先用 dry-run 检查任务解析和 Harbor 命令：

```bash
worldline eval tasks/<benchmark>/<task> -a oracle --dry-run
```

实际运行单任务或任务集：

```bash
worldline eval tasks/<benchmark>/<task> -a codex -m <model>
worldline eval-many tasks/<benchmark>/<task-a> tasks/<benchmark>/<task-b> \
  -a codex -m <model> --max-workers 2
```

每个 Run 独立保存生命周期、事件、后端日志和归一化结果。success、failure 与 infra_error 分开记录，
基础设施错误不会混入 Agent 成功率。历史 Harbor trial 可用 `worldline ingest <trial-dir>` 纳入同一格式。

## 蒸馏

```bash
worldline distill runs/ --output-dir datasets/rua-v1
```

该命令一次生成评测报告和三种训练视图：成功轨迹组成 SFT 数据；同任务的成功与失败 rollout 组成
preference pair；全部有效 rollout 连同 reward、metrics 和 provenance 组成 RL 数据。每种数据按
`task_id` 做确定性切分，同一任务的 rollout 不会泄漏到不同集合。

训练完成后，使用相同 held-out task、seed、sandbox 和 verifier 重新执行评测，并用
`worldline report` 比较固定 `comparison_key` 下的结果。训练后端接入是下一层工作，当前数据包可直接由
verl 或自定义 SFT/DPO/RL 管线读取。
