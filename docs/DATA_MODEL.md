# Worldline Data Model

Worldline 将一次实验拆成 Run、Trace 和 Result。三者使用版本化 JSON 格式，不要求下游训练代码理解 Harbor、RUA 或 CUA 的内部目录。

**Run** 保存任务、Agent、模型、backend、执行命令和生命周期状态。`run.json` 在状态变化时原子更新，进程中断不会留下半个 JSON 文件。

**Trace** 是只追加的事件流。每个事件包含 run id、类型、时间、payload、可选父事件和 schema 版本。评测、环境交互、结果导入及后续数据转换都写入同一事件模型。

**Result** 是可信验证结果的统一表示。`result.normalized.json` 区分 `success`、`failure` 和 `infra_error`，并分别保存最终奖励、过程指标、资源与模型用量以及错误原因。

一次 Run 只有一个 trial 时同时写出 `result.normalized.json` 和 `results.normalized.json`；存在多个 attempt 时，后者保留全部结果，避免基础设施重试或重复采样被最后一次结果覆盖。

**Artifact** 默认保存来源引用而不是复制文件。引用包含产物类型、相对 trial 的路径和文件大小，可覆盖 Agent trace、环境轨迹、episode journal、视频、图像和 verdict evidence。后续的数据发布阶段再根据策略复制、链接或上传产物。

当前 schema 为 `worldline.trace/0.1` 和 `worldline.result/0.1`。新增字段应保持向后兼容；破坏性修改需要提升 schema 版本。
