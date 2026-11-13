# Worldline 0.5 工程补齐

本版补环境供应、主机与中央资源协调、模型请求边界、训练格式保真和可执行验收入口。真实模型、GPU、Docker/VM 和正式 benchmark 的最终验收由部署者执行；自动回归与小环境验证覆盖代码的生命周期和数据约束。

| 能力 | 实现 | 自动验证范围 |
| --- | --- | --- |
| 外部环境供应 | `runtime/provisioning.py` 的命令 provider、随机 session ID、任务/种子/版本 descriptor 校验、端点独占和永久消费凭据 | provider 启停、身份错误、重复领取与单次使用 |
| RUA 托管 | `rua.managed` + `adapters/rua/provider.py` 启动拥有者指定的模拟器命令；保留不自动 retry 的 borrowed socket | 原线协议、响应 ID、动作边界、图像 hash 和字节限制 |
| 主机资源 | `HostResourcePool` 默认协调不同进程的 CPU slot 和 GPU 名称；支持取消、超时、watchdog 跨线程释放 | 两个独立进程争抢、SIGKILL 后释放、取消等待 |
| 多运行器领取 | 中央 SQLite/WAL HTTP coordinator、签名、owner heartbeat、过期 orphaned、原 owner 的晚到结果 | 两个 HTTP client 的唯一领取及过期后拒绝重派 |
| Worker 生命周期 | 创建 worker 的 launcher 线程持续到 worker 退出，避免 Linux PDEATHSIG 随短 HTTP 线程结束误杀环境 | 真实 MuJoCo 通过多次 HTTP 请求保持状态；父死亡/后代回收 |
| 容器硬限制 | 可选 digest 固定 Docker worker、CPU/内存/pids 限制、独立 cleanup guard、显式只读资产挂载 | 命令策略；实际容器与 GPU 限制留给部署验收 |
| 模型调用 | 受控 HTTP I/O 子进程覆盖 DNS/缓慢 body；跨进程 endpoint concurrency/RPM；请求上下文、输出 token、响应字节及 usage 预算 | 真 HTTP 的 deadline、响应上限、retry、并发 admission 与上下文完整性 |
| 长视觉上下文 | 先按文件尺寸估算引用图片的编码体积，再裁剪完整 interaction group，最后只加载入选图片；完整轨迹和实际请求 context 均留存 | 图片预算、工具响应配对、导出内存回归 |
| 数据索引与质量 | 去重、split assignment、ID 和 group overlap 使用临时 SQLite；媒体缓存有界；显式质量限制和每任务配额 | split 隔离、重复 ID、原子发布、保真与质量筛选 |
| LF 数据 | 默认 preserve prose + function calls；附原生模板 formatter entry 和 canonical 数据；omit/reject 为显式策略 | formatter 中 assistant header/工具顺序与原模板不被修改；实际 tokenizer/collator由验收脚本运行 |
| 外部训练 | 独立 LF/verl loader-check；EpisodeRegistry/HTTP service 与可选 verl BaseTool；release 只释放 tool handle | 真 MuJoCo 会话与 idempotency；verl 生命周期测试使用 API-compatible stub，不是 PPO 作业 |
| 受控并行 fork | 新 worker 从签发 checkpoint 导入；共享整棵树的 fork quota、截止时间、已用 action/turn/reset 和累计仿真计算；记录 lineage | 独立物理状态、预算与 held-out/group 传播 |
| 桌面 CUA | structured-action adapter 与独立 OSWorld controller provider | HTTP、图片尺寸、动作边界和代码生成编码；实际 OSWorld VM 留给部署验收 |
| 计划与插件检查 | `experiment` 的 dry-run/执行隔离、owner lock、resume、失败阶段日志、数据包 hash 验证；`check-plugin` | 配置、协议和小环境检查；正式成绩由实际运行产生 |

## 迁移

0.5 批次使用新的版本和工厂代码签名；旧批次不能直接跨版本恢复。旧结果仍能导入和报告。原生外部 socket 默认需要 session descriptor，临时兼容旧服务时显式使用 `allow_unverified_session=true`；这不构成正式身份验收。只读 borrowed session 与 `rua.managed` 的恢复能力分别声明。

LLaMA-Factory preserve 包的模板名以 `worldline__` 开头，必须从包内 `adapters/llama_factory_entry.py` 启动。该别名让直接绕过 formatter 的 native CLI 报错，避免静默丢掉文字。完整原始消息另存于 `data/*-worldline.jsonl`。verl 数据仍通过独立薄 loader 交付。

## 可复现边界

token 总量限制依赖 provider 的 usage，只能阻止后续请求并收紧输出上限；它不是服务端精确计费上限。失败/无 usage 请求计入 `unknown_usage_requests`。单次模型请求默认最多 120 秒，可显式配置；lease 取消会终止等候及 I/O。fork 的 cumulative 预算与 executed 实际工作量分开记录，避免将继承前缀重复计入成本。模型 revision 是部署者声明的身份，须在最终验收中核对 serving checkpoint。环境 reset/restore/fork 都不能扩展正式 EFE 协议；正式任务继续由原 Harbor/EFE runner 执行。

中央 coordinator 协调任务归属，部署层汇集产物。过期未知状态保留 orphaned；没有原结果或人工核验，不自动重新分配。主机 GPU 名称必须在各 runner 中一致，不自动将同一设备的不同别名归并。

GitHub 自动 workflow 的可执行模板保存在 `docs/worldline/ci/worldline.yml`。当前发布凭据只有 repo scope、没有 workflow scope；本地必跑检查可以执行，workflow 的注册需具有对应写权限的发布凭据。

详细入口见 [环境供应](provisioning.md)、[多运行器协调](distributed.md) 与 [最终验收](acceptance.md)。
