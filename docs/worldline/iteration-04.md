# Worldline 0.4 工程迭代

本轮补评测、视觉数据、训练会话和受控状态恢复的可执行缺口，保留独立上游边界。

| 模块 | 本轮行为 |
| --- | --- |
| 运行状态 | Native/Harbor 共用 `DurableQueue`；Harbor 支持恢复、停止领取和结果完整性检查。已启动但不能确认结束的任务保留 orphaned，等待后端结果，避免叠加上游 retry。 |
| 进程回收 | 主 worker 已退出时仍回收其进程组；launcher 丢失时连忽略 SIGTERM 的后代一起终止。 |
| IPC / 媒体 | 二进制先写内容寻址共享文件，管道传引用，公开观察仍恢复原格式。验证了超过 16 MiB 的编码观察；缺图/错误 hash 拒收。 |
| Harness | 消息历史保留图片引用；模型请求可设置最近 N 张图预算，完整轨迹保留。自动记录 Agent 类、代码 hash、有效配置和模型名；无效 usage 不污染结果。 |
| 视觉蒸馏 | distill 默认 reference 媒体，流式校验/复制和去重；搬动整包后仍可交付真实图片。 |
| 数据准入 | 未声明 split 默认排除；显式审核开关被记录。development 归 validation；冲突 split/group 拒收；偏好配对也隔离 split/group。 |
| 交付内存 | JSONL 在解析前检查单行字节预算；Parquet 按字节和行数限制缓冲，超大记录拒收。SFT/RL 两种行先校验，再发布。 |
| Live training | 新 `NativeTrainingSession` 供外部 sampler 逐轮交互，train-only、新状态 verifier reward、不可评分 mask、资源池和 deadline watchdog。 |
| Checkpoint | MuJoCo fixture 支持 session 内 checkpoint/restore，校验身份及状态；恢复预算默认 0，运行/累计计算预算不回退。不是并行 fork。 |
| 渲染 / readiness | 增加离屏视觉示例、EGL 设备绑定和实际渲染实现记录；nvidia-smi 仅存在而没有可读 GPU 信息不会判为可用。 |
| 开发检查 | 配置人类提交身份规则和 AI attribution 拦截；pre-push 运行平台测试/Ruff，短期缓存仅匹配同一源码及环境。 |

## 验证与性能证据

平台回归覆盖真实 MuJoCo、真实 BrowserGym、API 工具客户端与浏览器全链路、数据泄漏、资源回收、Harbor 恢复、状态恢复及两种训练包。完整结果和本地验收产物见 `reports/platform-check/`、`reports/v04-verification/`。

可复现导出基准：

```bash
worldline profile-data --frames 100 --unique-images 10 --image-size 576 \
  --output-dir reports/data-profile
```

100 帧共 99,732,230 字节、10 张不同的有效 PNG。实际 export 路径测得 inline 的 Python 峰值约 399 MB，reference 约 0.69 MB；JSONL 约 133 MB 对 30.5 KB，reference 连同媒体约 10 MB。该机器上耗时约 1.51 秒对 0.072 秒。它是合成图像的导出 I/O/Python allocation 测试，临时样本会删除，不代表模型或模拟器加速，也不是 task 成绩。

真实 RUA 视觉样例可在该环境使用 EGL 离屏渲染，输出 7 张图片并通过 verifier；CPU 渲染 checkpoint 的图片和后续状态一致性已验证。当前环境没有可见 NVIDIA GPU，nvidia-smi 路径是空文件；GPU 示例与设备绑定不冒充已完成硬件验收。

## 迁移和边界

- `distill` 默认保留媒体引用；整包移动后交付可自动定位根目录，独立 JSONL 也可以显式 `--media-root`。旧 inline 导出仍可选。
- 未声明来源 split 的历史 runs 默认排除，审核后可使用 `--allow-unspecified-split`；开关不是隐藏测试集的豁免。
- 0.4 的 Worldline/环境版本进入恢复配置，旧批次不能静默跨版本继续。已有结果仍可读，新运行使用新 batch ID。
- 本版完成单机与外部 sampler 的交互接口及数据交付；分布式租约、桌面 VM、独立并行 fork、全量 GPU benchmark 和具体训练器的完整 PPO/GRPO 作业仍需相应后端与资源验收。
- GitHub contributor API、主分支作者/提交者和提交消息中均没有 Claude。未改写现有人类作者历史；提交钩子防止后续 AI attribution。GitHub 自动 workflow 仍受发布凭据 scope 限制，当前使用可执行本地检查和已有 CI 模板。
