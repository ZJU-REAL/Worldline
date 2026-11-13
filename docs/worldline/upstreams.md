# 上游接入与版本

EFE 是第三方 benchmark 项目。原来 Worldline 把它的完整快照放在根目录，造成平台源码、benchmark、构建脚本和贡献规则混杂。0.3 改为可选 Git submodule `third_party/embodied-first-exam/`；Worldline 主包和 EFE 各自拥有目录、许可证与版本。

| 组件 | 来源 | 固定提交 |
| --- | --- | --- |
| EFE | https://github.com/mll-lab-nu/embodied-first-exam.git | `e695c327deacb346fb468564a5cf40b72569cd34` |
| EFE 内部 RoboTwin | https://github.com/RoboTwin-Platform/RoboTwin.git | `6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755` |
| RoboTwin 内部 XPolicyLab | 上游的递归 submodule | `fa431ecd893ee706883e64fe5fe1464ec8cd928d` |

迁移前逐文件比较 Git blob：已有任务、镜像、服务、脚本与协议和该 EFE 提交一致。Worldline 改写过的根 README、pyproject、AGENTS 和 ignore 由 Worldline 管理；EFE checkout 恢复该提交自己的文件。上游工作流在 submodule 内，原根 CI 模板和 EFE issue/PR 模板也随上游隔离。

默认 clone 不初始化 EFE，wheel/sdist 不包含它。现有 Git 历史中的旧快照仍然保留；需要减少首次 clone 下载量可使用 Git 的 `--filter=blob:none`。本轮没有改写仓库历史。

## 选择 checkout

```bash
git submodule update --init third_party/embodied-first-exam
worldline upstream-info
worldline efe run --core --list
```

适配器按显式 `--root`、`WORLDLINE_EFE_ROOT`、`EFE_ROOT`、当前目录及其父目录下的可选 submodule 查找。显式路径无效会报错，不会悄悄切到其他 checkout。安装 wheel 后可在任意位置处理数据；EFE 命令需要用户指定或初始化 checkout。

`worldline efe` 使用子进程启动该 checkout 的原 CLI，设置 `EFE_ROOT`，不修改上游、不向 Worldline wheel 注入 `.pth`、不改变全局 Harbor 导入行为。使用同一运行器环境时安装 `.[harbor]`，它提供上游 CLI 的 Harbor/dotenv 等依赖；也可通过 `worldline efe --python /path/to/efe-env/bin/python run ...` 选用准备好的独立上游 Python 环境。零依赖 core 安装只保证平台和离线工具可用。直接 `worldline eval path/to/task` 继续使用原生 Harbor。

## 更新

1. 在 EFE checkout 选择并审查目标提交，读取该版本自己的协议与贡献要求。
2. 固定 gitlink；同时更新 `worldline/adapters/efe.py` 的 pin 和本文，记录协议变化。
3. 从 EFE checkout 运行其 invariant 检查，从 Worldline 根运行平台测试，再验证所用真实环境。
4. 提交两个项目之间的适配变化和 pin，保持上游 checkout 干净。

外部 checkout 可选其他版本，`upstream-info` 会报告实际提交及是否与默认 pin 一致，使用者负责核对兼容性。EFE 的任务目录、DIGEST 和八条 Appendix H invariant 不因它移动到 submodule 而改变。

EFE 使用 MIT 许可证，原许可证在 checkout 内完整保留。模拟器、资产、模型和任务数据使用各自许可证，按上游说明分发。
