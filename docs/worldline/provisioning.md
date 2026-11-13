# 外部环境与桌面 CUA 供应

供应层只管理环境生命周期，使用者指定 simulator/controller 的独立安装和经过核对的资产。它不复制 RUA、EFE 或 OSWorld，也不让环境导入训练框架。

## Provider 协议

`[environment.provider].command` 是参数数组，支持 `{request_file}`、`{ready_file}`、`{session_root}`。不经过 shell。request 包含 `worldline.session/0.1`、task_id、task_version、seed、source_revision 和随机 session_id；可选的 task_config_sha256/image_digest/controller_revision 也必须核对。

provider 必须在真实就绪后原子写 ready descriptor，逐项回传以上身份；RUA 另外返回 socket_path/image_root，桌面返回 endpoint。进程持续运行，stdin 的 `{"method":"close"}` 或 EOF 表示结束。provider 和它的本地后代属于 worker 的进程生命周期，禁止把需要回收的进程自行 daemonize。远程 VM/云资产的 cleanup 和 lease 由相应 provider/部署系统负责。

Worldline 会保存 request、ready、session receipt 和 provider log。每个端点必须独占，同一 session_id 的消费凭据不能重用。`WORLDLINE_STATE_DIR` 默认 `~/.cache/worldline`，同一主机的所有 runner 应使用相同、持久的目录；执行期间不要删除锁或消费凭据。

## RUA

复制 `examples/integrations/rua-managed` 到自己的任务目录，填入正确指令、source revision、split/group、算力与模拟器 command 文件。该 JSON 文件例如：

```json
["/approved/simulator-python", "/approved/session_server.py", "--socket", "{socket_path}", "--images", "{image_root}", "--seed", "{seed}", "--task", "{task_id}"]
```

Worldline 的 `adapters/rua/provider.py` 在允许的节点启动这条命令，等待真实 `rua-robot/1.0` observe 响应，生成短路径 Unix socket 和独立图片目录。它不恢复已退役的 RUA runner，也不替使用者决定运行节点。源码、资产和源任务的合法训练 split 必须由拥有者固定。

托管入口 `rua.managed` 允许已分类的基础设施错误使用平台的一次新会话 retry。预先由外部启动的 `rua.socket` 不可自动重新创建；需要 session_manifest，或显式兼容选项。descriptor 的 task_version 可从 `worldline inspect TASK` 获取。每次 borrowed rollout 都要提供新的、实际初始化的环境及身份。

socket adapter 校验响应 ID、向量维度/数值边界、相机路径/hash 和整个观察的图像字节预算。`include_state=false` 可去掉公开 state；默认传递时保守声明 privileged_state。是否属于 image-only 仍要检查源指令和其余公开数据，不能由这个开关替代协议核验。

## 桌面与 OSWorld

复制 `examples/integrations/desktop` 并填入 OSWorld 的独立 Python 环境、reviewed task config 的 SHA256、源码 revision 和 snapshot。controller 使用 OSWorld 的 reset(task_config, seed)、step 和 evaluate；键盘/鼠标代码只由可信 controller 从结构化参数生成，不接收任意 Python/JavaScript。

```bash
worldline inspect path/to/desktop-task
worldline check-plugin path/to/desktop-task --output desktop-contract.json
worldline rollout path/to/desktop-task -a openai-compatible -m YOUR_MODEL --base-url YOUR_ENDPOINT
```

HTTP 观察包含 PNG、真实宽高和 session ID；adapter 校验尺寸和 viewport 坐标。最终 success 来自 controller 的原生 evaluate，另记录 native_score。此入口的代码/线协议已有自动检查，OSWorld VM 的安装、快照与任务评分仍需实际验收。

## Docker worker

可用 `examples/deployment/native.Dockerfile` 构建自己的镜像，发布后将 digest 固定到任务 `[metadata.isolation]`：

```toml
[metadata.isolation]
backend = "docker"
image = "registry/worldline-native@sha256:YOUR_64_HEX_DIGEST"
memory_mb = 8192
pids_limit = 256
network = "none"
mounts = ["/approved/read-only-assets"]
```

默认 process 后端是可信插件的生命周期边界。Docker 后端提供环境进程的硬资源限制和 cleanup guard；不把主机模型 API 凭据传给环境。GPU 使用 admission 选中的设备。需要访问宿主机 controller 时，部署者明确选择相应 network。非可信 coding agent 的正式 benchmark 继续使用 Harbor 的隔离、verifier 和 egress 协议。
