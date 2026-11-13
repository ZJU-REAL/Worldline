# 主机资源与多运行器协调

原生 batch 默认使用主机级 `HostResourcePool`，不同进程共享 CPU slot 和 GPU 名称锁。等待支持取消和 timeout；持有者退出或 SIGKILL 后内核释放锁。旧进程内资源池通过 `--resource-scope process` 显式选择。`--resource-dir` 或 `WORLDLINE_STATE_DIR` 指定资源命名空间；同一主机不能给各 runner 配不同目录来规避配额。

GPU 必须使用一致的物理设备标识。CPU 参数是调度容量，建议不超过部署分配；内存/显存和实际 GPU 能力由部署及环境后端验证。Docker 资源限制与 admission 分开，见 [环境供应](provisioning.md)。

## 中央任务 lease

```bash
python -m worldline.runtime.coordinator --database /persistent/coordinator.sqlite --port 8769
worldline rollout TASK_A TASK_B -a openai-compatible -m MODEL --batch-id experiment-01 \
  --coordinator-url http://127.0.0.1:8769 --runs-dir node-a/runs
```

其他 runner 使用同一 batch ID、任务身份及出现次数、模型/Harness 配置和 coordinator，保存到自己的 runs 目录。task 路径可以经部署挂载，但内容版本和配置必须一致。跨主机服务通过 TLS reverse proxy 发布，并用 `WORLDLINE_COORDINATOR_KEY` 授权；不把 key 写入 argv、轨迹或结果。

coordinator 使用持久 SQLite/WAL 保存 task signature、唯一 owner、heartbeat 与结果指针。签名不同拒绝领取；过期变成 orphaned，不能假设环境已经停止而重派。原 owner 的晚到结果可以完成该任务。正在运行的环境在 heartbeat 失败后由运行器回收，未知结果仍须保留和核验。

当前已完成任务的产物由共享盘或部署系统同步。`remote_completed` 表示其他 runner 保存了原结果，不是假造的本地 trial；必须取回其 result/trace/media，再用 `worldline report node-a/runs node-b/runs -o combined.json` 合并。只有原结果和 verifier evidence 能进入数据交付。

单个 batch 的本地持久状态仍只有一个 owner；中央 lease 负责不同 runner 的任务归属，不能代替数据库/存储的备份与权限管理。中央服务本身的主备/负载均衡属于部署，不宣称为本版验收范围。
