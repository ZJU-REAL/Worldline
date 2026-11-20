# 指定模型并运行评测

Worldline 0.7 的入口是 `worldline run`。输入模型、选择 Harness 和任务，运行器完成
服务 readiness、协议探测、环境 rollout、独立 verifier 评分和报告。`python -m worldline`
也可以使用同一入口。旧的 `rollout`、`eval`、`experiment` 保留。

## 最短路径

已有 OpenAI-compatible Chat Completions 服务：

```bash
worldline run --model your-served-model --base-url http://localhost:8000/v1 \
  --batch-id first-rua
```

默认 `--harness codex --suite rua-smoke`：使用 MuJoCo 的 held-out reach 集成任务。
这是一个真实模拟器任务，但不是完整 RUA benchmark。评测失败也会输出报告；模型不需要
function calling。Codex 需 Linux、Docker 和 CLI 0.162+，接入见 [Codex 指南](codex.md)；无需 Codex 时显式使用 `--harness json`。需要认证时通过环境变量提供凭据：

```bash
export WORLDLINE_API_KEY='your-key'
worldline run --model your-served-model --base-url https://your-service/v1
```

GPU 上直接使用 Hugging Face 模型或本地 Transformers 权重：

```bash
uv pip install 'worldline-agent[rua,serving]'
worldline run --model Qwen/Qwen2.5-7B-Instruct --batch-id qwen-rua
worldline run --model /path/to/checkpoint --batch-id checkpoint-rua
```

不提供 `--base-url` 时，Worldline 启动一个 vLLM 服务，整个批次共享权重，完成后回收。
需要模型架构受 vLLM 支持、足够的 GPU/显存和权重访问权限。`serving` extra 单独安装；
核心导入不会加载 vLLM、torch 或 Transformers。Python 3.14 的运行器可以通过
`--server-executable /path/to/serving-env/bin/vllm` 使用独立 Python 3.12/3.13 serving 环境。

已有 llama.cpp `llama-server` 和本地 GGUF 时，可在 CPU 或相应设备上运行：

```bash
worldline run --model /path/to/model.gguf \
  --server-executable /path/to/llama-server --batch-id gguf-rua
```

GGUF 自动选择 `llama-cpp`；其他无 endpoint 的权重引用默认选择 `vllm`。
Worldline 不替用户编译 llama.cpp。后端以一个公开 HTTP 服务为接口，不把权重实现放进环境。

## Harness 和模型分别选择

```bash
worldline harnesses
worldline suites
worldline run --model MODEL --base-url URL --harness tools
worldline run --model VISION_MODEL --base-url URL --vision --suite cua-smoke
worldline run --model MODEL --base-url URL --tasks /path/to/native/task1 /path/to/native/task2
```

| Harness | 模型接口 | 环境执行 | 使用范围 |
| --- | --- | --- | --- |
| `json` | Chat Completions 文本/图像 | 原生公开工具 | chat 模型，不要求 function calling |
| `tools` | Chat Completions + function calling | 原生公开工具 | 支持工具调用的模型 |
| `codex`（默认） | Chat Completions 或 Responses | 独立 Codex 容器 + HTTP MCP | 原生 RUA/CUA 公开工具任务 |
| `harbor:codex` | Responses | Harbor 的原生 Codex agent | Harbor/EFE 任务，容器内执行代码 |
| `harbor:<agent>` | 对应上游 agent 的协议 | Harbor | 其他 Harbor agent |
| 已安装插件名 | 插件声明的协议 | native 或 Harbor | 团队自己的 Harness |

JSON Harness 要求模型逐轮输出 `{"tool":"act","arguments":{"action":...}}`，并回传实际
工具结果。格式错误会在同一上下文中反馈并消耗 turn；它属于模型行为，不能重新开环境
冒充基础设施重试。模型只能 observe/act/reset/finish；评分由运行器持有的 verifier 完成。

视觉任务使用 `--vision` 明确声明模型支持图像。预检会发送一张 RGB 图片检查传输接受能力；
这不会验证模型的视觉智能，也不会把图片路径字符串当成图像。

Codex 示例：

```bash
uv pip install 'worldline-agent[harbor]'
worldline run --model served-alias --base-url https://responses-service.example/v1 \
  --harness harbor:codex --tasks /path/to/harbor-task --batch-id codex-rua
```

`harbor:codex` 自动选择 Responses 协议。服务必须支持该协议，并能从 Harbor 容器访问；主机
`localhost` 不能充当容器的模型地址。Harbor 0.23 的 Codex 模型解析需要不含 `/` 的 served
alias；Hugging Face 源名称可另用 `--served-model-name ALIAS` 指定服务别名。
运行器通过 Harbor 的 `config_source` 传入 `openai_base_url`，凭据只通过子进程环境传递。
原生 reach/form 任务使用 `codex/json/tools`；完整代码执行保持在 Harbor 隔离边界内。
正式 EFE 运行继续遵守上游 verifier、网络、预算和 task DIGEST 协议。

## 可复现配置和资源

```bash
worldline run --model-config examples/models/open-model.toml \
  --batch-id model-v1 --max-workers 4 --cpu-budget 8
worldline run --model-config examples/models/open-model.toml \
  --batch-id model-v1 --max-workers 4 --cpu-budget 8 --resume
```

模型 TOML 只包含 `[model]`；CLI 的显式模型参数覆盖对应配置项。模型 revision、采样 seed、
温度、上下文图像/字节预算、输出 token 和请求限额进入记录。建议使用不可变的 HF commit
或 checkpoint revision；endpoint 的实际权重版本由服务提供者声明。GGUF 记录文件 SHA256，
本地目录记录权重/配置清单；目录清单不是权重内容哈希，不能冒充不可变 revision。

`--gpu-devices` 是**环境 worker**的 GPU admission。模型 serving 的 GPU/并行度通过其独立
环境或 `--server-args` 配置，例如 vLLM 的 `--tensor-parallel-size`、`--gpu-memory-utilization`。
请为模型和模拟器分配足够资源；环境 lease 不会替 serving 进程预留显存。

```bash
worldline run --model MODEL --model-revision HF_COMMIT \
  --server-args '["--tensor-parallel-size","2","--gpu-memory-utilization","0.75"]'
```

`tools` + vLLM 还需要适配该模型的 tool parser/auto-tool-choice 设置；如果服务拒收 `tools`，
预检会提示使用 JSON Harness。llama.cpp 默认请求单 slot 和 16384 context，但不同版本可能
重设 slot 容量，必须以服务的实际容量为准。完整提示词和历史仍需容纳于模型上下文；字节
预算不等于精确 tokenizer token 数。token 总预算按 endpoint 报告用量在后续调用前检查，
首次输入和已提交请求可能超过这个预算，它不是服务端硬 token 配额。

## 输出、恢复和失败

```text
runs/evaluations/<batch-id>/
  plan.json                   # 稳定配置、任务版本/seed/split、预算、恢复签名
  preflight.json              # 协议探测；不计入 scored trials
  service-<id>/server.log      # owned serving 日志，文件权限 0600
  report.json                 # 成功率、infra_error、reward、用量和耗时
  result.json                 # 状态与产物链接
runs/batches/<batch-id>/       # 持久化队列与各 attempt 索引
runs/<trial-id>/               # 原始消息、公开动作、独立 verdict 和媒体
```

一个 owned 模型服务由所有 trial worker 共享，随机端口不会进入比较键或恢复签名。
完成的 `--resume` 先校验结果/报告 hash，直接返回，不启动模型、不新增推理。
中断批次使用原有 durable queue 规则恢复。API 请求的暂时故障在原上下文重试；永久错误
阻止后续调用；环境故障只按任务可恢复能力允许基础设施重试。

退出码 0 表示评测完成（可以全部模型失败）或 dry-run 计划成功；1 表示评测不完整；
配置/服务预检失败返回 2。`--dry-run` 不下载/启动模型、不启动模拟器、不发送模型请求。
修改计划需新 batch ID；dry-run 建议使用独立 ID。held-out smoke 轨迹继续被训练导出排除。

## 接入其他服务和 Harness

自定义 serving 使用命令数组，完全不经过 shell：

```bash
worldline run --model MODEL --model-backend command \
  --server-command '["/path/to/server","--model","{model}","--host","{host}","--port","{port}","--alias","{name}"]'
```

支持 `{model}`、`{name}`、`{host}`、`{port}`，服务需提供 `/v1/models` 和选择的推理协议。
owned 服务须以前台进程运行，不能 daemonize 或逃离所启动的进程组。独立 guard 在控制器
正常退出、启动失败或 stdin 因控制器退出而关闭时回收其进程组。已有 endpoint 不归
Worldline 所有，评测结束时不会终止。

新 Harness 注册 `worldline.harnesses` entry point：

```toml
[project.entry-points."worldline.harnesses"]
my-harness = "my_package.plugin:plugin"
```

```python
from worldline.harnesses import HarnessPlugin

def plugin():
    return HarnessPlugin("my-harness", "my-harness/1.0", "native", ("chat",), create_agent)

def create_agent(*, model, options, **runtime):
    # model 是 ResolvedModel；runtime 是公开 spec、initial_observation、deadline、
    # messages、usage、emit、max_turns。返回满足 Agent.run(task, sandbox) 的对象。
    return MyAgent(model=model, options=options, **runtime)
```

注册表拒绝同名覆盖和不兼容契约。插件是部署者选择的可信代码；公共 Agent 契约保持
与 Harbor、机器人动作和浏览器实现无关。Harness 自己报告配置/版本并追加 canonical
messages，评测运行器仍不接受 Harness 的自报 reward。
