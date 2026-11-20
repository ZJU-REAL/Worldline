# Codex、多模型与原生环境

Worldline 0.7 的默认 Harness 是真实 Codex CLI。模型推理、Codex 交互循环、环境和可信
verifier 分别持有独立接口。以后自研 Harness 可以替换 Codex，复用模型、环境与轨迹层。

## 安装和运行

原生 Codex 需要 Linux、Docker daemon 和 Codex CLI 0.162+；本轮验证版本是 0.162.1。

```bash
npm install -g @openai/codex
uv pip install 'worldline-agent[rua]'
# 已有 vLLM、SGLang、llama.cpp 等模型服务
worldline run --model qwen-served-alias --base-url http://localhost:8000/v1
# 安装 serving extra 后，由 Worldline 加载 HF 或本地 Transformers 权重
worldline run --model Qwen/Qwen2.5-7B-Instruct --harness codex
worldline run --model /path/to/checkpoint --harness codex
# 已有 llama-server 和本地 GGUF
worldline run --model /path/to/model.gguf --server-executable /path/to/llama-server
```

默认仍是 held-out `rua-smoke` 集成任务，完整评测用 `--tasks`。模型名称与 Harness 分开，
不要求使用 GPT；具体权重架构需要 serving 后端支持，并有足够的计算资源。
Qwen2.5-7B-Instruct 是文本模型，可用状态/代码接口；机器人图像任务需要视觉模型及
`--vision`。CUA 示例也可以使用 `--suite cua-smoke --vision`。

## 协议转换和模型能力

Codex 使用 Responses，但模型服务可以只实现 Chat Completions。
`adapters/codex_responses.py` 将模型决策转换成 Codex 消费的 Responses 事件，包括 function
calls、工具结果、图片和 SSE。适配器不执行动作、不计算 reward，Codex 保持实际交互循环。

默认 `--codex-decision-mode json`：实际模型提示词包含公开工具 schema，模型生成 JSON
决策，适配器返回对应 Codex function call。不要求 vLLM 原生 tool parser。
已支持 function calling 的服务可以使用 `--codex-decision-mode tools`。
llama.cpp/GGUF 默认使用 JSON Schema constrained decoding；这不保证模型能完成任务，
环境仍会独立校验动作。

原生 Responses 服务使用 `--model-protocol responses`；当前该路径用 provider 默认
sampling，非默认 temperature/top_p 或 seed 会明确拒绝，不记录为已生效的设置。

Canonical trajectory 保存模型实际看到的适配消息。原始 Codex 事件、Responses 输入和
公开 MCP 反馈分别进入 trace，不把 JSON 适配结果冒充未经转换的 Codex transcript。
finish 后的框架 acknowledgement 不作为教师模型生成文本进入训练消息。

## 隔离、资源与生命周期

运行器从本机 CLI 安装定位官方 Linux static binary，按 SHA256 构建并缓存独立 scratch
image，无需 EFE 或 OS base image。image ID、CLI 版本、binary SHA 和 adapter 配置进入 run。
CLI/image 检查在模型和环境启动之前完成；模型服务仍由整个批次共享。

Codex 容器使用只读 root、独立 tmpfs、CPU/内存/进程限制。它不挂载运行器源码、任务目录、
评分文件或模型 API 凭据。Shell、浏览器、多 Agent 等能力关闭，模型工具列表只声明
observe/act/reset/finish。对这四个有预算约束的工具预先配置 approval，评测不等待交互确认。
真正模型 key 留在运行器；容器只得到本 trial 的临时 bridge token。

`harnesses/codex_gateway.py` 提供认证的 HTTP MCP。公开通道没有 verify、checkpoint、fork。
模型错误不会冒充环境成功；Harness 启动/运行异常归入基础设施错误。正常结束、deadline、
lease 取消和运行器退出都回收容器；关闭 gateway 会取消并等待在途模型请求。
`--cpu-budget` 和 `--gpu-devices` 仍管理环境 admission；Codex 容器另外限为 1 CPU/1 GiB，
模型 serving 的资源由 serving 环境配置。

默认探测运行器出站地址，Docker bridge 必须能访问它。特殊部署使用
`--codex-bridge-host <容器可访问的运行器 IP>`；远程 Docker daemon 需正确的网络路由。

## 正式 EFE 与其他 Harness

原生 Codex 用于公开工具实验。需要完整 shell/code execution 的上游任务使用
`--harness harbor:codex`，仍遵守 Harbor/EFE 的 verifier、网络、资源和 DIGEST 协议。
该入口要求容器可达的 Responses 服务，和原生 MCP 入口分别报告。

`--harness json`、`--harness tools` 保留；团队的新 Harness 注册 `worldline.harnesses`
entry point，接收同一个 `ResolvedModel` 和公开 runtime 参数，无需修改任务或训练器。

## 验证边界

```bash
WORLDLINE_TEST_CODEX=1 python -m unittest discover -s tests -p test_codex_harness.py -v
```

该检查实际启动 CLI/容器和 MuJoCo，分别通过两个协议测试模型别名完成多轮动作和可信
评分，并验证完成恢复不增加推理。它验证接入链路，不代表两种真实模型的能力成绩。
真实 CPU GGUF 单独验证生成、工具交互、预算与评分；Qwen2.5-7B 的 GPU 性能及正式
benchmark 仍需在相应资源执行。
