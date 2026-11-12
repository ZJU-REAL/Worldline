# RUA 接入与运行

本版提供真实 MuJoCo 原生环境、现有 RUA socket 会话适配、RUA/EFE Harbor 任务目录索引、模型工具 Agent、评测/蒸馏数据闭环和独立 SFT 训练入口。MuJoCo 两连杆 reach 是小型集成 fixture，用来验证平台；它不是 LIBERO 或其他正式 benchmark 的替代。

## 安装

```bash
uv venv --python 3.12
uv pip install -e '.[rua,dev]'
source .venv/bin/activate
worldline plugins
```

核心安装不要求 Harbor、MuJoCo 或训练库。按用途选择 `harbor`、`rua`、`training`、`dev` extras。已有 EFE 功能需要 `uv pip install -e '.[harbor]'` 和 Docker/Compose；GPU benchmark 还需相应 GPU/renderer。

## CPU 上验证闭环

```bash
# 真实 MuJoCo 仿真：两个初始场景，每个独立进程
worldline rollout examples/rua/reach-train examples/rua/reach-test \
  --agent reference --max-workers 2

# 空操作应被可信 verifier 判为失败
worldline rollout examples/rua/reach-train --agent noop

worldline report runs/ -o reports/rua-smoke.json
worldline distill runs/ --output-dir datasets/rua-v1
worldline train examples/rua/sft.toml --dry-run
```

Run 包含 `task.json`、`run.json`、`trace.jsonl`、`worker.log`、`agent/trajectory.json`、`verifier/final.json` 和 `result.normalized.json`。参考策略的反馈来自实际 `mj_step`；verifier 检查最终末端距离与关节速度，空操作、曾到达后又离开都不能保留成功分。

train 和 test fixture 有不同目标、种子和分组。它们仅证明管线与 held-out 保护生效，不构成训练泛化实验。参考策略轨迹可用于检查 loader，不能作为教师模型性能证据。

## 接入开源或远程模型

准备支持 chat completions 和 function/tool calling 的模型服务：

```bash
export WORLDLINE_API_KEY='<your-key>'
worldline rollout examples/rua/reach-train \
  --agent openai-compatible --model '<served-model-name>' \
  --base-url http://localhost:8000/v1
```

API key 从环境变量读取，不写入配置、命令记录或轨迹。可用 `--api-key-env` 选择已有变量。本版同步 API 不自动重试；服务错误属于基础设施错误。模型看到动作 Schema、单位、坐标系、初始观测及每次实际反馈，`finish` 只结束尝试。

MuJoCo 默认提供关节、末端和目标的公开状态，声明 `privileged_state = true`。需要 RGB 时复制任务到自己的原生实验目录，把 `render` 改为 true；无 GPU Linux 可安装 OSMesa 并设置：

```bash
MUJOCO_GL=osmesa worldline rollout path/to/rgb-task --agent reference
```

RGB 进入原生轨迹时保留为多模态消息，可用于支持图像的模型服务；当前两连杆示例同时保留状态，不能声称 image-only 评测。

## 接入现有 RUA/EFE 资产

对现有 checkout 或 task handoff 包做只读索引：

```bash
worldline rua-catalog /path/to/embodied-first-exam \
  --benchmark robotwin-2 --mode standard -o datasets/efe-robotwin-index.json

worldline rua-catalog /path/to/rua-training-handoff \
  --split train -o datasets/rua-training-index.json

worldline eval-catalog datasets/efe-robotwin-index.json \
  -a codex -m '<harbor-model>' --max-workers 1 --dry-run
```

索引读取现有 `task.toml`、指令、DIGEST，以及可选的 `portable-catalog.json` 中的 split、eligible_tasks 和 readiness；不复制环境、参考答案、公司请求配置或发布凭据。任务路径由发现结果保存，执行前重新检查 Task ID 和声明的 DIGEST。DIGEST 内容真实性继续由 EFE 的规范检查和执行层负责。

删去 `--dry-run` 会启动 Harbor；正式 EFE runs 继续使用原生 Codex/Claude Code、官方 gateway、egress 与资源规则。参考 RUA 工作区已停用本地模拟器 rollout，Worldline 的目录索引不恢复旧 runner，也不自动向公司平台提交任务。

现有 `rua-robot/1.0` 服务已由环境拥有者在允许的节点启动时，可以连接现有 socket；原生 manifest 设置：

```toml
schema_version = "worldline.task/0.1"

[task]
id = "rua/libero-session"
domain = "rua"
environment = "rua.socket"
split = "train"
seed = 0

[environment]
socket_path = "/path/to/trusted/agent-independent-simulator.sock"
image_root = "/path/to/trusted/simulator-artifacts"
source_revision = "pinned-simulator-and-task-revision"

[evaluation]
timeout_sec = 1800
max_actions = 100
max_turns = 100
reset_budget = 0
```

`instruction.md` 与此会话任务保持一致，使用者负责固定任务、种子和环境版本并为每次 rollout 创建新会话。adapter 从可信图片根目录取图并校验 SHA256，仅转发公开观测；动作是 `{"actions": [[...], ...]}`，仍遵循源环境 limits/units。模型不能调用 `evaluate`，独立 worker 在结束后请求原生 success 判据。源协议不支持 reset，也不提供快照；Worldline 不伪装这些能力。外部 simulator 的创建与销毁归其拥有者。

## 训练与训练后复评

```bash
uv pip install -e '.[training]'
worldline train examples/rua/sft.toml
```

示例模型是可替换的公开小模型；实际运行会下载模型并更新权重。先用 `--dry-run` 检查输入和 manifest。训练器在独立进程运行，保留工具消息，以 tokenizer 的 chat template 构造文本监督，只对 assistant tokens 计算 loss。训练轨迹中被截断后没有 assistant tokens、空数据和不兼容 chat template 会报错。

本版内置后端是文本 SFT。含图像的轨迹会明确报错，需接入支持对应 VLM processor 的外部训练器；不会把图片伪装成文字监督。偏好/RL JSONL 已可导出，但内置后端尚不包含 DPO、在线 GRPO 或 verl 环境桥。

外部训练配置示例：

```toml
schema_version = "worldline.training/0.1"
[trainer]
backend = "command"
train_file = "data/train.jsonl"
validation_file = "data/validation.jsonl"
output_dir = "models/student"
command = ["python", "my_vlm_trainer.py", "--train", "{train_file}", "--output", "{output_dir}"]
```

路径相对于训练配置解析。训练 manifest 固化数据哈希、参数、命令和退出状态；非空模型输出目录会被拒绝以避免串用旧产物。训练结束后用模型服务加载 checkpoint，再通过相同 rollout 入口在 `split = "test"` 的任务上评测。test 结果用于报告，默认不流入 distill。

## 开发验证

```bash
python3 -m unittest discover -s tests -v
ruff check worldline tests
ruff format --check worldline tests
```

真实 MuJoCo 测试需要 `rua` extra；核心契约、目录适配、数据和训练计划测试只需要 Python 3.12。
GitHub Actions 配置保存在 [CI 模板](ci/worldline.yml)，安装 MuJoCo 后运行全部平台测试和 Ruff。
具备 workflow 写权限的维护者可以复制到 `.github/workflows/worldline.yml` 启用；本次推送凭据缺少该 scope，
因此保留模板，不注册自动 workflow。正式 GPU benchmark 和模型训练性能需在对应资源节点另行验收。
