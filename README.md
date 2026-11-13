# Worldline

Worldline 为需要有状态环境的 Agent 提供评测、教师 rollout、轨迹蒸馏和训练数据交付。RUA 是首个 Domain；CUA 已接入真实 BrowserGym 浏览器。环境、Agent Harness、可信验证器和外部训练器通过独立接口协作。

目前是研发框架，支持本地运行、主机多进程资源协调和可选中央任务 lease。正式机器人 benchmark、桌面 VM、GPU 与真实模型/训练作业的最终验收由部署者在对应资源执行。

0.5 版加入 owned RUA/OSWorld provider、独占会话身份、主机资源锁、中央 lease、硬 deadline HTTP、完整 interaction context 预算、磁盘数据索引、LF prose/tool 保真 formatter、在线 episode service/verl tool adapter、受控独立 fork 和可执行验收计划。EFE 保持可选上游 submodule。实现和验证边界见 [本轮复审](docs/worldline/iteration-05.md)。

## 目录和依赖

```text
worldline/
  contracts.py, tasks.py, plugins.py     # 通用契约、任务、插件发现
  runtime/                              # 环境进程、交互 session、状态恢复、批次与资源
  agents/                               # 模型调用、工具 Harness
  domains/rua/, domains/cua/             # 领域实现与动作语义
  adapters/                             # EFE/Harbor、RUA socket、BrowserGym、ATIF
  learning/formats/                     # 公共消息校验、LLaMA-Factory、verl 格式
  learning/                             # 交付编排、独立训练作业
  datasets.py, distill.py                # 数据准入、配对、切分、蒸馏
  trace.py, results.py, report.py        # 轨迹、可信结果、成本和比较
  storage.py, media.py                   # 原子存储、图片去重与校验
examples/rua/, examples/cua/             # 小型可运行任务
tests/                                 # 平台与集成回归测试
docs/worldline/                         # 框架文档
third_party/embodied-first-exam/         # 可选上游 submodule，保留其内部完整布局
```

[Embodied First Exam](https://github.com/mll-lab-nu/embodied-first-exam) 是独立上游项目，由适配器接入。它的任务、镜像、脚本、Python 服务和协议都留在自己的 checkout 内。Worldline 的核心 wheel 不包含 EFE、Harbor import hook、模拟器或训练库；安装和数据处理不要求初始化 submodule。详见 [上游接入与版本](docs/worldline/upstreams.md) 和 [模块分工](docs/worldline/architecture.md)。

## 安装和快速验证

需要 Python 3.12–3.14。按用途安装 extras：`rua`、`cua`、`harbor`、`handoff`、`training`、`dev`。

```bash
uv venv --python 3.12
uv pip install -e '.[rua,cua,handoff,dev]'
source .venv/bin/activate
python -m playwright install chromium
worldline doctor
worldline doctor --backend rua
worldline doctor --backend cua
```

`doctor` 默认检查核心是否可用；native 检查表示依赖可用，实际环境启动仍以 rollout 为准。`doctor --backend harbor` 还检查容器 daemon 和 Compose，单有 Docker 可执行文件不会被判为可运行。

真实 RUA 小环境：

```bash
worldline rollout examples/rua/reach-train -a reference --batch-id rua-v1
worldline rollout examples/rua/reach-test -a reference --batch-id rua-test
worldline rollout examples/rua/reach-train -a noop --batch-id rua-failure
worldline distill runs/ --output-dir datasets/rua-v1
```

真实 CUA 浏览器小环境：

```bash
worldline rollout examples/cua/form-train examples/cua/form-test \
  -a cua-reference --max-workers 2 --batch-id cua-v1
```

参考策略用于检查平台链路。接入模型时使用工具 Agent；模型服务由用户选择的 serving 框架运行：

```bash
export WORLDLINE_API_KEY='your-key'
worldline rollout examples/rua/reach-train \
  -a openai-compatible -m your-model --base-url http://localhost:8000/v1 \
  --max-workers 2 --cpu-budget 4 --batch-id model-v1
```

动作由环境的 JSON Schema 定义。模型只能观察、执行动作、在预算内 reset 和 finish，最终奖励来自运行器的可信 verifier 通道。RUA socket、多模态模型及正式 benchmark 入口见 [RUA 指南](docs/worldline/rua.md)；浏览器运行设置和源 benchmark 适配见 [CUA 指南](docs/worldline/cua.md)。

## 批次与恢复

```bash
worldline batch-status model-v1
worldline batch-stop model-v1
worldline rollout examples/rua/reach-train \
  -a openai-compatible -m your-model --base-url http://localhost:8000/v1 \
  --max-workers 2 --cpu-budget 4 --batch-id model-v1 --resume
```

`batch-stop` 停止领取新 trial，让已经运行的 trial 按自身预算完成。待运行任务保留，恢复不消耗它们的 attempt。SQLite/WAL 保存领取、attempt 和结果；同一批次仅有一个 owner。恢复会校验任务/配置/Worldline 版本及已完成结果，缺失结果必须先修复，不能免费重跑。

模型暂时故障在原上下文中重试；永久认证/endpoint 错误阻止后续调用。仅可恢复环境的暂时基础设施故障允许一次新环境重试，旧 attempt 和用量保留。Agent 失败不重试。CPU/GPU 参数是 admission 和独占设备 lease，硬资源限制仍由部署/容器层执行。批次控制同时支持 `rollout`、`eval-many` 和 `eval-catalog`；Harbor 的基础设施重试仍由上游 runner 负责，不能确认结束的任务不会自动重跑。

## 蒸馏和训练交付

```bash
worldline report runs/ -o reports/evaluation.json
worldline distill runs/ --output-dir datasets/experiment-v1
worldline validate-dataset \
  datasets/experiment-v1/splits/sft/train.jsonl \
  datasets/experiment-v1/splits/sft/validation.jsonl \
  datasets/experiment-v1/splits/sft/test.jsonl

worldline training-handoff datasets/experiment-v1/splits/sft/train.jsonl \
  --framework llama-factory --output-dir datasets/lf-package \
  --model /path/to/model --template qwen2_vl
worldline training-handoff datasets/experiment-v1/splits/sft/train.jsonl \
  --framework verl --output-dir datasets/verl-package
```

蒸馏输出报告、SFT/preference/RL JSONL、分组划分和 manifest。基础设施错误、被替代 attempt 与显式 held-out 轨迹不会进入训练导出；样本保存实际 split，训练入口拒收 test/evaluation/validation。未声明 split 的历史轨迹默认排除，审核后才可显式使用 `--allow-unspecified-split`。ATIF 图片必须归档在 trial 的可信目录内，缺图或不支持的模态会计入拒收，不能把图片文件名当成视觉输入。

LLaMA-Factory 包包含 ShareGPT、图片、dataset_info、原生模板的 prose/tool formatter entry 和可修改配置；verl 包包含 SFT Parquet、图片字节、薄数据适配器和初始任务 prompt。两者都附 canonical messages 和实际 trainer loader-check。用户在**独立训练环境**按包内说明运行。在线 RL 使用新的环境 rollout 和可信 verifier reward；历史 reward 无法评判新动作。完整说明见 [训练交付](docs/worldline/training-handoff.md) 和 [最终验收](docs/worldline/acceptance.md)。

完整数据包在导出和校验成功后发布；失败产物保留在单独的 `.failed-*` 目录，并在错误中给出位置。非空输出目录会拒绝覆盖。数据文件哈希使用流式读取，媒体按内容寻址去重。distill 默认输出媒体引用，JSONL/Parquet 有字节预算，可配置 `--parquet-batch-mb` 和 `--max-record-mb`。

## 外部 EFE / Harbor 任务

```bash
uv pip install -e '.[harbor]'
git submodule update --init third_party/embodied-first-exam
worldline upstream-info
worldline efe run --core --list
worldline inspect third_party/embodied-first-exam/tasks/<benchmark>/<task>
worldline eval third_party/embodied-first-exam/tasks/<benchmark>/<task> -a oracle --dry-run
```

也可以设置 `WORLDLINE_EFE_ROOT=/path/to/external/embodied-first-exam`，无需下载仓库中的 submodule。上游执行依赖由 `harbor` extra 或独立 EFE 环境提供；可用 `worldline efe --python /path/to/efe-env/bin/python run ...` 指定后者。目录索引接受 EFE 或 RUA 的外部 task handoff：

```bash
worldline rua-catalog /path/to/rua-handoff --split train -o datasets/rua-index.json
worldline eval-catalog datasets/rua-index.json -a codex -m your-model --dry-run
worldline ingest /path/to/harbor/trial --task-dir /path/to/reviewed/source/task
```

上游任务内部路径和 DIGEST 保持原样。正式 EFE runs 仍使用其原生 Agent、隔离 verifier、网络和资源协议，Worldline 原生示例独立报告。

## 开发

```bash
python3 -m unittest discover -s tests -v
ruff check worldline tests
ruff format --check worldline tests
```

参见 [贡献指南](CONTRIBUTING.md)。GitHub Actions 配置保存在 [CI 模板](docs/worldline/ci/worldline.yml)，目前尚未注册为自动 workflow。

## 视觉 RUA 和外部训练循环

```bash
worldline rollout examples/rua/reach-visual-cpu -a reference --batch-id visual-rua
# 有真实 GPU 时可选此示例；设备池是 admission，不会提供不存在的 GPU
worldline rollout examples/rua/reach-visual-train -a reference --gpu-devices 0 --batch-id gpu-rua
worldline rollout examples/cua/form-train -a openai-compatible -m your-model \
  --context-images 8 --batch-id visual-model
worldline profile-data --output-dir reports/data-profile
```

模型请求的图像预算是显式 Harness 配置，完整数据记录仍保留全部图片。`NativeTrainingSession` 可由外部训练器驱动新动作并取 verifier reward；任务需明确 train，超时自动回收。MuJoCo fixture 的 checkpoint/restore 使用单独声明的恢复预算，模型没有该管理工具。详见 [Live training](docs/worldline/live-training.md)。

本地提交与推送检查可启用：

```bash
git config core.hooksPath .githooks
python -m worldline.development check
```

本仓库使用配置的人类作者身份，不添加 AI 作者或 AI 共同作者标记。检查缓存只匹配同一源码/环境并短期有效。

## 环境供应、协调与最终验收

`rua.managed` 管理每个 trial 的模拟器启动和回收；`cua.desktop` 接入独立 OSWorld controller。预先启动的 socket 需要 session descriptor 和独占领取，legacy 兼容必须显式声明。配置与 Docker 硬限制见 [环境供应](docs/worldline/provisioning.md)。

```bash
worldline check-plugin examples/rua/reach-train --output reports/rua-contract.json
worldline rollout examples/rua/reach-train -a openai-compatible -m MODEL \
  --context-images 4 --context-mb 8 --max-output-tokens 2048 \
  --endpoint-concurrency 4 --endpoint-rpm 0 --resource-scope host
worldline experiment examples/experiments/acceptance.toml --dry-run
```

实际运行先填写计划中的模型、版本、任务与资源。该计划分开教师数据交付和训练后复评，不假定外部训练已经完成。跨运行器可配置 `--coordinator-url`；未知状态过期保留 orphaned。说明见 [协调](docs/worldline/distributed.md)、[验收入口](docs/worldline/acceptance.md)。
