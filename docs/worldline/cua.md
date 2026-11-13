# CUA：BrowserGym 接入

CUA 使用与 RUA 相同的 Task、独立 worker、工具 Agent、可信结果、媒体存储、批次队列和训练交付。BrowserGym/Chromium 覆盖浏览器方向；0.5 增加 `cua.desktop` 与独立 OSWorld controller provider，配置见 [环境供应](provisioning.md)。桌面线协议已有自动检查，真实 VM 的最终验收由对应部署执行。

## 安装与本地检验

```bash
uv pip install -e '.[cua,handoff]'
# BrowserGym 0.14.3 固定 Playwright 1.44；使用当前官方 CDN 取得其匹配的 Chromium。
PLAYWRIGHT_DOWNLOAD_HOST=https://cdn.playwright.dev/dbazure/download/playwright \
  python -m playwright install chromium

worldline rollout examples/cua/form-train examples/cua/form-test \
  --agent reference --batch-id cua-form-v1 --max-workers 1
worldline rollout examples/cua/form-train --agent noop
```

服务器端检查表单是否正确提交；模型文字、页面自行显示的“成功”和代码退出都不替代 native verdict。form fixture 验证接入与数据链，不是正式 WebArena/MiniWoB 分数。示例同时提供 screenshot 和 AXTree/BID，不能称 image-only 评测。

如果已有经过验证的 Chromium，可用 `WORLDLINE_CHROMIUM_PATH` 指定可执行文件。实际浏览器版本会纳入有效环境配置，避免把不同版本结果误认为相同配置。旧版 Playwright 默认下载端点在部分网络上不可用；使用上述官方 CDN，并先确认 browser 安装完成。

模型入口沿用：

```bash
worldline rollout examples/cua/form-train -a openai-compatible \
  -m '<served-model>' --base-url http://localhost:8000/v1 \
  --model-retries 2 --batch-id cua-model-v1
```

公开工具动作包括 BID click/fill/press、viewport mouse_click、scroll、keyboard_type 和用户消息提交；允许导航时额外暴露 goto/go_back。参数经过明确校验并编码为 BrowserGym 高层调用，模型没有直接执行任意 Python/JavaScript 的工具。

## 已有 BrowserGym benchmark

原生任务 manifest 的 `task.environment = "cua.browsergym"`，环境配置例如：

```toml
[environment]
gym_id = "browsergym/miniwob.click-test"
benchmark_modules = ["browsergym.miniwob"]
render = true
allow_navigation = false
headless_chat = false
```

先在允许的节点独立安装对应 benchmark 包和资产/网站，设置其官方环境变量；Worldline 不自动下载 benchmark 或部署外部网站。任务的 instruction、seed、版本、split/group 与源 benchmark 保持一致。实际 goal、聊天更新、AXTree 和截图通过公开观察传给 Agent；step 和最终判分复用 BrowserGym。

外部 benchmark 默认保留 BrowserGym 聊天 UI。非交互实验可明确 `headless_chat = true`，采用内存消息通道减少一个 Chromium 实例；这个模式仍传递 task.validate 需要的消息，但不能录制聊天 UI 视频或等待真实用户聊天。模式和真实浏览器版本纳入比较配置。

本地 fixture 默认 `cpu_headless = true`：使用 Chromium 的现代 headless 模式并关闭 GPU/软件 3D rasterizer。
本机默认 Chromium 启动组合曾阻塞在 `new_page()`，该 CPU 模式已通过真实单任务与并发验证。
外部 benchmark 默认 `cpu_headless = false` 保持原启动设置；需要 WebGL/GPU 的任务也应保留或配置合适的运行模式。
选择 CPU 模式时可显式设置该字段，实际模式进入比较配置。浏览器缓存宜放在已准备的本地存储，
例如安装和运行时统一设置 `PLAYWRIGHT_BROWSERS_PATH=/path/to/local/browser-cache`。

## 恢复与数据

```bash
worldline rollout examples/cua/form-train examples/cua/form-test \
  --agent reference --batch-id cua-form-v1 --resume --max-workers 1
worldline distill runs/ --output-dir datasets/cua-v1
worldline training-handoff datasets/cua-v1/splits/sft/train.jsonl \
  --framework llama-factory --template qwen2_vl --output-dir datasets/cua-lf
```

resume 必须沿用任务、模型/Agent 与 Harness 配置；完成的 trial 不会再次执行。运行数据建议放入独立实验 runs 目录，避免把不同方法的历史试次混在一次报告里。CPU/GPU admission、重试和有效 attempt 的规则见 [基础设施复审](infra-review.md)，训练格式见 [训练数据交付](training-handoff.md)。
