# 最终模型与部署验收

平台自动检查由开发阶段完成。这里提供真实模型、正式环境、训练框架和硬件的最终验收入口，部署者使用自己的资源执行，结果必须来自实际日志和可信 verifier。

## 先检查配置，再执行模型

复制 `examples/experiments/acceptance.toml`，填入模型/serving revision、endpoint、输出目录和任务集合。现成计划只使用小型集成环境；正式 RUA/CUA 成绩必须换成对应 benchmark 的 reviewed 任务，保持源种子、split、协议和资源要求。

```bash
worldline experiment YOUR_PLAN.toml --dry-run
worldline experiment YOUR_PLAN.toml
worldline experiment YOUR_PLAN.toml --resume
```

dry-run 的 `.planned` 状态不影响随后实际运行。恢复要求原 plan 和任务/工厂版本一致；完成结果与已有数据包要通过 hash 验证。失败记录 active_phase/stage/error，未完成发布的目录保留为 `.failed-*`。同一输出目录只有一个 experiment owner。

需要核对真实成功/失败、多个 seed、重试前后的用量、停止领取与恢复、图片完整性以及 held-out 排除。task failure 不作为 infrastructure retry；未知的远端任务状态不能免费重新执行。report 保留配置分组和置信区间，环境/model request 耗时在 metrics/usage 中按阶段记录，不把导出 I/O 基准当模型吞吐。

## 独立训练框架

LF/verl 数据包都含 `manifest.json`、portable media 和 canonical `data/*-worldline.jsonl`。在对应框架的独立 Python 环境、包根目录运行：

```bash
python adapters/check_package.py . --framework llama-factory \
  --model /local/student --template qwen2_vl --expected-version YOUR_LF_VERSION --output loader-check.json
python adapters/llama_factory_entry.py llama_factory_sft.yaml
```

LF 默认 preserve 同一 assistant 消息中的文字和 function calls。包内 entry 复制原生模板并替换 formatter，保留原生 assistant header、工具语法与结束符。omit/reject 可显式选择，omit 数量被报告；严格 loader-check 拒绝 lossy 包。

```bash
python adapters/check_package.py . --framework verl \
  --model /local/student --expected-version YOUR_VERL_VERSION --output loader-check.json
```

verl 的 `verl_data.yaml` 合并进实际版本的 trainer 配置。loader-check 调用真实 framework dataset/collator，检查有效 assistant supervision 和 multimodal inputs；它明确记录 weight_update_tested=false。短训练、loss、checkpoint 保存/加载及正式 PPO/GRPO 的验证由实际作业完成，不由格式 roundtrip 代替。

需要统一记录框架版本、tokenizer/processor、模板、truncate 配置、response mask/logprob 与数据 hash。训练完成后通过 serving 加载新 checkpoint，核验 revision，再复制并填写 `examples/experiments/student-evaluation.toml` 进行显式 test 任务复评。教师交付计划不会在外部训练尚未执行时假定 student 已训练。

## 在线多轮训练

```bash
python -m worldline.learning.service examples/deployment/training-service.toml
```

trainer 用 `EpisodeClient.call("begin", task_key="reach")` 为每个 policy sample 创建独立 train-only episode，保存返回 episode_id、spec 和初始 observation。逐轮 `advance` 必须给 request_id；重复 ID 返回已保存结果，内容不同或上次状态不明则拒绝再次执行。结束后由可信 trainer 调 `finish` 获取 valid/reward。valid=false 的样本从 advantage/loss 中排除，不能填成 0。

可选 `worldline.learning.verl_tools.WorldlineEnvironmentTool` 适配 verl BaseTool。每次 create_kwargs 都传同一个 episode_id；release 只释放短 tool handle。最终 verifier reward 另从服务取，模型工具不暴露它。完整 policy token/logprob/response mask 由 verl AgentLoop 保存；该薄工具 adapter 的 Worldline trace 是结构化 action view，SFT 教师全消息请用直接 sampler 的 advance 接口。

外部 sampler 也可直接用 `NativeTrainingSession`，无需服务/verl。checkpoint 与 fork 只在实现该能力的环境开放；fork quota 跨树共享，原始截止时间和累计预算不回退。其他环境不会通过 reload 或 replay 冒充 snapshot。

## 自动 CI 注册

`docs/worldline/ci/worldline.yml` 是可执行的 Python 3.12/3.13 模板，包含平台测试、Ruff、contract check 和构建。当前 GitHub 发布 token 没有 workflow scope，模板尚未注册。具备对应权限后放入 `.github/workflows/worldline.yml` 即可；本地提交/推送检查继续有效。Docker/GPU、正式 benchmark、OSWorld VM 和具体 trainer 作业应接到对应资源的验收 runner。
