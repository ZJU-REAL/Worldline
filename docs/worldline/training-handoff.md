# 将 Worldline 数据交给训练框架

Worldline 负责环境运行、可信结果、轨迹、数据准入和格式交付。用户可以独立安装训练框架，根据说明消费数据；不需要让环境依赖 LLaMA-Factory、verl 或模型训练库。

## 准备数据

```bash
worldline distill runs/my-experiment --output-dir datasets/my-experiment
```

SFT 默认成功轨迹；被替代 attempt、基础设施错误和显式 held-out 轨迹不进入训练视图。train/validation/test 与来源分组保持隔离。先检查 `manifest.json` 的准入/排除计数；小实验可能没有 validation 数据，不能把 train 复制成 validation。

## LLaMA-Factory

```bash
worldline training-handoff datasets/my-experiment/splits/sft/train.jsonl \
  --framework llama-factory --output-dir datasets/lf-package \
  --model /path/to/your/model --template qwen2_vl
```

如果存在非空的独立 validation 数据，增加 `--validation-file path/to/validation.jsonl`。文本模型选择对应文本 template；包含图像时选择匹配 VLM 的 template/processor。

输出包含：

- `data/train.jsonl`、可选 validation：ShareGPT conversation，含 function_call 与 observation。
- `media/`：相对路径图片，图片字节经过解码与 hash 检查。
- `dataset_info.json`：字段与 role tags 映射。
- `llama_factory_sft.yaml`：可修改的 LoRA SFT 示例配置。
- `manifest.json` 和 rejected 文件：来源哈希、计数、排除原因与文件哈希。

在**独立的 LLaMA-Factory 环境**中：

```bash
cd /path/to/lf-package
python adapters/check_package.py . --framework llama-factory --model /local/model --template qwen2_vl --expected-version YOUR_LF_VERSION
python adapters/llama_factory_entry.py llama_factory_sft.yaml
```

调整模型、template、batch size、长度与训练参数。包可以整体移动；从包根目录运行，以便相对图片路径正确解析。0.5 默认 preserve 同时出现的 prose 与 function calls：包内 entry 复制原生模板并用保真 formatter 将两者放在正确的 assistant slots 内；配置中的 `worldline__` 别名防止直接绕过 entry 静默丢文字。`--tool-prose-policy reject/omit` 是显式选择，omit 数量写入 manifest，严格 loader-check 会拒绝。末尾工具输入可从 LF 监督视图裁去，对应裁剪记录在样本 metadata；canonical 原消息保存在 `data/*-worldline.jsonl`。

## verl

```bash
uv pip install -e '.[handoff]'
worldline training-handoff datasets/my-experiment/splits/sft/train.jsonl \
  --framework verl --output-dir datasets/verl-package
```

输出 `data/train.parquet` 保存多轮 messages、tools、图像 bytes 和 metadata；`adapters/verl_sft_dataset.py` 是训练环境中导入的薄 MultiTurnSFTDataset 适配器，用于恢复 JSON 编码的对话与异构工具 Schema。将 `verl_data.yaml` 合入所安装版本的 SFT 配置，提供 tokenizer、模型与图像 processor。也可直接用 `WorldlineSFTDataset` 作为该版本支持的 custom dataset class。

`*-rl.parquet` 是另一种数据：只含初始上下文、初始图像与任务身份，不包含教师动作/后续观察作为 prompt。`reward_model.ground_truth` 是任务身份，**不是可复制到新 rollout 的历史 reward**。在线 AgentLoop 可用 0.5 的 episode service 和可选 verl BaseTool，保持多轮环境生命周期并从可信 finish 通道取得新动作奖励；具体版本的 PPO/GRPO 作业仍由训练框架运行。

该 RL 文件的任务集合来自输入 SFT 样本；它不是自动构建的完整 RL 课程。若课程需要覆盖教师失败任务，应独立从训练任务定义或全部验证过的 rollout 构建任务清单，避免成功样本筛选带来的偏差。

格式接口按官方数据协议核对；本轮有 JSONL/角色/图片/Parquet 检查。完整训练仍取决于实际安装版本、模型 chat template、processor、资源和训练目标。

原生多模态轨迹可按 hash 完整恢复。0.4 的 distill 默认产物是媒体引用，整包移动后交付自动定位 root；独立数据可显式指定 `--media-root`。0.3 还支持 ATIF-v1.6+ 的 `source.media_type/path` 本地图片，在 trial 的可信目录内定位并转为真实视觉内容；不会联网补图。缺失、越界、远程或歧义图片引用，以及未支持的 audio/copy-context loss-mask 轨迹会拒收，原始评测证据保留。非标准 gateway 附件仍需先按任务声明归档。

数据切分写回每条样本的 split，训练交付拒收 test/evaluation/held_out/validation。catalog 的 split/group 和每个 Harbor trial 的任务元数据传播到导出；没有声明 split 的历史数据默认排除，审核后才可显式导入。

完整交付包在成功校验后从同级 staging 目录原子发布，不覆盖非空输出；失败目录单独保留为 `.failed-*`，错误报告其位置。训练输入及 Parquet 哈希使用流式读取，内存不会因一次整体读入数据文件而增长。转换仍需要容纳单个 episode，以及一个 Parquet batch 的图像；长视觉轨迹尚未实现逐 turn 的完全流式处理。

0.4 交付支持 `--parquet-batch-mb`（默认 8）、`--max-record-mb`（默认 64）、`--max-episode-media-mb`（默认 64）。Parquet batch 按字节控制；单条大记录可占一个单独 row group，但不能超过 record budget。JSONL 在解析前执行单行预算。只影响训练格式，不改变原始评测证据。

在线新动作评分接口见 [Live training](live-training.md)，有效 0 与不可评分结果必须区分。

0.5 的实际 framework loader/collator 检查由包内 `adapters/check_package.py` 执行；指定安装版本和本地模型，检查真实 assistant supervision 与 multimodal inputs。权重更新和训练后复评分开，参见 [最终验收](acceptance.md)。大规模去重、split assignment、ID/group 校验使用磁盘索引；仍按明确单 episode/record 字节预算拒绝超大样本，不声称完全流式解码任意长度的一条 trajectory JSON。
