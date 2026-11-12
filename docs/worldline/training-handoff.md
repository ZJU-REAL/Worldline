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
llamafactory-cli train llama_factory_sft.yaml
```

调整模型、template、batch size、长度与训练参数。包可以整体移动；从包根目录运行，以便相对图片路径正确解析。若函数调用同时带 prose，函数监督仅转换 name/arguments，原 prose 保留在原始 Worldline 轨迹，并在 manifest 统计省略次数。末尾工具输入可从 LF 监督视图裁去，对应裁剪记录在样本 metadata；原轨迹保持完整。

## verl

```bash
uv pip install -e '.[handoff]'
worldline training-handoff datasets/my-experiment/splits/sft/train.jsonl \
  --framework verl --output-dir datasets/verl-package
```

输出 `data/train.parquet` 保存多轮 messages、tools、图像 bytes 和 metadata；`adapters/verl_sft_dataset.py` 是训练环境中导入的薄 MultiTurnSFTDataset 适配器，用于恢复 JSON 编码的对话与异构工具 Schema。将 `verl_data.yaml` 合入所安装版本的 SFT 配置，提供 tokenizer、模型与图像 processor。也可直接用 `WorldlineSFTDataset` 作为该版本支持的 custom dataset class。

`*-rl.parquet` 是另一种数据：只含初始上下文、初始图像与任务身份，不包含教师动作/后续观察作为 prompt。`reward_model.ground_truth` 是任务身份，**不是可复制到新 rollout 的历史 reward**。在线 GRPO/PPO 还需要训练侧 AgentLoop 启动任务、执行新工具调用、请求独立 verifier；本版未将这些桥接器假装成可运行的默认 reward function。

该 RL 文件的任务集合来自输入 SFT 样本；它不是自动构建的完整 RL 课程。若课程需要覆盖教师失败任务，应独立从训练任务定义或全部验证过的 rollout 构建任务清单，避免成功样本筛选带来的偏差。

格式接口按官方数据协议核对；本轮有 JSONL/角色/图片/Parquet 检查。完整训练仍取决于实际安装版本、模型 chat template、processor、资源和训练目标。

原生 0.2 轨迹的图像可校验并完整恢复。历史 Harbor/ATIF 转换目前保留文本与图像引用信息，
不会自动从所有 gateway/附件格式恢复图片；这些轨迹在进行 VLM 训练前，需要核对并补齐实际图像资产。
文本引用存在并不代表已经得到可用的多模态训练样本。
