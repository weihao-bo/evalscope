# 自定义多模态评测指南

本指南结合仓库架构（见 `docs/OVERVIEW.md`）和官方文档，说明如何将自有多模态数据集转换为 EvalScope 支持的评测集，并给出可直接运行的示例项目。

## 1. 数据格式与适配器

- **通用问答（General-VQA，生成式）**
  - 文件：JSONL/TSV，字段 `messages`（OpenAI ChatCompletion 多模态格式）、可选 `answer`。
  - `messages` 元素类型：
    - 文本：`{"type": "text", "text": "..."}`
    - 图片：`{"type": "image_url", "image_url": {"url": "<本地/远程/Base64>"}}`
    - 支持 system/user/assistant，多图片与系统提示。
  - 参考：`custom_eval/multimodal/vqa/example_openai.jsonl`；适配器 `general_vqa`（BLEU、ROUGE）。

- **通用选择题（General-VMCQ，多选一）**
  - 文件：JSONL/TSV，字段 `question`（可含 `<image x>`）、`options`（字符串列表或其 JSON 字符串）、`image_1`…`image_100`、`answer`（大写字母）。
  - 图片直接写字符串（本地/URL/Base64），不包 `{"url": ...}`。
  - 参考：`custom_eval/multimodal/mcq/example.jsonl`；适配器 `general_vmcq`（准确率）。

- **子集与拆分**
  - 多文件时，以文件名（无扩展名）作为子集名，在 `dataset_args.{benchmark}.subset_list` 声明。
  - `general_vmcq` 的 `dev/val` 分别对应训练/评测拆分。

- **额外字段**
  - 当前适配器仅消费核心字段。若需利用自定义标签（如难度、场景），可继承适配器并覆盖 `record_to_sample` / `aggregate_scores` / `generate_report`，将标签写入 `Sample.metadata` 并在聚合中输出。

- **指标与裁判模型**
  - `general_vqa` 默认 BLEU/ROUGE，若需准确率可配置裁判模型：`judge_model_args` + `judge_worker_num` + `judge_strategy=llm`。
  - `general_vmcq` 默认 `acc`。需更多指标可注册自定义 Metric 并放入基准 `metric_list`。

## 2. 运行配置要点

核心入口 `run_task` 支持字典、YAML、CLI。关键参数来自 `TaskConfig`（见 `evalscope/config.py`）：

- 模型：`model`、`api_url`、`api_key`、`generation_config`（温度、max_tokens 等）。
- 数据集：`datasets`、`dataset_args`、`dataset_dir`、`limit`。
- 评测：`eval_type`（如 `openai_api`）、`eval_backend`（本例使用 `Native`）、`eval_batch_size`。
- 裁判：`judge_model_args`、`judge_worker_num`、`judge_strategy`、`analysis_report`。
- 输出：`work_dir`、`use_cache`、`rerun_review`。

## 3. 示例项目（在 `examples/custom_vlm_demo/`）

内容见示例目录，结构与用途：

- `configs/task_vlm.yaml`：统一管理模型、数据集、裁判与运行参数，支持环境变量占位。
- `run.py`：读取 `.env` 注入占位后调用 `run_task`。
- `.env.example`：示意需要的密钥键名（不含真实值）。
- `README.md`：运行说明（包含 CLI 等价命令）。

数据示例直接复用仓库已有文件：

- VQA：`custom_eval/multimodal/vqa/example_openai.jsonl`
- MCQ：`custom_eval/multimodal/mcq/example.jsonl`

## 4. 结果与可视化

- 运行后在 `work_dir` 下生成：
  - `predictions/<model>/<subset>.jsonl`：模型输出。
  - `reviews/<model>/<subset>.jsonl`：裁判或指标评分。
  - `reports/<model>/<dataset>.json`：聚合指标（可自定义生成器）。
- 内置可视化 WebUI：
  ```
  evalscope app --outputs <work_dir> --lang zh --server-port 7860 --server-name 0.0.0.0 --allowed-paths <work_dir>
  ```
  兼容自定义评测集，只要保持上述输出目录结构即可。

## 5. 隐私与密钥管理

- 不将密钥写入仓库，使用 `.env`（加入 `.gitignore`）或外部密钥管理。
- 在 YAML 中用占位符（如 `${MAIN_API_URL}`）并在运行脚本中注入。
- 裁判模型与主模型的 base URL、API Key 分开管理，分别注入 `api_url` / `api_key` 与 `judge_model_args`。

## 6. 快速检查清单

- [ ] VQA/MCQ 数据按规范准备，图片路径可访问。
- [ ] `dataset_args` 指向正确目录与子集名。
- [ ] `eval_type`、`eval_backend` 与模型后端匹配。
- [ ] 主模型与裁判模型的 `api_url`/`api_key` 均已注入。
- [ ] `work_dir` 可写，输出用于可视化或后处理。


