# custom_vlm_demo

一个可直接运行的多模态自定义评测示例，统一用 YAML 管理模型/数据集/裁判配置，密钥放本地 `.env`（不提交）。

## 目录
- `configs/task_vlm.yaml`：主配置，包含主模型、裁判模型、数据集与运行参数，占位 `${...}` 由脚本注入。
- `run.py`：加载 `.env` 后填充占位并调用 `run_task`。
- `env.example`：需要的环境变量示例，复制为 `.env` 后填写真实值。

## 依赖（使用 uv 管理虚拟环境，开发模式安装本仓库）
```bash
# 创建并激活虚拟环境
uv venv .venv
source .venv/bin/activate

# 开发模式安装当前仓库 + 运行脚本所需依赖
uv pip install -e .
uv pip install python-dotenv pyyaml
```

## 准备
```bash
cp env.example .env
# 编辑 .env 填入主模型与裁判模型的 API URL/KEY
```
如使用内置示例数据，无需额外下载，路径已指向 `custom_eval/multimodal/...`。

## 运行（Python，使用 uv）
```bash
uv run python examples/custom_vlm_demo/run.py
```

> 运行与配置优先级（唯一推荐方案）：`.env` 提供敏感变量，`configs/task_vlm.yaml` 提供结构化配置，`run.py` 读取 `.env` 并用其中变量替换 YAML 占位后调用 `run_task`。无 CLI 叠加参数，避免冲突。

## 自定义数据集/基准
- VQA：将自有 JSONL/TSV 放到自定义目录，`dataset_args.general_vqa.local_path` 指向该目录，并在 `subset_list` 写文件名（不含扩展名）。
- MCQ：同理，格式参考 `custom_eval/multimodal/mcq/example.jsonl`。
- 若需要新字段或新指标，继承 `general_vqa`/`general_vmcq` 适配器并在配置中更换 `datasets` 指向新基准名。

## 可视化
评测完成后，可用内置 WebUI 浏览 `outputs/custom_vlm_demo`：
```bash
evalscope app --outputs outputs/custom_vlm_demo --lang zh --server-port 7860 --server-name 0.0.0.0 --allowed-paths outputs/custom_vlm_demo
```

## 输出位置
- 预测：`outputs/custom_vlm_demo/predictions/...`
- 评分：`outputs/custom_vlm_demo/reviews/...`
- 报告：`outputs/custom_vlm_demo/reports/...`

