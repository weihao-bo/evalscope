# Repository Guidelines

## 项目结构与模块组织
- `evalscope/`：核心库与 API，涵盖基准注册、评测后端、指标实现等。
- `tests/`：单测按场景分目录（如 `benchmark/`, `cli/`, `perf/`, `rag/`, `vlm/`），可使用 `test_run_all.py` 统一触发。
- `examples/`：示例脚本与配置，便于快速参考运行方式。
- `docs/`：Sphinx 文档源文件（`docs/en`, `docs/zh`），`docs/scripts` 包含文档生成辅助脚本。
- `requirements/`：分模块的可选依赖声明（如 `opencompass.txt`, `perf.txt`, `vlmeval.txt` 等）。
- 根目录工具：`Makefile` 提供安装、开发、文档、lint 任务；`setup.cfg` 定义 isort/yapf/flake8 规范。

## 构建、测试与开发命令
- `make install`：以 editable 模式安装核心依赖，便于本地开发调试。
- `make dev`：安装开发、性能与文档所需的扩展依赖，并安装 pre-commit 钩子。
- `make lint`：运行 isort + yapf + flake8 的预提交检查，需在提交前保持通过。
- `TEST_LEVEL_LIST=0,1 python -m unittest discover tests`：运行主要单测集；可在根目录直接执行或使用 `python tests/test_run_all.py` 触发相同命令。
- `make docs`：生成中英文文档（调用 `docs-en` 与 `docs-zh`），需先满足文档依赖。

## 代码风格与命名约定
- Python 3.10+，缩进 4 空格，单行长度 120 字符。
- 统一通过 isort 排序导入，yapf 进行格式化，flake8 进行静态检查（配置见 `setup.cfg`）。
- 模块与文件用 `snake_case`，类名使用 `CapWords`，常量大写，下划线分隔。
- 提交前请运行 `pre-commit run --all-files` 确保格式与静态检查一致。

## 测试指南
- 首选 `unittest` 框架，测试文件放置于 `tests/<模块>/test_*.py`。
- 使用 `TEST_LEVEL_LIST` 控制测试范围（默认 `0,1` 覆盖核心用例）；新增用例尽量保持可在此级别运行。
- 新增特性需至少覆盖主要路径与异常分支，并复用现有基准/数据样例以减少重复构造。

## 提交与 Pull Request 指南
- 提交信息遵循仓库常见格式：`[Feature] ...`, `[Fix] ...`，末尾可附 PR 编号或问题引用，如 `[#1051]`。
- PR 描述应包含：变更摘要、测试结果（命令及结论）、影响范围或兼容性说明；涉及界面或报告的变更建议附截图或样例输出。
- 在创建 PR 前，请确保本地通过 `make lint` 与主要测试命令，并确认未引入多余依赖或大型文件。
