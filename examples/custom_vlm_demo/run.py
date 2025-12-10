import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from evalscope.config import TaskConfig
from evalscope.run import run_task


def fill_env_placeholders(obj):
    """递归替换 ${VAR} 占位符为环境变量。"""
    if isinstance(obj, dict):
        return {k: fill_env_placeholders(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fill_env_placeholders(v) for v in obj]
    if isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
        key = obj[2:-1]
        val = os.environ.get(key)
        if val is None:
            raise ValueError(f'环境变量 {key} 未设置，请在 .env 中提供。')
        return val
    return obj


def main():
    base = Path(__file__).parent
    # 优先加载本地 .env（未提交版本库），不存在则忽略
    load_dotenv(base / '.env')

    with open(base / 'configs/task_vlm.yaml', 'r', encoding='utf-8') as f:
        raw_cfg = yaml.safe_load(f)

    cfg = fill_env_placeholders(raw_cfg)
    run_task(TaskConfig(**cfg))


if __name__ == '__main__':
    main()

