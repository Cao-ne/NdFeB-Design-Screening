# 本脚本在对数尺度训练 Hcj 的 XGBoost 模型，并保存指标、区间、适用域和数据日志。
"""运行：python Demo/Hcj/train.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Demo.common.data import ROOT as PROJECT_ROOT, file_hash
from Demo.common.model import train_and_save
from Demo.Hcj.data import SOURCE, TASK, load_data


def main() -> int:
    table, logs, conflicts = load_data()
    metadata = train_and_save(
        table=table,
        logs=logs,
        conflicts=conflicts,
        task_config=TASK,
        source_path=PROJECT_ROOT / SOURCE,
        source_hash=file_hash(PROJECT_ROOT / SOURCE),
        output_dir=Path(__file__).resolve().parent / "output",
    )
    print(f"模型与指标已保存；使用 {metadata['rows_used']} 个独立配方。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
