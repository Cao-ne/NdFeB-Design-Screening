# 本脚本用已训练好的 Ms 模型预测一个或多个化学式。
"""用法：python Demo/Ms/predict.py Nd2Fe14B"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Demo.common.data import add_features
from Demo.common.model import predict_package


def main() -> int:
    parser = argparse.ArgumentParser(description="预测饱和磁化强度 Ms")
    parser.add_argument("formulas", nargs="+", help="一个或多个化学式")
    args = parser.parse_args()

    model_path = Path(__file__).resolve().parent / "output" / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("请先运行 python Demo/Ms/train.py")
    package = joblib.load(model_path)

    formulas = [formula.strip() for formula in args.formulas]
    features = add_features(pd.DataFrame({"formula": formulas}))
    result = predict_package(package, features)
    result.insert(0, "formula", formulas)
    result["unit"] = "emu/g"
    print(result.to_string(index=False))
    print("说明：以上为模型预测，非实验实测。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
