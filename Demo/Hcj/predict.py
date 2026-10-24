# 本脚本用已训练好的 Hcj 模型预测化学式，并给出 Ha 上界约束与 ME 合理性校验。
"""用法：python Demo/Hcj/predict.py Nd2Fe14B"""

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
from Demo.Hcj.bound import attach_bound, summarize


def main() -> int:
    parser = argparse.ArgumentParser(description="预测内禀矫顽力 Hcj")
    parser.add_argument("formulas", nargs="+", help="一个或多个化学式")
    args = parser.parse_args()

    model_path = Path(__file__).resolve().parent / "output" / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("请先运行 python Demo/Hcj/train.py")
    package = joblib.load(model_path)

    formulas = [formula.strip() for formula in args.formulas]
    features = add_features(pd.DataFrame({"formula": formulas}))
    result = predict_package(package, features)
    result = attach_bound(result, formulas)
    result["unit"] = "kA/m"
    show = result[
        [
            "formula",
            "pred",
            "pred_bounded",
            "lower_80",
            "upper_80",
            "ha_kA_m",
            "me",
            "me_status",
            "in_domain",
        ]
    ].round({"pred": 1, "pred_bounded": 1, "lower_80": 1, "upper_80": 1, "ha_kA_m": 1, "me": 3})
    print(show.to_string(index=False))
    summary = summarize(result)
    print(
        f"校验：{summary['n_checked']}/{summary['n_total']} 个配方有 Ha；"
        f"越界 {summary['n_violates_upper_bound']} 个，无 Ha {summary['n_no_ha']} 个"
    )
    if summary["n_checked"]:
        print(
            f"ME 范围 {summary['me_min']:.3f}~{summary['me_max']:.3f}，"
            f"中位 {summary['me_median']:.3f}"
        )
    print(
        "说明：ME = Hcj/Ha 为工艺效率系数，物理上应落在 (0,1)；"
        "pred_bounded 为 Ha 上界约束后的值，越界时取 Ha。"
    )
    print("Hcj 为粗筛预测，不是实验实测；无 Ha 的配方不做上界校验。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
