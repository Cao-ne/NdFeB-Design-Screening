# 本文件给 Hcj 预测加上 Ha 上界约束，并计算工艺效率系数 ME = Hcj/Ha。
"""Ha 上界约束与合理性校验。

Ha 只作为上界使用，不参与预测：
  - pred_bounded = min(pred, Ha)：有 Ha 时不允许超过上界；
  - ME = pred / Ha：工艺效率系数，物理上应落在 (0, 1)；
  - me_status 标出越界、零值和缺少 Ha 的情况，供人工判断。

数据源：dataset/shared/processed/33_ha_dataset.csv（77 个配方）。
注意：Ha 覆盖率很低，多数配方查不到 Ha，此时不校验、ME 为空。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Demo.common.data import reduced_formula  # noqa: E402

HA_PATH = ROOT / "dataset/shared/processed/33_ha_dataset.csv"

HA_COLUMNS = ["formula", "ha_kA_m", "ha_T", "n_records", "median_basis", "sources"]


def load_ha_bounds() -> pd.DataFrame:
    """读取按配方聚合的 Ha 中位数，返回约化化学式索引。"""
    ha = pd.read_csv(HA_PATH, low_memory=False)
    ha = ha.rename(columns={"ha_kA_m_median": "ha_kA_m", "ha_T_median": "ha_T"})
    return ha[HA_COLUMNS].drop_duplicates("formula").set_index("formula")


def attach_bound(prediction: pd.DataFrame, formulas: list[str]) -> pd.DataFrame:
    """给预测表补 Ha、ME、上界约束值和校验状态。

    输入 prediction 需含 pred 列（与 formulas 一一对应）。
    """
    if len(prediction) != len(formulas):
        raise ValueError("prediction 行数与 formulas 不一致")

    ha = load_ha_bounds()
    out = prediction.copy()
    out.insert(0, "formula", formulas)
    out["formula_reduced"] = out["formula"].map(reduced_formula)
    out["ha_kA_m"] = out["formula_reduced"].map(ha["ha_kA_m"])
    out["me"] = out["pred"] / out["ha_kA_m"]
    out["pred_bounded"] = out[["pred", "ha_kA_m"]].min(axis=1)
    out["bound_hit"] = out["ha_kA_m"].notna() & (out["pred"] > out["ha_kA_m"])
    out["me_status"] = [
        _status(me, has_ha) for me, has_ha in zip(out["me"], out["ha_kA_m"].notna())
    ]
    return out


def _status(me: float, has_ha: bool) -> str:
    """ME 的合理性分档；只做标记，不改预测。"""
    if not has_ha:
        return "no_ha"
    if pd.isna(me):
        return "no_ha"
    if me > 1.0:
        return "violates_upper_bound"
    if me <= 0.0:
        return "me_not_positive"
    return "within_bound"


def summarize(result: pd.DataFrame) -> dict:
    """汇总校验状态，供脚本打印或写元数据。"""
    status = result["me_status"].value_counts().to_dict()
    checked = result[result["me_status"].ne("no_ha")]
    return {
        "n_total": int(len(result)),
        "n_checked": int(len(checked)),
        "n_no_ha": int(status.get("no_ha", 0)),
        "n_violates_upper_bound": int(status.get("violates_upper_bound", 0)),
        "n_within_bound": int(status.get("within_bound", 0)),
        "me_min": float(checked["me"].min()) if len(checked) else None,
        "me_median": float(checked["me"].median()) if len(checked) else None,
        "me_max": float(checked["me"].max()) if len(checked) else None,
    }
