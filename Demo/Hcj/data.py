# 本文件负责读取和过滤 Hcj 数据，并加载公共成分特征。
"""Hcj 数据入口。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from Demo.common.data import add_features, add_log, normalize_table, read_raw

SOURCE = "dataset/NEMAD/processed/hcj_cleaned.csv"

TASK = {
    "name": "Hcj",
    "task_cn": "内禀矫顽力",
    "unit": "kA/m",
    "transform": "log10_1p",
    "cv_splits": 5,
    "early_stopping_rounds": 60,
    "params": {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "max_depth": 2,
        "learning_rate": 0.03,
        "min_child_weight": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 3.0,
        "reg_alpha": 0.5,
        "n_estimators": 600,
    },
}


def load_data() -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """返回清洗后的 Hcj 表、处理日志和重复标签冲突表。"""
    raw = read_raw(SOURCE)
    logs: list[dict] = []
    before = len(raw)
    raw["Hcj_kA_m"] = pd.to_numeric(raw["Hcj_kA_m"], errors="coerce")
    filtered = raw[
        raw["hcj_value_class"].eq("plausible_hcj")
        & raw["hcj_flag"].eq("ok")
        & raw["Hcj_kA_m"].notna()
        & np.isfinite(raw["Hcj_kA_m"])
        & raw["Hcj_kA_m"].ge(0)
    ].copy()
    add_log(
        logs,
        "quality_filter",
        before,
        len(filtered),
        "只保留 plausible_hcj、hcj_flag=ok 且 Hcj 非负的记录",
    )
    table, conflicts = normalize_table(
        filtered, formula_col="material_name", value_col="Hcj_kA_m", logs=logs
    )
    return add_features(table), logs, conflicts
